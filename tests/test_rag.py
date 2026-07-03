import io
import pytest
from pypdf import PdfWriter
from unittest.mock import MagicMock, patch

from rag import _chunk, _extract_pages


def _make_pdf(num_pages: int) -> bytes:
    # PdfWriter genera páginas en blanco (sin texto extraíble).
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_blank_pdf_raises():
    with pytest.raises(ValueError, match="sin texto"):
        _extract_pages(_make_pdf(2))


def test_extract_pages_accepts_file_path(tmp_path):
    # Streaming a disco: _extract_pages debe leer desde una ruta, no solo bytes.
    p = tmp_path / "blank.pdf"
    p.write_bytes(_make_pdf(2))
    with pytest.raises(ValueError, match="sin texto"):
        _extract_pages(str(p))


def test_short_text_single_chunk():
    assert _chunk("hola mundo") == ["hola mundo"]


def test_empty_text_no_chunks():
    assert _chunk("") == []
    assert _chunk("   ") == []


def test_long_text_splits_with_overlap():
    text = "a" * 2500
    chunks = _chunk(text, size=1000, overlap=150)
    assert len(chunks) == 3
    assert all(c for c in chunks)          # ninguno vacío
    assert all(len(c) <= 1000 for c in chunks)
    # el segundo chunk arranca 150 chars antes del final del primero
    assert chunks[1][:150] == chunks[0][-150:]


def test_embed_calls_cohere_and_returns_vectors(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    fake_resp = MagicMock()
    fake_resp.ok = True
    fake_resp.json.return_value = {"embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]}}
    with patch("rag.requests.post", return_value=fake_resp) as post:
        from rag import _embed
        vectors = _embed(["uno", "dos"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    # envía los textos, la API key y el input_type de indexado por defecto
    body = post.call_args.kwargs["json"]
    assert body["texts"] == ["uno", "dos"]
    assert body["input_type"] == "search_document"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_embed_uses_query_input_type(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    fake_resp = MagicMock()
    fake_resp.ok = True
    fake_resp.json.return_value = {"embeddings": {"float": [[0.0]]}}
    with patch("rag.requests.post", return_value=fake_resp) as post:
        from rag import _embed
        _embed(["pregunta"], input_type="search_query")
    assert post.call_args.kwargs["json"]["input_type"] == "search_query"


def test_embed_batches_large_input(monkeypatch):
    # Con muchos chunks, _embed trocea en lotes de EMBED_BATCH y concatena.
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    from rag import EMBED_BATCH, _embed

    n = EMBED_BATCH * 2 + 5  # obliga a 3 lotes
    texts = [f"t{i}" for i in range(n)]

    def fake_post(*args, **kwargs):
        batch = kwargs["json"]["texts"]
        r = MagicMock()
        r.ok = True
        r.json.return_value = {"embeddings": {"float": [[float(len(t))] for t in batch]}}
        return r

    with patch("rag.requests.post", side_effect=fake_post) as post:
        vectors = _embed(texts)

    assert len(vectors) == n           # todos los vectores, en orden
    assert post.call_count == 3        # 3 lotes
    # ningún lote supera EMBED_BATCH
    for call in post.call_args_list:
        assert len(call.kwargs["json"]["texts"]) <= EMBED_BATCH


def test_ensure_collection_creates_when_missing():
    fake = MagicMock()
    fake.collection_exists.return_value = False
    with patch("rag._qdrant", return_value=fake):
        from rag import ensure_collection
        ensure_collection()
    fake.create_collection.assert_called_once()


def test_ensure_collection_skips_when_present():
    fake = MagicMock()
    fake.collection_exists.return_value = True
    with patch("rag._qdrant", return_value=fake):
        from rag import ensure_collection
        ensure_collection()
    fake.create_collection.assert_not_called()


def test_index_pdf_upserts_with_room_metadata():
    fake_q = MagicMock()
    fake_q.collection_exists.return_value = True
    with patch("rag._extract_pages", return_value=["texto pagina uno"]), \
         patch("rag._embed", return_value=[[0.0] * 768]), \
         patch("rag._qdrant", return_value=fake_q):
        from rag import index_pdf
        result = index_pdf("room-1", "manual.pdf", b"%PDF-fake")

    assert result["filename"] == "manual.pdf"
    assert result["chunks"] == 1
    assert "doc_id" in result
    # inspeccionar el payload del punto subido
    points = fake_q.upsert.call_args.kwargs["points"]
    payload = points[0].payload
    assert payload["room_id"] == "room-1"
    assert payload["filename"] == "manual.pdf"
    assert payload["page"] == 1
    assert payload["text"] == "texto pagina uno"
    assert payload["doc_id"] == result["doc_id"]


def test_retrieve_filters_by_room_and_maps_payload():
    hit = MagicMock()
    hit.payload = {"text": "respuesta", "filename": "manual.pdf", "page": 3}
    fake_q = MagicMock()
    fake_q.query_points.return_value.points = [hit]
    with patch("rag._embed", return_value=[[0.0] * 768]), \
         patch("rag._qdrant", return_value=fake_q):
        from rag import retrieve
        chunks = retrieve("room-1", "¿qué dice?", k=5)

    assert chunks == [{"text": "respuesta", "filename": "manual.pdf", "page": 3}]
    # el filtro por room_id se aplicó
    called = fake_q.query_points.call_args
    assert called.kwargs["query_filter"] is not None


def test_list_documents_dedupes_and_paginates():
    p1 = MagicMock()
    p1.payload = {"doc_id": "d1", "filename": "a.pdf"}
    p2 = MagicMock()
    p2.payload = {"doc_id": "d1", "filename": "a.pdf"}
    p3 = MagicMock()
    p3.payload = {"doc_id": "d2", "filename": "b.pdf"}
    fake_q = MagicMock()
    fake_q.scroll.side_effect = [([p1, p2], "next"), ([p3], None)]
    with patch("rag._qdrant", return_value=fake_q):
        from rag import list_documents
        docs = list_documents("room-1")
    assert docs == [
        {"doc_id": "d1", "filename": "a.pdf"},
        {"doc_id": "d2", "filename": "b.pdf"},
    ]


def test_retrieve_degrades_to_empty_on_error():
    with patch("rag._embed", side_effect=RuntimeError("embeddings caídos")):
        from rag import retrieve
        assert retrieve("room-1", "x") == []


def test_list_documents_degrades_to_empty_on_error():
    # Deploy fresco: la colección aún no existe, scroll lanza -> devolver [] (no 500).
    fake_q = MagicMock()
    fake_q.scroll.side_effect = RuntimeError("collection not found")
    with patch("rag._qdrant", return_value=fake_q):
        from rag import list_documents
        assert list_documents("room-1") == []
