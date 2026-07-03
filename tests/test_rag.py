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


def test_embed_calls_jina_and_returns_vectors(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "test-key")
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
    }
    fake_resp.raise_for_status.return_value = None
    with patch("rag.requests.post", return_value=fake_resp) as post:
        from rag import _embed
        vectors = _embed(["uno", "dos"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    # envía los textos y la API key en la petición
    body = post.call_args.kwargs["json"]
    assert body["input"] == ["uno", "dos"]
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"


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


def test_retrieve_degrades_to_empty_on_error():
    with patch("rag._embed", side_effect=RuntimeError("embeddings caídos")):
        from rag import retrieve
        assert retrieve("room-1", "x") == []
