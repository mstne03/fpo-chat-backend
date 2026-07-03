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


def test_embed_calls_gemini_and_returns_vectors():
    fake_resp = MagicMock()
    fake_resp.embeddings = [MagicMock(values=[0.1, 0.2]), MagicMock(values=[0.3, 0.4])]
    with patch("rag._gemini_client") as get_client:
        get_client.return_value.models.embed_content.return_value = fake_resp
        from rag import _embed
        vectors = _embed(["uno", "dos"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


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
