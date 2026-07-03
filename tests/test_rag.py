import io
import pytest
from pypdf import PdfWriter

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
