import io
import os

from pypdf import PdfReader
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

EMBED_MODEL = "text-embedding-004"
COLLECTION = "fpo_documents"
VECTOR_SIZE = 768

_client_cache = {}


def _extract_pages(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    if not any(pages):
        raise ValueError("PDF sin texto extraíble")
    return pages


def _gemini_client():
    if "c" not in _client_cache:
        _client_cache["c"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client_cache["c"]


def _embed(texts: list[str]) -> list[list[float]]:
    resp = _gemini_client().models.embed_content(model=EMBED_MODEL, contents=texts)
    return [list(e.values) for e in resp.embeddings]


def _chunk(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + size]
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def _qdrant() -> QdrantClient:
    if "q" not in _client_cache:
        _client_cache["q"] = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
    return _client_cache["q"]


def ensure_collection() -> None:
    client = _qdrant()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
