import io
import os
import uuid

import requests
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FieldCondition, Filter, MatchValue

# Embeddings vía Cohere (embed-v4.0). embed-v4.0 solo admite output_dimension
# en [256, 512, 1024, 1536]; usamos 1024. La colección Qdrant debe crearse con
# esta misma dimensión (ver ensure_collection).
COHERE_URL = "https://api.cohere.com/v2/embed"
EMBED_MODEL = "embed-v4.0"
COLLECTION = "fpo_documents"
VECTOR_SIZE = 1024

_client_cache = {}


def _extract_pages(pdf: bytes | str) -> list[str]:
    # Acepta bytes o una ruta de fichero. Con ruta, PdfReader lee de disco
    # (streaming) en vez de mantener el PDF entero en RAM -> menos memoria.
    source = io.BytesIO(pdf) if isinstance(pdf, bytes) else pdf
    reader = PdfReader(source)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    if not any(pages):
        raise ValueError("PDF sin texto extraíble")
    return pages


# Cohere limita el nº de textos por request (96); trocear en lotes evita
# rechazos y cuelgues con PDFs grandes (miles de chunks).
EMBED_BATCH = 96


def _embed(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    # input_type: "search_document" al indexar, "search_query" al recuperar
    # (embeddings asimétricos de Cohere -> mejor recuperación).
    if not texts:
        return []
    headers = {"Authorization": f"Bearer {os.environ['COHERE_API_KEY']}"}
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        resp = requests.post(
            COHERE_URL,
            headers=headers,
            json={
                "model": EMBED_MODEL,
                "input_type": input_type,
                "embedding_types": ["float"],
                "output_dimension": VECTOR_SIZE,
                "texts": batch,
            },
            timeout=60,
        )
        if not resp.ok:
            # Cohere explica el motivo (key inválida, cuota...) en el cuerpo;
            # sin esto el error queda opaco.
            raise RuntimeError(f"Cohere {resp.status_code}: {resp.text[:300]}")
        vectors.extend(resp.json()["embeddings"]["float"])
    return vectors


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


def _create_collection(client) -> None:
    params = VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    client.recreate_collection(collection_name=COLLECTION, vectors_config=params)
    # Índice de payload sobre room_id: sin él, Qdrant no filtra de forma fiable
    # por sala y retrieve/list_documents devuelven 0 aunque los puntos existan.
    client.create_payload_index(
        collection_name=COLLECTION, field_name="room_id", field_schema="keyword"
    )


def ensure_collection() -> None:
    client = _qdrant()
    if not client.collection_exists(COLLECTION):
        _create_collection(client)
        return
    # Recrear si la dimensión cambió (cambio de proveedor de embeddings) o si
    # falta el índice de room_id (colecciones antiguas creadas sin él).
    info = client.get_collection(COLLECTION)
    dim = info.config.params.vectors.size
    has_index = "room_id" in (info.payload_schema or {})
    if dim != VECTOR_SIZE or not has_index:
        _create_collection(client)


def index_pdf(room_id: str, filename: str, pdf: bytes | str) -> dict:
    # pdf: bytes o ruta de fichero (el endpoint pasa una ruta temporal en disco).
    ensure_collection()
    pages = _extract_pages(pdf)
    doc_id = uuid.uuid4().hex

    texts, metas = [], []
    for page_num, page_text in enumerate(pages, start=1):
        for chunk in _chunk(page_text):
            texts.append(chunk)
            metas.append({"page": page_num, "text": chunk})

    vectors = _embed(texts)
    points = [
        PointStruct(
            id=uuid.uuid4().hex,
            vector=vec,
            payload={
                "room_id": room_id,
                "doc_id": doc_id,
                "filename": filename,
                "page": meta["page"],
                "text": meta["text"],
            },
        )
        for vec, meta in zip(vectors, metas)
    ]
    _qdrant().upsert(collection_name=COLLECTION, points=points)
    return {"doc_id": doc_id, "filename": filename, "chunks": len(points)}


def list_documents(room_id: str) -> list[dict]:
    # Degrada a [] igual que retrieve: en un deploy fresco la colección aún no
    # existe y scroll lanza; el listado nunca debe devolver 500.
    try:
        room_filter = Filter(
            must=[FieldCondition(key="room_id", match=MatchValue(value=room_id))]
        )
        seen = {}
        offset = None
        while True:
            points, offset = _qdrant().scroll(
                collection_name=COLLECTION,
                scroll_filter=room_filter,
                with_payload=True,
                limit=100,
                offset=offset,
            )
            for p in points:
                doc_id = p.payload["doc_id"]
                if doc_id not in seen:
                    seen[doc_id] = {"doc_id": doc_id, "filename": p.payload["filename"]}
            if offset is None:
                break
        return list(seen.values())
    except Exception:
        return []


def retrieve(room_id: str, query: str, k: int = 5) -> list[dict]:
    try:
        vector = _embed([query], input_type="search_query")[0]
        room_filter = Filter(
            must=[FieldCondition(key="room_id", match=MatchValue(value=room_id))]
        )
        res = _qdrant().query_points(
            collection_name=COLLECTION,
            query=vector,
            query_filter=room_filter,
            limit=k,
        )
        return [
            {
                "text": p.payload["text"],
                "filename": p.payload["filename"],
                "page": p.payload["page"],
            }
            for p in res.points
        ]
    except Exception:
        return []
