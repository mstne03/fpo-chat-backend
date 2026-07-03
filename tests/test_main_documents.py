from unittest.mock import patch
from fastapi.testclient import TestClient


def _client():
    import main
    return main, TestClient(main.app)


def test_upload_rejects_bad_token():
    main, client = _client()
    with patch.object(main, "_verify", return_value=None):
        resp = client.post(
            "/rooms/room-1/documents?token=bad",
            files={"file": ("x.pdf", b"data", "application/pdf")},
        )
    assert resp.status_code == 401


def test_upload_rejects_missing_room():
    main, client = _client()
    with patch.object(main, "_verify", return_value={"uid": "u1", "email": "a@x.com"}), \
         patch.object(main.manager, "get_room", return_value=None):
        resp = client.post(
            "/rooms/nope/documents?token=ok",
            files={"file": ("x.pdf", b"data", "application/pdf")},
        )
    assert resp.status_code == 404


def test_upload_rejects_too_large():
    main, client = _client()
    with patch.object(main, "_verify", return_value={"uid": "u1", "email": "a@x.com"}), \
         patch.object(main.manager, "get_room", return_value=object()), \
         patch.object(main, "MAX_PDF_BYTES", 10), \
         patch("main.rag.index_pdf") as idx:
        resp = client.post(
            "/rooms/room-1/documents?token=ok",
            files={"file": ("big.pdf", b"x" * 50, "application/pdf")},
        )
    assert resp.status_code == 413
    idx.assert_not_called()  # se aborta antes de indexar


def test_upload_indexes_and_returns_result():
    main, client = _client()
    room = object()
    with patch.object(main, "_verify", return_value={"uid": "u1", "email": "a@x.com"}), \
         patch.object(main.manager, "get_room", return_value=room), \
         patch("main.rag.index_pdf", return_value={"doc_id": "d1", "filename": "x.pdf", "chunks": 3}) as idx:
        resp = client.post(
            "/rooms/room-1/documents?token=ok",
            files={"file": ("x.pdf", b"%PDF-data", "application/pdf")},
        )
    assert resp.status_code == 200
    assert resp.json() == {"doc_id": "d1", "filename": "x.pdf", "chunks": 3}
    assert idx.call_args.args[0] == "room-1"


def test_upload_pdf_without_text_returns_422():
    main, client = _client()
    with patch.object(main, "_verify", return_value={"uid": "u1", "email": "a@x.com"}), \
         patch.object(main.manager, "get_room", return_value=object()), \
         patch("main.rag.index_pdf", side_effect=ValueError("PDF sin texto extraíble")):
        resp = client.post(
            "/rooms/room-1/documents?token=ok",
            files={"file": ("x.pdf", b"%PDF", "application/pdf")},
        )
    assert resp.status_code == 422


def test_list_documents_endpoint():
    main, client = _client()
    with patch.object(main, "_verify", return_value={"uid": "u1", "email": "a@x.com"}), \
         patch.object(main.manager, "get_room", return_value=object()), \
         patch("main.rag.list_documents", return_value=[{"doc_id": "d1", "filename": "x.pdf"}]):
        resp = client.get("/rooms/room-1/documents?token=ok")
    assert resp.status_code == 200
    assert resp.json() == {"documents": [{"doc_id": "d1", "filename": "x.pdf"}]}
