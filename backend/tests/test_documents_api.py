import io

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_upload_txt_success():
    file_content = b"Hello, this is a test document."
    files = {"file": ("note.txt", io.BytesIO(file_content), "text/plain")}

    response = client.post("/documents/upload", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["file_name"] == "note.txt"
    assert body["document_type"] == "txt"


def test_upload_unsupported_extension_returns_415():
    file = {"file": ("script.exe", io.BytesIO(b"echo Hello"), "application/octet-stream")}
    response = client.post("/documents/upload", files=file)
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_file_too_large_returns_413(monkeypatch):
    # Temporarily shrink the limit to 0.001 MB so we don't need a real giant file
    from backend.app.core import config

    monkeypatch.setattr(config.settings, "max_upload_size_mb", 0.000001)

    file_content = b"This is more than a few bytes of content for sure."
    files = {"file": ("note.txt", io.BytesIO(file_content), "text/plain")}

    response = client.post("/documents/upload", files=files)

    assert response.status_code == 413
