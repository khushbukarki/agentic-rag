import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 3 and body["llm"] is False


def test_ask_returns_grounded_answer_with_sources(client):
    body = client.post("/ask", json={"question": "What is the minimum password length?"}).json()
    assert "14 characters" in body["answer"]
    assert body["sources"][0]["source"] == "it_security_policy.md"
    assert body["mode"] == "extractive"


def test_search_strategy_validation(client):
    assert client.post("/search", json={"query": "leave", "strategy": "magic"}).status_code == 400


def test_upload_and_delete(client):
    files = {"file": ("faq.md", b"# FAQ\nThe office kitchen closes at 4pm.", "text/markdown")}
    assert client.post("/documents", files=files).status_code == 201
    body = client.post("/ask", json={"question": "When does the kitchen close?", "source": "faq.md"}).json()
    assert "4pm" in body["answer"]
    assert client.delete("/documents/faq.md").status_code == 200


def test_upload_rejects_unsupported_type(client):
    files = {"file": ("image.png", b"\x89PNG", "image/png")}
    assert client.post("/documents", files=files).status_code == 400
