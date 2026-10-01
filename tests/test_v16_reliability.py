import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.ollama import chat

client = TestClient(main.app)


def test_ollama_unavailable_returns_clean_error(monkeypatch):
    async def fake_post(self, *args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(chat(messages=[{"role": "user", "content": "hi"}]))

    assert result["error"] == "ollama_unavailable"
    assert "Ollama is unavailable" in result["message"]


def test_ollama_timeout_returns_clean_error(monkeypatch):
    async def fake_post(self, *args, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(chat(messages=[{"role": "user", "content": "hi"}]))

    assert result["error"] == "ollama_timeout"
    assert "timed out" in result["message"].lower()


def test_model_unavailable_returns_clean_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"error": "model 'qwen3:1.7b' not found"}

    async def fake_post(self, *args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(chat(messages=[{"role": "user", "content": "hi"}]))

    assert result["error"] == "model_unavailable"
    assert "qwen3:1.7b" in result["message"]
    assert "ollama list" in result["message"]


def test_invalid_ollama_response_is_handled(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"unexpected": "payload"}

    async def fake_post(self, *args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = asyncio.run(chat(messages=[{"role": "user", "content": "hi"}]))

    assert result["error"] == "invalid_ollama_response"
    assert "response" in result["message"].lower()


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        ("ollama_unavailable", 503),
        ("ollama_timeout", 503),
        ("model_unavailable", 404),
        ("invalid_ollama_response", 502),
    ],
)
def test_chat_endpoint_returns_controlled_ollama_errors(
    monkeypatch,
    error,
    status_code,
):
    async def fake_chat(messages, tools=None):
        return {
            "error": error,
            "message": "simulated Ollama failure",
        }

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == status_code
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"] == error


def test_tool_failure_is_reported_safely(monkeypatch):
    async def fake_chat(messages, tools=None):
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "workspace/missing.txt"},
                        }
                    }
                ],
            }
        }

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post("/chat", json={"message": "read missing file"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_calls"][0]["result"].startswith("Tool error:")


def test_invalid_tool_arguments_are_safe(monkeypatch):
    async def fake_chat(messages, tools=None):
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {},
                        }
                    }
                ],
            }
        }

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post("/chat", json={"message": "bad tool call"})

    assert response.status_code == 200
    payload = response.json()
    assert "Tool error" in payload["tool_calls"][0]["result"]


def test_corrupted_memory_recovers(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    memory_file = memory_dir / "conversation.json"
    memory_file.write_text("{not valid json}", encoding="utf-8")

    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    assert main.load_memory() == []


def test_memory_save_failure_does_not_crash(tmp_path, monkeypatch):
    async def fake_chat(messages, tools=None):
        return {
            "message": {
                "role": "assistant",
                "content": "Saved successfully.",
                "tool_calls": [],
            }
        }

    monkeypatch.setattr(main, "chat", fake_chat)
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    memory_file = memory_dir / "conversation.json"
    memory_file.mkdir()
    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    response = client.post("/chat", json={"message": "remember this"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["response"] == "Saved successfully."
