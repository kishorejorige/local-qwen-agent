import json

from fastapi.testclient import TestClient

from app import audit, main

client = TestClient(main.app)


def test_ollama_unavailable_writes_failure_audit_record(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"
    memory_dir = tmp_path / "memory"
    memory_file = memory_dir / "conversation.json"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    async def fake_chat(messages, tools=None):
        return {
            "error": "ollama_unavailable",
            "message": "Ollama is unavailable. Please make sure Ollama is running.",
        }

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 503
    assert response.json()["error"] == "ollama_unavailable"

    records = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1

    record = json.loads(records[0])

    assert record["status"] == "error"
    assert record["error"] == "ollama_unavailable"
    assert record["request_id"].startswith("req-")
    assert isinstance(record["duration_ms"], int)
    assert record["rounds"] == 1


def test_ollama_timeout_writes_failure_audit_record(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"
    memory_dir = tmp_path / "memory"
    memory_file = memory_dir / "conversation.json"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    async def fake_chat(messages, tools=None):
        return {
            "error": "ollama_timeout",
            "message": "Ollama request timed out. Please try again.",
        }

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 503
    assert response.json()["error"] == "ollama_timeout"

    records = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1

    record = json.loads(records[0])

    assert record["status"] == "error"
    assert record["error"] == "ollama_timeout"


def test_model_unavailable_writes_failure_audit_record(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"
    memory_dir = tmp_path / "memory"
    memory_file = memory_dir / "conversation.json"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    async def fake_chat(messages, tools=None):
        return {
            "error": "model_unavailable",
            "message": "Configured model 'qwen3:1.7b' is unavailable.",
        }

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 404
    assert response.json()["error"] == "model_unavailable"

    records = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1

    record = json.loads(records[0])

    assert record["status"] == "error"
    assert record["error"] == "model_unavailable"


def test_invalid_ollama_response_writes_failure_audit_record(
    tmp_path, monkeypatch
):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"
    memory_dir = tmp_path / "memory"
    memory_file = memory_dir / "conversation.json"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    async def fake_chat(messages, tools=None):
        return {
            "error": "invalid_ollama_response",
            "message": "Ollama returned an invalid response. Please try again.",
        }

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
    assert response.json()["error"] == "invalid_ollama_response"

    records = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1

    record = json.loads(records[0])

    assert record["status"] == "error"
    assert record["error"] == "invalid_ollama_response"


def test_max_tool_rounds_exceeded_writes_failure_audit_record(
    tmp_path, monkeypatch
):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"
    memory_dir = tmp_path / "memory"
    memory_file = memory_dir / "conversation.json"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    async def fake_chat(messages, tools=None):
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"path": "."},
                        }
                    }
                ],
            }
        }

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post("/chat", json={"message": "keep listing files"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["rounds"] == main.MAX_TOOL_ROUNDS
    assert (
        payload["response"]
        == f"The task reached the maximum tool-call limit of {main.MAX_TOOL_ROUNDS} rounds."
    )

    records = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1

    record = json.loads(records[0])

    assert record["status"] == "error"
    assert record["error"] == "max_tool_rounds_exceeded"
    assert record["rounds"] == main.MAX_TOOL_ROUNDS


def test_unexpected_exception_writes_failure_audit_record(
    tmp_path, monkeypatch
):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"
    memory_dir = tmp_path / "memory"
    memory_file = memory_dir / "conversation.json"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    def failing_load_memory():
        raise RuntimeError("SECRET_INTERNAL_DETAILS")

    monkeypatch.setattr(main, "load_memory", failing_load_memory)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 500
    assert response.json()["error"] == "unexpected_error"
    assert "SECRET_INTERNAL_DETAILS" not in response.text

    records = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1

    record = json.loads(records[0])

    assert record["status"] == "error"
    assert record["error"] == "unexpected_error"
    assert "SECRET_INTERNAL_DETAILS" not in records[0]


def test_failure_audit_record_privacy(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"
    memory_dir = tmp_path / "memory"
    memory_file = memory_dir / "conversation.json"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    async def fake_chat(messages, tools=None):
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "PRIVATE_TOOL_ARGUMENT_456"},
                        }
                    }
                ],
            }
        }

    async def fake_execute_tool(name, arguments):
        raise RuntimeError("PRIVATE_EXCEPTION_ABC")

    monkeypatch.setattr(main, "chat", fake_chat)
    monkeypatch.setattr(main, "execute_tool", fake_execute_tool)

    response = client.post(
        "/chat",
        json={"message": "PRIVATE_USER_PROMPT_123"},
    )

    assert response.status_code == 500
    assert response.json()["error"] == "unexpected_error"

    raw_audit_text = audit_file.read_text(encoding="utf-8")
    assert "PRIVATE_USER_PROMPT_123" not in raw_audit_text
    assert "PRIVATE_TOOL_ARGUMENT_456" not in raw_audit_text
    assert "PRIVATE_TOOL_RESULT_789" not in raw_audit_text
    assert "PRIVATE_EXCEPTION_ABC" not in raw_audit_text
    assert "traceback" not in raw_audit_text.lower()

    records = raw_audit_text.splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["error"] == "unexpected_error"


def test_audit_write_failure_does_not_break_error_response(
    tmp_path, monkeypatch
):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"
    memory_dir = tmp_path / "memory"
    memory_file = memory_dir / "conversation.json"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    def failing_save_audit(record):
        return False

    monkeypatch.setattr(audit, "save_audit_record", failing_save_audit)

    async def fake_chat(messages, tools=None):
        return {
            "error": "ollama_unavailable",
            "message": "Ollama is unavailable. Please make sure Ollama is running.",
        }

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 503
    assert response.json()["error"] == "ollama_unavailable"
