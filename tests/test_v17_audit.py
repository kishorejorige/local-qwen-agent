import json

from fastapi.testclient import TestClient

from app import audit, main

client = TestClient(main.app)


def test_create_request_id_is_unique():
    first = audit.create_request_id()
    second = audit.create_request_id()

    assert first.startswith("req-")
    assert second.startswith("req-")
    assert first != second


def test_utc_timestamp_is_valid():
    timestamp = audit.utc_timestamp()

    assert timestamp.endswith("+00:00")


def test_save_audit_record_creates_jsonl_file(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)

    record = {
        "request_id": "req-test123",
        "timestamp": "2026-10-01T10:00:00+00:00",
        "status": "success",
    }

    result = audit.save_audit_record(record)

    assert result is True
    assert audit_file.exists()

    lines = audit_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    saved_record = json.loads(lines[0])

    assert saved_record == record


def test_save_audit_record_appends_records(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)

    first = {
        "request_id": "req-one",
        "status": "success",
    }

    second = {
        "request_id": "req-two",
        "status": "error",
    }

    assert audit.save_audit_record(first) is True
    assert audit.save_audit_record(second) is True

    lines = audit_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == first
    assert json.loads(lines[1]) == second


def test_audit_failure_does_not_raise(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"

    audit_dir.mkdir()

    # Make the target path a directory so opening it for append fails.
    audit_file.mkdir()

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)

    result = audit.save_audit_record(
        {
            "request_id": "req-failure",
            "status": "success",
        }
    )

    assert result is False


def test_chat_success_writes_sanitized_audit_record(tmp_path, monkeypatch):
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
                "content": "Done.",
                "tool_calls": [],
            }
        }

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post(
        "/chat",
        json={"message": "private user message"},
    )

    assert response.status_code == 200

    records = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1

    record = json.loads(records[0])

    assert set(record) == {
        "request_id",
        "timestamp",
        "model",
        "status",
        "duration_ms",
        "rounds",
        "tool_calls",
        "error",
    }
    assert record["request_id"].startswith("req-")
    assert record["timestamp"].endswith("+00:00")
    assert record["model"] == "qwen3:1.7b"
    assert record["status"] == "success"
    assert isinstance(record["duration_ms"], int)
    assert record["duration_ms"] >= 0
    assert record["rounds"] == 1
    assert record["tool_calls"] == []
    assert record["error"] is None
    assert "private user message" not in records[0]
    assert "arguments" not in record
    assert "result" not in record


def test_chat_list_files_writes_sanitized_tool_audit_record(tmp_path, monkeypatch):
    audit_dir = tmp_path / "audit"
    audit_file = audit_dir / "audit.jsonl"
    memory_dir = tmp_path / "memory"
    memory_file = memory_dir / "conversation.json"

    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "AUDIT_FILE", audit_file)
    monkeypatch.setattr(main, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(main, "MEMORY_FILE", memory_file)

    responses = iter(
        [
            {
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
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "The files are listed.",
                    "tool_calls": [],
                }
            },
        ]
    )

    async def fake_chat(messages, tools=None):
        return next(responses)

    monkeypatch.setattr(main, "chat", fake_chat)

    response = client.post(
        "/chat",
        json={"message": "list the workspace files"},
    )

    assert response.status_code == 200

    records = audit_file.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1

    record = json.loads(records[0])

    assert record["status"] == "success"
    assert record["rounds"] == 2
    assert record["tool_calls"] == [{"round": 1, "tool": "list_files"}]
    assert record["error"] is None
    assert "list the workspace files" not in records[0]
    assert "arguments" not in record
    assert "result" not in record