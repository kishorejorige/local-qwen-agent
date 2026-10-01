import json

from app import audit


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