import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "audit"
AUDIT_FILE = AUDIT_DIR / "audit.jsonl"


def create_request_id() -> str:
    """Create a unique identifier for an agent request."""
    return f"req-{uuid.uuid4().hex}"


def utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def save_audit_record(record: dict) -> bool:
    """Append one audit record to the JSONL audit file.

    Audit failures are intentionally non-fatal. The application should
    continue working even if the audit record cannot be written.
    """
    try:
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)

        with AUDIT_FILE.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

        return True

    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Unable to save audit record: %s", exc)
        return False