import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.audit import create_request_id, save_audit_record, utc_timestamp
from app.ollama import chat
from app.tools import list_files, read_file, safe_command, write_file

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Local Qwen Agent",
    version="1.8.0",
)


MAX_TOOL_ROUNDS = 5
MAX_MEMORY_MESSAGES = 20

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"
MEMORY_FILE = MEMORY_DIR / "conversation.json"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files and directories inside the workspace. "
                "Use path '.' for the workspace root. "
                "Never use an operating-system path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path inside the workspace. "
                            "Use '.' for the workspace root."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file inside the workspace. "
                "Only relative workspace paths are allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path to a UTF-8 text file "
                            "inside the workspace."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a UTF-8 text file inside the workspace. "
                "Only relative workspace paths are allowed. "
                "The content argument must contain the complete final file "
                "content. Use actual information from previous tool results. "
                "Never write placeholders such as '[tool result]' or "
                "'[list of files]'. "
                "Never write outside the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path for the file inside the workspace."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Complete UTF-8 text content to write. "
                            "Use the actual results returned by previous "
                            "tools."
                        ),
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "safe_command",
            "description": (
                "Run one predefined read-only diagnostic operation. "
                "Arbitrary shell commands are NOT supported."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [
                            "python_version",
                            "git_version",
                            "git_status",
                            "git_log",
                        ],
                        "description": "One allowed read-only operation.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


class ChatRequest(BaseModel):
    message: str


SYSTEM_PROMPT = (
    "You are a local AI agent operating inside a safe workspace. "
    "You can only access files inside the workspace. "
    "Use the available tools whenever you need actual workspace "
    "information. "
    "Never invent files, file contents, command results, or other "
    "workspace facts. "
    "If the user gives you information directly, treat it as information "
    "provided by the user. "
    "For requests such as 'remember this', do not search the workspace "
    "unless the user explicitly asks you to do so. "
    "Always treat tool results as authoritative information. "
    "When a tool returns information, use the actual returned information "
    "in your reasoning and in any files you create. "
    "Never replace real tool results with placeholders such as "
    "'[tool result]', '[list of files]', or similar text. "
    "If the user asks you to create a report, the report must contain "
    "the actual information discovered by the tools. "
    "If a requested file is outside the workspace, explain that access "
    "is denied. "
    "You may create or overwrite text files only inside the workspace "
    "using write_file. "
    "For commands, use safe_command only for its predefined read-only "
    "operations. "
    "Never invent or request arbitrary shell commands. "
    "You may use multiple tools when needed to complete a task. "
    "After receiving tool results, check whether the task is actually "
    "complete before giving the final answer. "
    "When the task is complete, provide a concise final answer."
)


def clean_message(message: dict) -> dict:
    """Keep only message fields needed for conversation memory."""

    cleaned = {
        "role": message.get("role", ""),
    }

    if "content" in message:
        cleaned["content"] = message.get("content")

    if "tool_calls" in message:
        cleaned["tool_calls"] = message["tool_calls"]

    if "name" in message:
        cleaned["name"] = message["name"]

    return cleaned


def load_memory() -> list[dict]:
    """Load recent conversation messages from local memory."""

    if not MEMORY_FILE.exists():
        return []

    try:
        data = json.loads(
            MEMORY_FILE.read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Conversation memory is unreadable or corrupted: %s", exc)
        return []

    if not isinstance(data, list):
        logger.warning("Conversation memory has an invalid top-level structure.")
        return []

    cleaned = []

    for message in data:
        if isinstance(message, dict):
            cleaned.append(clean_message(message))
        else:
            logger.warning("Ignoring malformed conversation memory entry: %r", message)

    return cleaned[-MAX_MEMORY_MESSAGES:]


def save_memory(messages: list[dict]) -> None:
    """Save only useful conversation messages."""

    try:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        cleaned_messages = [
            clean_message(message)
            for message in messages
            if isinstance(message, dict)
            and message.get("role") in {
                "user",
                "assistant",
                "tool",
            }
        ]

        recent_messages = cleaned_messages[-MAX_MEMORY_MESSAGES:]

        MEMORY_FILE.write_text(
            json.dumps(
                recent_messages,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Unable to persist conversation memory: %s", exc)


def _tool_result_to_error(result: str) -> str:
    if isinstance(result, str) and any(
        marker in result
        for marker in (
            "Access denied",
            "Path does not exist",
            "File does not exist",
            "Not a file",
            "Not a directory",
            "Cannot write to a directory",
            "Command '",
            "Command timed out",
            "Command executable not found",
            "Error running command",
            "Error reading file",
            "Error writing file",
            "unknown tool",
        )
    ):
        return f"Tool error: {result}"
    return result


async def execute_tool(name: str, arguments: dict) -> str:
    if not isinstance(arguments, dict):
        return "Tool error: invalid tool arguments."

    try:
        if name == "list_files":
            path = arguments.get("path", ".")
            if not isinstance(path, str):
                return "Tool error: invalid arguments for list_files: 'path' must be a string."
            return _tool_result_to_error(list_files(path))

        if name == "read_file":
            if "path" not in arguments or not isinstance(arguments["path"], str):
                return "Tool error: invalid arguments for read_file: missing or invalid 'path'."
            return _tool_result_to_error(read_file(arguments["path"]))

        if name == "write_file":
            if (
                "path" not in arguments
                or "content" not in arguments
                or not isinstance(arguments["path"], str)
                or not isinstance(arguments["content"], str)
            ):
                return (
                    "Tool error: invalid arguments for write_file: "
                    "'path' and 'content' must be strings."
                )
            return _tool_result_to_error(
                write_file(arguments["path"], arguments["content"])
            )

        if name == "safe_command":
            if "command" not in arguments or not isinstance(arguments["command"], str):
                return (
                    "Tool error: invalid arguments for safe_command: "
                    "missing or invalid 'command'."
                )
            return _tool_result_to_error(safe_command(arguments["command"]))

        return f"Tool error: unknown tool: {name}"
    except (TypeError, ValueError) as exc:
        return f"Tool error: {exc}"


def _duration_ms(started_at: float) -> int:
    """Return elapsed request time in milliseconds."""
    return round((time.perf_counter() - started_at) * 1000)


def _write_audit(
    request_id: str,
    timestamp: str,
    started_at: float,
    status: str,
    rounds: int,
    tool_calls_log: list[dict],
    error: str | None = None,
) -> None:
    """Record a sanitized audit record for a request."""
    save_audit_record(
        {
            "request_id": request_id,
            "timestamp": timestamp,
            "model": "qwen3:1.7b",
            "status": status,
            "duration_ms": _duration_ms(started_at),
            "rounds": rounds,
            "tool_calls": [
                {
                    "round": call["round"],
                    "tool": call["tool"],
                }
                for call in tool_calls_log
                if isinstance(call, dict) and "round" in call and "tool" in call
            ],
            "error": error,
        }
    )



@app.get("/")
async def root():
    return {
        "name": "Local Qwen Agent",
        "version": "1.8.0",
        "model": "qwen3:1.7b",
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "max_memory_messages": MAX_MEMORY_MESSAGES,
        "memory_file": "memory/conversation.json",
        "tools": [
            "list_files",
            "read_file",
            "write_file",
            "safe_command",
        ],
    }


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    request_id = create_request_id()
    started_at = time.perf_counter()
    timestamp = utc_timestamp()
    round_number = 1
    tool_calls_log = []

    try:
        memory = load_memory()

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            *memory,
            {
                "role": "user",
                "content": request.message,
            },
        ]

        for round_number in range(1, MAX_TOOL_ROUNDS + 1):
            response = await chat(messages, TOOLS)

            if isinstance(response, dict) and "error" in response:
                _write_audit(
                    request_id=request_id,
                    timestamp=timestamp,
                    started_at=started_at,
                    status="error",
                    rounds=round_number,
                    tool_calls_log=tool_calls_log,
                    error=response["error"],
                )
                payload = {
                    "error": response["error"],
                    "message": response["message"],
                }
                status_code = 503
                if response["error"] == "model_unavailable":
                    status_code = 404
                elif response["error"] == "invalid_ollama_response":
                    status_code = 502
                return JSONResponse(status_code=status_code, content=payload)

            assistant_message = response.get("message")

            if not isinstance(assistant_message, dict):
                _write_audit(
                    request_id=request_id,
                    timestamp=timestamp,
                    started_at=started_at,
                    status="error",
                    rounds=round_number,
                    tool_calls_log=tool_calls_log,
                    error="invalid_ollama_response",
                )
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": "invalid_ollama_response",
                        "message": "Ollama returned an invalid assistant message.",
                    },
                )

            tool_calls = assistant_message.get("tool_calls", [])

            if not tool_calls:
                messages.append(assistant_message)
                save_memory(messages[1:])

                save_audit_record(
                    {
                        "request_id": request_id,
                        "timestamp": timestamp,
                        "model": "qwen3:1.7b",
                        "status": "success",
                        "duration_ms": _duration_ms(started_at),
                        "rounds": round_number,
                        "tool_calls": [
                            {
                                "round": call["round"],
                                "tool": call["tool"],
                            }
                            for call in tool_calls_log
                        ],
                        "error": None,
                    }
                )

                return {
                    "response": assistant_message.get("content", ""),
                    "tool_calls": tool_calls_log,
                    "rounds": round_number,
                    "memory_messages": len(load_memory()),
                }

            messages.append(assistant_message)

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    result = "Tool error: invalid tool call structure."
                    tool_calls_log.append(
                        {
                            "round": round_number,
                            "tool": "unknown",
                            "arguments": {},
                            "result": result,
                        }
                    )
                    messages.append({"role": "tool", "content": result})
                    continue

                function = tool_call.get("function")
                if not isinstance(function, dict):
                    result = "Tool error: invalid tool call structure."
                    tool_calls_log.append(
                        {
                            "round": round_number,
                            "tool": "unknown",
                            "arguments": {},
                            "result": result,
                        }
                    )
                    messages.append({"role": "tool", "content": result})
                    continue

                name = function.get("name")
                arguments = function.get("arguments", {})

                if not isinstance(name, str):
                    result = "Tool error: invalid tool name."
                    tool_calls_log.append(
                        {
                            "round": round_number,
                            "tool": "unknown",
                            "arguments": arguments,
                            "result": result,
                        }
                    )
                    messages.append({"role": "tool", "content": result})
                    continue

                result = await execute_tool(name, arguments)

                tool_calls_log.append(
                    {
                        "round": round_number,
                        "tool": name,
                        "arguments": arguments,
                        "result": result,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                    }
                )

        save_memory(messages[1:])

        _write_audit(
            request_id=request_id,
            timestamp=timestamp,
            started_at=started_at,
            status="error",
            rounds=MAX_TOOL_ROUNDS,
            tool_calls_log=tool_calls_log,
            error="max_tool_rounds_exceeded",
        )

        return {
            "response": (
                "The task reached the maximum tool-call limit "
                f"of {MAX_TOOL_ROUNDS} rounds."
            ),
            "tool_calls": tool_calls_log,
            "rounds": MAX_TOOL_ROUNDS,
            "memory_messages": len(load_memory()),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error during chat processing: %s", exc)
        _write_audit(
            request_id=request_id,
            timestamp=timestamp,
            started_at=started_at,
            status="error",
            rounds=round_number,
            tool_calls_log=tool_calls_log,
            error="unexpected_error",
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "unexpected_error",
                "message": "An unexpected error occurred. Please try again.",
            },
        )
