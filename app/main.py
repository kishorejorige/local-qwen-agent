import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from app.ollama import chat
from app.tools import list_files, read_file, write_file, safe_command


app = FastAPI(
    title="Local Qwen Agent",
    version="1.5.1",
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

        if not isinstance(data, list):
            return []

        cleaned = []

        for message in data:
            if isinstance(message, dict):
                cleaned.append(clean_message(message))

        return cleaned[-MAX_MEMORY_MESSAGES:]

    except (OSError, json.JSONDecodeError):
        return []


def save_memory(messages: list[dict]) -> None:
    """Save only useful conversation messages."""

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


async def execute_tool(name: str, arguments: dict) -> str:
    if name == "list_files":
        return list_files(arguments.get("path", "."))

    if name == "read_file":
        return read_file(arguments["path"])

    if name == "write_file":
        return write_file(
            arguments["path"],
            arguments["content"],
        )

    if name == "safe_command":
        return safe_command(arguments["command"])

    return f"Unknown tool: {name}"


@app.get("/")
async def root():
    return {
        "name": "Local Qwen Agent",
        "version": "1.5.1",
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

    tool_calls_log = []

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        response = await chat(messages, TOOLS)
        assistant_message = response["message"]

        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            messages.append(assistant_message)

            save_memory(messages[1:])

            return {
                "response": assistant_message.get("content", ""),
                "tool_calls": tool_calls_log,
                "rounds": round_number,
                "memory_messages": len(load_memory()),
            }

        messages.append(assistant_message)

        for tool_call in tool_calls:
            function = tool_call["function"]

            name = function["name"]
            arguments = function.get("arguments", {})

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

    return {
        "response": (
            "The task reached the maximum tool-call limit "
            f"of {MAX_TOOL_ROUNDS} rounds."
        ),
        "tool_calls": tool_calls_log,
        "rounds": MAX_TOOL_ROUNDS,
        "memory_messages": len(load_memory()),
    }
