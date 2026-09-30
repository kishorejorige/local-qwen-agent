from fastapi import FastAPI
from pydantic import BaseModel

from app.ollama import chat
from app.tools import list_files, read_file, safe_command


app = FastAPI(
    title="Local Qwen Agent",
    version="1.2.0",
)


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
    "You are a local AI agent. "
    "You can only access files inside the workspace. "
    "Use the available tools when you need information about files. "
    "Never invent file contents. "
    "If a requested file is outside the workspace, explain that access "
    "is denied. "
    "For commands, use safe_command only for its predefined read-only "
    "operations. Never invent or request arbitrary shell commands."
)


async def execute_tool(name: str, arguments: dict) -> str:
    if name == "list_files":
        return list_files(arguments.get("path", "."))

    if name == "read_file":
        return read_file(arguments["path"])

    if name == "safe_command":
        return safe_command(arguments["command"])

    return f"Unknown tool: {name}"


@app.get("/")
async def root():
    return {
        "name": "Local Qwen Agent",
        "version": "1.2.0",
        "model": "qwen3:1.7b",
        "tools": [
            "list_files",
            "read_file",
            "safe_command",
        ],
    }


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": request.message,
        },
    ]

    tool_calls_log = []

    response = await chat(messages, TOOLS)
    assistant_message = response["message"]

    tool_calls = assistant_message.get("tool_calls", [])

    if tool_calls:
        messages.append(assistant_message)

        for tool_call in tool_calls:
            function = tool_call["function"]
            name = function["name"]
            arguments = function.get("arguments", {})

            result = await execute_tool(name, arguments)

            tool_calls_log.append(
                {
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

        final_response = await chat(messages, TOOLS)

        return {
            "response": final_response["message"].get("content", ""),
            "tool_calls": tool_calls_log,
        }

    return {
        "response": assistant_message.get("content", ""),
        "tool_calls": [],
    }