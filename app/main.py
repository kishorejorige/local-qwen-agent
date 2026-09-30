from fastapi import FastAPI
from pydantic import BaseModel

from app.ollama import chat
from app.tools import list_files, read_file, write_file, safe_command


app = FastAPI(
    title="Local Qwen Agent",
    version="1.4.1",
)


MAX_TOOL_ROUNDS = 5


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
    "Use the available tools whenever you need information about files. "
    "Never invent file contents. "
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
        "version": "1.4.1",
        "model": "qwen3:1.7b",
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "tools": [
            "list_files",
            "read_file",
            "write_file",
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

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        response = await chat(messages, TOOLS)
        assistant_message = response["message"]

        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            return {
                "response": assistant_message.get("content", ""),
                "tool_calls": tool_calls_log,
                "rounds": round_number,
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

    return {
        "response": (
            "The task reached the maximum tool-call limit "
            f"of {MAX_TOOL_ROUNDS} rounds."
        ),
        "tool_calls": tool_calls_log,
        "rounds": MAX_TOOL_ROUNDS,
    }
