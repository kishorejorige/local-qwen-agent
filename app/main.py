import json

from fastapi import FastAPI
from pydantic import BaseModel

from app.ollama import chat
from app.tools import list_files, read_file


app = FastAPI(
    title="Local Qwen Agent",
    version="1.1.0",
)


class ChatRequest(BaseModel):
    message: str


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories at a path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to inspect.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the text file to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    if name == "list_files":
        return list_files(**arguments)

    if name == "read_file":
        return read_file(**arguments)

    return f"Unknown tool: {name}"


@app.get("/")
def root():
    return {
        "name": "Local Qwen Agent",
        "model": "qwen3:1.7b",
        "version": "1.1.0",
        "status": "running",
        "tools": ["list_files", "read_file"],
    }


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a local AI agent. "
                "You can only access files inside the workspace. "
                "Use the available tools when you need information about files. "
                "Never invent file contents. "
                "If a requested file is outside the workspace, explain that access is denied."
            ),
        },
        {
            "role": "user",
            "content": request.message,
        },
    ]

    # First model call
    result = await chat(messages, tools=TOOLS)

    assistant_message = result["message"]

    # Add the assistant response to the conversation
    messages.append(assistant_message)

    # Check whether Qwen requested a tool
    tool_calls = assistant_message.get("tool_calls", [])

    if not tool_calls:
        return {
            "response": assistant_message.get("content", ""),
            "tool_calls": [],
        }

    executed_tools = []

    # Execute requested tools
    for tool_call in tool_calls:

        function = tool_call["function"]

        name = function["name"]

        arguments = function.get("arguments", {})

        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        tool_result = execute_tool(name, arguments)

        executed_tools.append(
            {
                "tool": name,
                "arguments": arguments,
                "result": tool_result,
            }
        )

        messages.append(
            {
                "role": "tool",
                "tool_name": name,
                "content": tool_result,
            }
        )

    # Give tool results back to Qwen
    final_result = await chat(messages)

    return {
        "response": final_result["message"].get("content", ""),
        "tool_calls": executed_tools,
    }