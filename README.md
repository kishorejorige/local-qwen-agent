# Local Qwen Agent

A local-first AI agent built with Python, FastAPI, and Ollama. The project runs a local Qwen model, keeps a short conversation memory, and restricts file access to a safe workspace.

## Project status

- Project name: Local Qwen Agent
- Current version: 1.7.2
- Model: qwen3:1.7b
- Framework: FastAPI
- Local model runtime: Ollama
- Server URL: http://127.0.0.1:8017
- Maximum tool rounds: 5
- Maximum memory messages: 20
- Memory file: `memory/conversation.json`
- Workspace directory: `workspace/`

## Overview

This project implements a small local agent with:

- a FastAPI server
- local Ollama model access
- a restricted workspace sandbox
- a short-term local conversation memory
- read-only safe diagnostic commands
- resilient handling for Ollama failures, malformed responses, and tool errors

It does not execute arbitrary shell commands and does not allow access outside the workspace.

## Project structure

```text
local-qwen-agent/
|-- app/
|   |-- __init__.py
|   |-- main.py
|   |-- ollama.py
|   |-- tools.py
|   `-- audit.py
|-- memory/
|   `-- conversation.json       # runtime data; ignored by Git
|-- audit/
|   `-- audit.jsonl             # runtime data; ignored by Git
|-- workspace/
|   |-- hello.txt
|   |-- notes.txt
|   `-- test.py
|-- tests/
|   |-- test_v16_reliability.py
|   `-- test_v17_audit.py
|-- .gitignore
|-- README.md
`-- .venv/
```

> The `memory/` and `audit/` directories contain runtime data and are ignored by Git.

## Request tracking and audit trail

### Audit file

V1.7 audit data is written at runtime to:

```text
audit/
`-- audit.jsonl
```

The audit file contains JSONL records created while the app runs. It is runtime-generated data and the `audit/` directory is ignored by Git.

### Audit record fields

- `request_id` - unique request identifier
- `timestamp` - UTC request timestamp
- `model` - configured Ollama model
- `status` - current request status
- `duration_ms` - total request duration
- `rounds` - number of agent rounds
- `tool_calls` - tool names and rounds only
- `error` - error information when applicable

V1.7.2 currently writes a success audit record for successful `/chat` requests. Failure auditing is not yet implemented.

Example JSONL record:

```json
{
  "request_id": "req-example",
  "timestamp": "2026-10-01T13:12:20+00:00",
  "model": "qwen3:1.7b",
  "status": "success",
  "duration_ms": 81565,
  "rounds": 2,
  "tool_calls": [
    {
      "round": 1,
      "tool": "list_files"
    }
  ],
  "error": null
}
```

### Audit privacy and security

The V1.7.2 success audit record intentionally does not store:

- the full user message
- full tool arguments
- full tool results

Only the minimal metadata needed for observability is persisted.

## Requirements

- Python 3.11+
- Windows 10/11
- Ollama installed and running locally
- Git installed
- PowerShell
- Python packages:
  - fastapi
  - uvicorn
  - httpx
  - pydantic

## Windows PowerShell setup

### 1. Create the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install fastapi uvicorn httpx pydantic
```

### 3. Install and test the Ollama model

Check installed models:

```powershell
ollama list
```

If needed, pull the model:

```powershell
ollama pull qwen3:1.7b
```

Test the model:

```powershell
ollama run qwen3:1.7b
```

Exit the Ollama session:

```text
/bye
```

## Run the server

From the project root, start the FastAPI app:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8017
```

The server is available at:

```text
http://127.0.0.1:8017
```

## API endpoints

### GET /

Returns app metadata including the version, model, tool list, and memory configuration.

```powershell
Invoke-RestMethod http://127.0.0.1:8017/
```

### POST /chat

Send a JSON message to the chat endpoint:

```powershell
$body = @{
    message = "List the files in the workspace."
} | ConvertTo-Json

$result = Invoke-RestMethod `
    -Uri http://127.0.0.1:8017/chat `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

$result | ConvertTo-Json -Depth 20
```

## Available tools

The agent can use the following tools:

1. `list_files`
2. `read_file`
3. `write_file`
4. `safe_command`

### Workspace sandbox

File access is restricted to the `workspace/` directory:

```text
workspace/
```

Rules:

- Only paths under the workspace are allowed.
- Only relative paths are accepted.
- Access outside the workspace is denied.
- `write_file` can only create or overwrite files inside the workspace.

This keeps the agent operating inside a controlled sandbox rather than the full filesystem.

### Safe command restrictions

The `safe_command` tool only supports the following predefined read-only operations:

- `python_version`
- `git_version`
- `git_status`
- `git_log`

No arbitrary shell commands are supported. The tool is limited to safe diagnostics only.

## Short-term conversation memory

The app stores a short local conversation history in:

```text
memory/conversation.json
```

Behavior:

- recent messages are stored locally
- the last 20 messages are retained
- memory is short-term runtime state
- the `memory/` directory is ignored by Git
- the file is not treated as a committed project artifact

## Current behavior and safety rules

The current project behavior is intentionally limited:

- no arbitrary shell execution
- no access outside the workspace
- tool outputs are treated as the source of truth
- the agent stops after the configured tool limit of 5 rounds
- the model is expected to use actual workspace results instead of inventing information
- Ollama outages, timeouts, and model-unavailable states return structured JSON errors instead of crashing the server
- malformed Ollama payloads, invalid tool arguments, and tool failures are converted into safe tool or API responses
- corrupted or unreadable memory files are ignored gracefully
- memory write failures do not break a successful chat response

## Reliability improvements in V1.6.0

Version 1.6.0 adds operational safety around the local LLM and the agent loop:

- handles Ollama connection failures and timeout errors cleanly
- detects model-not-found conditions and returns a clear guidance message
- validates malformed Ollama responses before they are processed
- prevents crashes from invalid tool calls or tool argument payloads
- converts tool failures into safe tool results that keep the agent loop running
- recovers gracefully from corrupted conversation memory files
- skips memory-write errors without breaking a successful response
- includes targeted reliability regression tests for these scenarios

## Development history

### V1.7 - Agent Observability & Audit Trail

#### V1.7.1 - Audit Infrastructure

- unique request IDs
- UTC timestamps
- JSONL audit persistence
- non-fatal audit writing
- audit directory excluded from Git

#### V1.7.2 - Request Tracking

- successful `/chat` request tracking
- request duration in milliseconds
- model name
- number of agent rounds
- tool names and tool-call rounds
- request status
- error field
- privacy-conscious audit records
- automated audit tracking tests

### V1.6.0

Added reliability and error-handling improvements for local Ollama access, tool execution, and conversation memory.

### V1.0

Basic local Qwen tool-calling agent.

### V1.1

Added sandboxed workspace tools.

### V1.2

Added safe read-only command execution.

### V1.3

Added safe workspace file writing.

### V1.4

Added multi-step tool loops.

### V1.4.1

Improved tool-result handling and prevented placeholder values in file output.

### V1.5.1

Added local conversation memory.

## Validation

The V1.7.2 implementation has been validated with:

```text
19 tests passed
Ruff: all checks passed
Python compileall: passed
Real /chat smoke tests: passed
```

The test suite includes reliability tests and audit tracking tests.

## Notes

This README is intentionally focused on the actual Local Qwen Agent project in this workspace and does not describe unrelated or generic AI-agent features that are not currently implemented.