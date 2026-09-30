import httpx


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:1.7b"


async def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }

    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            OLLAMA_URL,
            json=payload,
        )

        response.raise_for_status()

        return response.json()