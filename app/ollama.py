import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:1.7b"


def _model_unavailable_message() -> str:
    return (
        f"Configured model '{MODEL}' is unavailable. "
        f"Please check `ollama list` or run `ollama pull {MODEL}`."
    )


async def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }

    if tools:
        payload["tools"] = tools

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json=payload,
            )
            response.raise_for_status()
            response_payload = response.json()
    except httpx.ConnectError:
        return {
            "error": "ollama_unavailable",
            "message": "Ollama is unavailable. Please make sure Ollama is running.",
        }
    except httpx.TimeoutException:
        return {
            "error": "ollama_timeout",
            "message": "Ollama request timed out. Please try again.",
        }
    except httpx.HTTPStatusError as exc:
        detail = {}
        try:
            detail = exc.response.json()
        except (TypeError, ValueError):
            detail = {}

        if isinstance(detail, dict):
            error_text = detail.get("error", "")
            if isinstance(error_text, str):
                lower_error = error_text.lower()
                if "not found" in lower_error or (
                    "model" in lower_error and MODEL.lower() in lower_error
                ):
                    return {
                        "error": "model_unavailable",
                        "message": _model_unavailable_message(),
                    }

        return {
            "error": "ollama_unavailable",
            "message": "Ollama is unavailable. Please make sure Ollama is running.",
        }

    if not isinstance(response_payload, dict):
        return {
            "error": "invalid_ollama_response",
            "message": "Ollama returned an invalid response. Please try again.",
        }

    error_text = response_payload.get("error")
    if isinstance(error_text, str):
        lower_error = error_text.lower()
        if "not found" in lower_error or (
            "model" in lower_error and MODEL.lower() in lower_error
        ):
            return {
                "error": "model_unavailable",
                "message": _model_unavailable_message(),
            }
        return {
            "error": "ollama_unavailable",
            "message": "Ollama is unavailable. Please make sure Ollama is running.",
        }

    if "message" not in response_payload or not isinstance(
        response_payload["message"], dict
    ):
        return {
            "error": "invalid_ollama_response",
            "message": (
                "Ollama returned an unexpected response payload. "
                "Please try again."
            ),
        }

    return response_payload