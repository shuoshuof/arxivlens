from __future__ import annotations

import json
import logging
from typing import Any

import requests


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def chat_json(
    model: str,
    system: str,
    user: str,
    base_url: str = "http://localhost:11434",
    timeout: int = 90,
    retries: int = 2,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/chat"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            return _extract_json(content)
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            logging.warning("Ollama chat failed (attempt %s/%s): %s", attempt + 1, retries + 1, exc)
            if attempt < retries:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid or not JSON. "
                            "Return ONLY valid JSON with the required keys, no markdown."
                        ),
                    }
                )
                continue
            break
    raise RuntimeError(f"Ollama chat failed after {retries + 1} attempts: {last_error}")
