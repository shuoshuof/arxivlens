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


def _extract_message(payload: dict[str, Any]) -> str:
    outputs = payload.get("outputs") or []
    for run_output in outputs:
        if not isinstance(run_output, dict):
            continue
        for result in run_output.get("outputs") or []:
            if not isinstance(result, dict):
                continue
            messages = result.get("messages") or []
            for message in messages:
                if isinstance(message, dict):
                    content = message.get("message")
                    if isinstance(content, str) and content.strip():
                        return content
            output_map = result.get("outputs") or {}
            if isinstance(output_map, dict):
                for value in output_map.values():
                    if isinstance(value, dict):
                        content = value.get("message") or value.get("text")
                        if isinstance(content, str) and content.strip():
                            return content
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message
    return ""


def llm_rerank_json(
    flow_id: str,
    overview: str,
    title: str,
    abstract: str,
    base_url: str = "http://localhost:7863",
    api_key: str | None = None,
    timeout: int = 90,
    retries: int = 1,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/v1/run/{flow_id}"
    payload = {
        "input_type": "chat",
        "output_type": "chat",
        "input_value": "",
        "tweaks": {
            "overview": {"input_value": overview},
            "title": {"input_value": title},
            "abstract": {"input_value": abstract},
        },
    }
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    last_error: Exception | None = None
    session = requests.Session()
    session.trust_env = False
    for attempt in range(retries + 1):
        try:
            response = session.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            content = _extract_message(data)
            if not content:
                raise ValueError("Langflow response missing message content")
            return _extract_json(content)
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logging.warning(
                "Langflow LLM rerank failed (attempt %s/%s): %s",
                attempt + 1,
                retries + 1,
                exc,
            )
            if attempt < retries:
                continue
            break
    raise RuntimeError(
        f"Langflow LLM rerank failed after {retries + 1} attempts: {last_error}"
    )
