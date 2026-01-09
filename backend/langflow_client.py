from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable

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


def _deep_find_text(value: Any, seen: set[int]) -> str:
    if value is None:
        return ""
    value_id = id(value)
    if value_id in seen:
        return ""
    seen.add(value_id)

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        for item in value:
            content = _deep_find_text(item, seen)
            if content:
                return content
        return ""
    if isinstance(value, dict):
        for key in ("message", "text", "content", "output"):
            if key in value and isinstance(value[key], str) and value[key].strip():
                return value[key].strip()
        for key in ("message", "text", "content", "output"):
            if key in value:
                content = _deep_find_text(value[key], seen)
                if content:
                    return content
        for item in value.values():
            content = _deep_find_text(item, seen)
            if content:
                return content
        return ""

    for attr in ("message", "text", "content", "output", "result", "outputs", "data"):
        if hasattr(value, attr):
            content = _deep_find_text(getattr(value, attr), seen)
            if content:
                return content
    if hasattr(value, "model_dump"):
        try:
            content = _deep_find_text(value.model_dump(), seen)
            if content:
                return content
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            content = _deep_find_text(value.dict(), seen)
            if content:
                return content
        except Exception:
            pass
    try:
        content = _deep_find_text(vars(value), seen)
    except TypeError:
        content = ""
    return content


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
    return _deep_find_text(payload, set())


def _resolve_langflow_runner() -> Callable[..., Any] | None:
    try:
        from langflow.load import run_flow_from_json

        return run_flow_from_json
    except ImportError:
        pass
    try:
        from langflow.load import load_flow_from_json
    except ImportError:
        load_flow_from_json = None
    if load_flow_from_json:
        def _runner(
            flow_path: str,
            input_value: str = "",
            input_type: str = "chat",
            output_type: str = "chat",
            tweaks: dict[str, Any] | None = None,
        ) -> Any:
            flow = load_flow_from_json(flow_path)
            return flow(
                input_value=input_value,
                input_type=input_type,
                output_type=output_type,
                tweaks=tweaks,
            )

        return _runner
    return None


@contextmanager
def _with_local_no_proxy() -> Any:
    keys = ("NO_PROXY", "no_proxy")
    originals = {key: os.environ.get(key) for key in keys}
    addrs = ["localhost", "127.0.0.1", "::1"]

    def _merge(value: str | None) -> str:
        if not value:
            return ",".join(addrs)
        parts = [item.strip() for item in value.split(",") if item.strip()]
        for addr in addrs:
            if addr not in parts:
                parts.append(addr)
        return ",".join(parts)

    try:
        for key in keys:
            os.environ[key] = _merge(os.environ.get(key))
        yield
    finally:
        for key, value in originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_flow_local(
    flow_path: str,
    tweaks: dict[str, Any],
    input_value: str = "",
    input_type: str = "chat",
    output_type: str = "chat",
) -> Any:
    runner = _resolve_langflow_runner()
    if runner is None:
        raise RuntimeError(
            "Langflow library not available. Install it with `pip install langflow` "
            "or use --langflow_mode http."
        )
    call_variants = [
        lambda: runner(
            flow_path,
            input_value=input_value,
            input_type=input_type,
            output_type=output_type,
            tweaks=tweaks,
        ),
        lambda: runner(flow_path, input_value, tweaks),
        lambda: runner(
            flow=flow_path,
            input_value=input_value,
            input_type=input_type,
            output_type=output_type,
            tweaks=tweaks,
        ),
        lambda: runner(flow=flow_path, input_value=input_value, tweaks=tweaks),
    ]
    last_error: Exception | None = None
    with _with_local_no_proxy():
        for call in call_variants:
            try:
                return call()
            except TypeError as exc:
                last_error = exc
                continue
    raise RuntimeError(
        "Unsupported langflow run_flow signature. "
        f"Last error: {last_error}"
    )


def llm_rerank_json(
    flow_id: str,
    overview: str,
    title: str,
    abstract: str,
    base_url: str = "http://localhost:7863",
    api_key: str | None = None,
    timeout: int = 90,
    retries: int = 1,
    mode: str = "local",
    flow_path: str | None = None,
) -> dict[str, Any]:
    mode = (mode or "http").strip().lower()
    tweaks = {
        "overview": {"input_value": overview},
        "title": {"input_value": title},
        "abstract": {"input_value": abstract},
    }
    last_error: Exception | None = None
    if mode == "local":
        if not flow_path:
            raise ValueError("flow_path is required when langflow_mode=local")
        for attempt in range(retries + 1):
            try:
                payload = _run_flow_local(flow_path, tweaks=tweaks)
                if isinstance(payload, dict) and {"relevant", "fit_score"} <= set(payload):
                    return payload
                content = ""
                if isinstance(payload, str):
                    content = payload
                elif isinstance(payload, dict):
                    content = _extract_message(payload)
                elif isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict):
                            content = _extract_message(item)
                            if content:
                                break
                    if not content:
                        content = _deep_find_text(payload, set())
                if not content:
                    raise ValueError("Langflow local run missing message content")
                return _extract_json(content)
            except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
                last_error = exc
                logging.warning(
                    "Langflow local LLM rerank failed (attempt %s/%s): %s",
                    attempt + 1,
                    retries + 1,
                    exc,
                )
                if attempt < retries:
                    continue
                break
        raise RuntimeError(
            f"Langflow local LLM rerank failed after {retries + 1} attempts: {last_error}"
        )

    url = f"{base_url.rstrip('/')}/api/v1/run/{flow_id}"
    payload = {
        "input_type": "chat",
        "output_type": "chat",
        "input_value": "",
        "tweaks": tweaks,
    }
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

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
