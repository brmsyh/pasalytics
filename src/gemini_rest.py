"""Minimal Gemini REST client built on httpx.

The official google-genai SDK is not used for network calls: on some
corporate networks its connections are dropped (RemoteProtocolError) while a
plain HTTPS request works. Calling the REST API directly keeps the project
portable across office networks, home networks and Streamlit Cloud."""
from __future__ import annotations

import json
import time

import httpx

from . import config

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiError(RuntimeError):
    pass


def post(path: str, payload: dict, api_key: str | None = None, retries: int = 5, timeout: float = 60) -> dict:
    key = api_key or config.GEMINI_API_KEY
    if not key:
        raise GeminiError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    url = f"{BASE_URL}/{path}"
    # Serialise as pure ASCII (non-ASCII characters become \\uXXXX escapes). Some
    # network proxies corrupt raw UTF-8 bytes such as "×" or "–" in request
    # bodies, which the API then rejects as "Invalid JSON payload".
    body = json.dumps(payload, ensure_ascii=True).encode("ascii")
    headers = {"x-goog-api-key": key, "Content-Type": "application/json; charset=utf-8"}
    last_error = ""
    for attempt in range(retries):
        try:
            resp = httpx.post(url, headers=headers, content=body, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code not in (429, 500, 502, 503, 504):  # not retryable (bad key, bad model name...)
                raise GeminiError(last_error)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        wait = 2 ** attempt * 5
        print(f"[gemini] {last_error} - retrying in {wait}s")
        time.sleep(wait)
    raise GeminiError(f"Gemini request failed after {retries} attempts. Last error: {last_error}")
