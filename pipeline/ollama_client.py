"""
Wrapper for the Ollama HTTP API (localhost:11434).
Provides a single generate() function with timeout and retry logic.
"""

import json
import time
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OLLAMA_MODEL, OLLAMA_URL, OLLAMA_TIMEOUT, OLLAMA_RETRIES


class OllamaConnectionError(Exception):
    pass


class OllamaTimeoutError(Exception):
    pass


def generate(prompt: str, system_prompt: str = "") -> str:
    """
    Send a prompt to Ollama and return the response text.

    Args:
        prompt: The user prompt.
        system_prompt: Optional system instruction.

    Returns:
        The model's response as a string.

    Raises:
        OllamaConnectionError: If Ollama is not running or unreachable.
        OllamaTimeoutError: If the request exceeds OLLAMA_TIMEOUT seconds.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt

    last_error = None
    for attempt in range(1, OLLAMA_RETRIES + 2):  # +2: retries + first try
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {OLLAMA_URL}. "
                "Is Ollama running? Try: ollama serve"
            ) from e

        except requests.exceptions.Timeout as e:
            last_error = OllamaTimeoutError(
                f"Ollama request timed out after {OLLAMA_TIMEOUT}s "
                f"(attempt {attempt}/{OLLAMA_RETRIES + 1})"
            )
            if attempt <= OLLAMA_RETRIES:
                time.sleep(1)
            continue

        except requests.exceptions.HTTPError as e:
            raise OllamaConnectionError(
                f"Ollama returned HTTP error: {e}"
            ) from e

        except (json.JSONDecodeError, KeyError) as e:
            raise OllamaConnectionError(
                f"Unexpected response from Ollama: {e}"
            ) from e

    raise last_error


def is_available() -> bool:
    """Check whether Ollama is reachable and the model is loaded."""
    try:
        resp = requests.get(
            OLLAMA_URL.replace("/api/generate", "/api/tags"),
            timeout=5,
        )
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        return False
