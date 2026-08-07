"""
ollama_client.py
=================
Thin wrapper around the local Ollama server. Centralizes all HTTP calls to
Ollama so the rest of the app never talks to the API directly. This makes it
easy to swap the backend later and keeps error handling in one place.
"""

from __future__ import annotations

import json
from typing import Generator, Iterable

import requests

from utils.constants import OLLAMA_HOST, FALLBACK_MODELS


class OllamaConnectionError(Exception):
    """Raised when the Ollama server can't be reached."""


class OllamaClient:
    """Small client for interacting with a local Ollama instance."""

    def __init__(self, host: str = OLLAMA_HOST, timeout: int = 120) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """
        Query Ollama for the list of locally installed models.

        Returns:
            A list of model name strings. Falls back to a static list of
            common model names if Ollama is unreachable, so the UI never
            breaks even when the server is offline.
        """
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return models if models else FALLBACK_MODELS
        except (requests.RequestException, ValueError, KeyError):
            return FALLBACK_MODELS

    def is_server_available(self) -> bool:
        """Check whether the Ollama server is reachable at all."""
        try:
            requests.get(f"{self.host}/api/tags", timeout=3)
            return True
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    # Chat generation
    # ------------------------------------------------------------------
    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        Stream a chat completion from Ollama, yielding text chunks as they
        arrive so the UI can render a ChatGPT-style typing effect.

        Args:
            model: Name of the installed Ollama model to use.
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate.

        Yields:
            Successive text fragments of the model's response.

        Raises:
            OllamaConnectionError: If the server can't be reached or the
                model isn't available.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=300.0,
            ) as resp:
                if resp.status_code == 404:
                    raise OllamaConnectionError(
                        f"Model '{model}' was not found. Please pull it first "
                        f"with: ollama pull {model}"
                    )
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                "Could not connect to Ollama. Make sure it's running "
                "(`ollama serve`) and reachable at "
                f"{self.host}."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                "The request to Ollama timed out. The model may be taking "
                "too long to respond."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(f"Ollama request failed: {exc}") from exc

    def generate_title(self, model: str, first_message: str) -> str:
        """Ask the model to generate a short, 3-5 word title for the conversation."""
        prompt = (
            f"Please write a very short title (maximum 5 words) summarizing this message: "
            f"'{first_message}'. Respond ONLY with the title text, nothing else."
        )
        try:
            title = self.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=20,
            )
            # Clean up the response (remove quotes, newlines, etc.)
            title = title.strip(' \n\r\t"\'')
            if len(title) > 40:
                title = title[:37] + "..."
            return title if title else "New Chat"
        except Exception:
            # Fallback if the LLM call fails
            short = first_message.strip()
            return (short[:37] + "...") if len(short) > 40 else short

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> str:
        """Non-streaming convenience wrapper that returns the full response."""
        return "".join(
            self.stream_chat(
                model, messages, temperature=temperature, top_p=top_p,
                max_tokens=max_tokens,
            )
        )
