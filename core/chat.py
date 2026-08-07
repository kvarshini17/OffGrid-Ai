"""
chat.py
=======
Core orchestration for a normal (non-PDF) chat turn: builds the message
list sent to Ollama, streams the response, and keeps the Conversation
object in sync with the database.
"""

from __future__ import annotations

from typing import Generator

from core.history import Conversation, HistoryManager
from core.ollama_client import OllamaClient, OllamaConnectionError


def build_message_payload(conv: Conversation) -> list[dict[str, str]]:
    """Convert a Conversation's stored messages into the Ollama chat format."""
    payload = []
    
    # Inject an invisible system prompt to handle PDF generation requests smoothly.
    system_prompt = (
        "You are OFFGRID AI, a helpful assistant. If the user asks you to generate a PDF, "
        "do not say you cannot do it. The application UI handles the file creation automatically. "
        "CRITICAL: When generating content for a PDF, you must output ONLY the raw requested document content. "
        "Do NOT include any conversational filler, introductions (e.g. 'Here is your PDF'), or disclaimers "
        "(e.g. 'I cannot generate physical files'). Just output the pure content."
    )
    payload.append({"role": "system", "content": system_prompt})
    
    for m in conv.messages:
        if m["role"] in ("user", "assistant", "system"):
            msg = {"role": m["role"], "content": m["content"]}
            if "images" in m:
                msg["images"] = m["images"]
            payload.append(msg)
    return payload


def send_user_message(
    conv: Conversation,
    user_text: str,
    client: OllamaClient,
    history: HistoryManager,
    temperature: float,
    top_p: float,
    max_tokens: int,
    images: list[str] | None = None,
) -> Generator[str, None, None]:
    """
    Append the user's message to the conversation, then stream back the
    assistant's reply. The caller is responsible for rendering the stream
    (e.g. via typing_animation) and must call `finalize_assistant_reply`
    once the full text has been collected.

    Raises:
        OllamaConnectionError: propagated from the client so the UI layer
            can display a friendly error message.
    """
    msg = {"role": "user", "content": user_text}
    if images:
        msg["images"] = images
    conv.messages.append(msg)
    history.save_conversation(conv)

    payload = build_message_payload(conv)
    yield from client.stream_chat(
        model=conv.model,
        messages=payload,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def finalize_assistant_reply(
    conv: Conversation, assistant_text: str, history: HistoryManager
) -> None:
    """Persist the assistant's finished reply to the conversation."""
    if not assistant_text.strip():
        assistant_text = "_(The model returned an empty response.)_"
    conv.messages.append({"role": "assistant", "content": assistant_text})
    history.save_conversation(conv)
