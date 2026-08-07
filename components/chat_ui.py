"""
chat_ui.py
==========
Renders the main chat window: custom dark theme CSS, scrollable message
list with styled bubbles, and the sticky bottom input. Also renders the
PDF-chat source citations.
"""

from __future__ import annotations

import streamlit as st

from core.history import Conversation
from core.pdf_chat import RetrievedChunk
from utils.constants import (
    COLOR_BACKGROUND, COLOR_SIDEBAR, COLOR_CARD, COLOR_USER_BUBBLE,
    COLOR_ASSISTANT_BUBBLE, COLOR_ACCENT_GRADIENT, COLOR_TEXT,
    COLOR_MUTED_TEXT, COLOR_BORDER,
)

FONT_SIZE_MAP = {"Small": "14px", "Medium": "16px", "Large": "18px"}


def inject_custom_css(font_size: str = "Medium") -> None:
    """Inject the app's dark-mode theme and chat-bubble styling."""
    base_font = FONT_SIZE_MAP.get(font_size, "16px")
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            font-size: {base_font};
        }}

        .stApp {{
            background-color: {COLOR_BACKGROUND};
            color: {COLOR_TEXT};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {COLOR_SIDEBAR};
            border-right: 1px solid {COLOR_BORDER};
        }}

        /* Chat bubbles */
        .chat-bubble {{
            padding: 12px 16px;
            border-radius: 16px;
            margin-bottom: 10px;
            max-width: 80%;
            line-height: 1.5;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
            animation: fadeIn 0.25s ease-in-out;
            word-wrap: break-word;
        }}

        .user-bubble {{
            background-color: {COLOR_USER_BUBBLE};
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }}

        .assistant-bubble {{
            background-color: {COLOR_ASSISTANT_BUBBLE};
            color: {COLOR_TEXT};
            margin-right: auto;
            border-bottom-left-radius: 4px;
            border: 1px solid {COLOR_BORDER};
        }}

        .bubble-row {{
            display: flex;
            width: 100%;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .thinking-indicator {{
            color: {COLOR_MUTED_TEXT};
            font-style: italic;
            padding: 8px 4px;
        }}

        /* Cards used in settings / about / PDF list */
        .app-card {{
            background-color: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            transition: box-shadow 0.2s ease-in-out;
        }}
        .app-card:hover {{
            box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        }}

        /* Gradient accent header */
        .accent-header {{
            background: {COLOR_ACCENT_GRADIENT};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }}

        /* Streamlit Button Overrides */
        .stButton > button {{
            border-radius: 8px;
            border: 1px solid {COLOR_BORDER};
            transition: all 0.15s ease-in-out;
        }}
        .stButton > button:hover {{
            border-color: #8B5CF6;
        }}
        
        /* ChatGPT-style Sidebar Buttons */
        section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {{
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            text-align: left !important;
            padding: 4px 8px !important;
            justify-content: flex-start !important;
            border-radius: 8px !important;
            color: #d1d5db !important;
            font-weight: 400 !important;
        }}
        section[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {{
            background-color: #2a2b32 !important;
            color: #ffffff !important;
        }}
        
        /* Sidebar Primary Button (New Chat) */
        section[data-testid="stSidebar"] [data-testid="baseButton-primary"] {{
            background-color: transparent !important;
            border: none !important;
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 8px !important;
            border-radius: 8px !important;
            color: #ffffff !important;
            font-weight: 500 !important;
        }}
        section[data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {{
            background-color: #2a2b32 !important;
        }}

        /* Source citation chips */
        .source-chip {{
            display: inline-block;
            background-color: {COLOR_CARD};
            border: 1px solid {COLOR_BORDER};
            border-radius: 8px;
            padding: 4px 10px;
            margin: 2px 4px 2px 0;
            font-size: 0.85em;
            color: {COLOR_MUTED_TEXT};
        }}

        /* Sticky-feeling chat input */
        div[data-testid="stChatInput"] {{
            background-color: #212121 !important;
            border: 1px solid #424242 !important;
            border-radius: 24px !important;
            padding-left: 12px !important;
        }}

        div[data-testid="stChatInput"] textarea {{
            background-color: transparent !important;
            color: #FFFFFF !important;
            caret-color: #FFFFFF !important;
        }}

        div[data-testid="stChatInput"] textarea::placeholder {{
            color: #888888 !important;
        }}

        /* Scrollable chat container */
        .chat-scroll-area {{
            max-height: 65vh;
            overflow-y: auto;
            padding-right: 6px;
        }}
        </style>
        """,unsafe_allow_html=True)
def render_message(role: str, content: str, msg_idx: int = 0) -> None:
    """Render a single chat bubble for a stored message."""
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    justify = "flex-end" if role == "user" else "flex-start"
    st.markdown(
        f'<div class="bubble-row" style="justify-content:{justify}">'
        f'<div class="chat-bubble {bubble_class}">{content}</div></div>',
        unsafe_allow_html=True,
    )
    if role == "assistant":
        from components.export import export_text_to_pdf
        pdf_data = export_text_to_pdf(content)
        cols = st.columns([0.02, 0.3, 0.68])
        with cols[1]:
            st.download_button(
                label="📄 Download Response as PDF",
                data=pdf_data,
                file_name=f"response_{msg_idx}.pdf",
                mime="application/pdf",
                key=f"dl_pdf_msg_{msg_idx}",
                help="Download this response as a PDF document",
                use_container_width=True
            )


def render_conversation(conv: Conversation) -> None:
    """Render every message in a conversation."""
    for i, msg in enumerate(conv.messages):
        render_message(msg["role"], msg["content"], msg_idx=i)


def render_sources(chunks: list[RetrievedChunk]) -> None:
    """Render the source citation chips beneath a PDF-chat answer."""
    if not chunks:
        return
    chips = "".join(
        f'<span class="source-chip">📄 {c.document_name} · p.{c.page_number}</span>'
        for c in chunks
    )
    with st.expander("Sources", expanded=False):
        st.markdown(chips, unsafe_allow_html=True)
        for i, c in enumerate(chunks, start=1):
            st.markdown(f"**{i}. {c.document_name} (page {c.page_number})**")
            st.caption(c.content[:400] + ("…" if len(c.content) > 400 else ""))


def render_empty_state() -> None:
    """Friendly placeholder shown when a conversation has no messages yet."""
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px; opacity: 0.7;">
            <h2 class="accent-header">👋 How can I help you today?</h2>
            <p>Ask me anything, or upload a PDF to chat with your documents.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
