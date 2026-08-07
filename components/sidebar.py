"""
sidebar.py
==========
Renders the full application sidebar: new-chat button, grouped chat
history with rename/delete, search, uploaded-PDF manager, export panel,
settings, about section, current model indicator, and storage usage.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from core.history import HistoryManager
from core.ollama_client import OllamaClient
from core.pdf_chat import PDFChatEngine, PDFProcessingError
from components.export import EXPORTERS, export_all_history
from utils.constants import UPLOADS_DIR, DATABASE_DIR, EXPORTS_DIR, MAX_PDF_SIZE_MB, \
    EXPORT_FORMATS
from utils.file_utils import (
    get_directory_size, human_readable_size, load_settings, save_settings,
)


def render_sidebar(
    history: HistoryManager,
    ollama_client: OllamaClient,
    pdf_engine: PDFChatEngine,
) -> None:
    """Render the entire sidebar. Mutates st.session_state to drive app.py."""
    with st.sidebar:
        st.markdown('<h2 class="accent-header">✨ OFFGRID AI</h2>', unsafe_allow_html=True)

        _render_new_chat_button(history)
        st.divider()

        _render_pdf_manager(pdf_engine)
        st.divider()

        _render_export_panel(history)
        st.divider()

        _render_search_and_history(history)
        st.divider()

        _render_settings_panel(ollama_client)
        st.divider()

        _render_status_and_about()


# --------------------------------------------------------------------------
# New chat
# --------------------------------------------------------------------------
def _render_new_chat_button(history: HistoryManager) -> None:
    if st.button("📝 New chat", use_container_width=True, type="primary"):
        settings = load_settings()
        conv = history.create_conversation(model=settings["model"])
        st.session_state.current_conv_id = conv.id
        st.session_state.pdf_mode = False
        st.rerun()


# --------------------------------------------------------------------------
# Search + grouped history
# --------------------------------------------------------------------------
def _render_search_and_history(history: HistoryManager) -> None:
    st.markdown("**Chat History**")
    query = st.text_input("Search chats", key="history_search", placeholder="🔍 Search…",
                           label_visibility="collapsed")

    conversations = history.search_conversations(query) if query else history.list_conversations()
    grouped = history.group_by_date(conversations)

    for label in ("Today", "Yesterday", "Last 7 Days", "Older"):
        convs = grouped.get(label, [])
        if not convs:
            continue
        st.caption(label)
        for conv in convs:
            _render_history_item(history, conv)


def _render_history_item(history: HistoryManager, conv) -> None:
    is_active = st.session_state.get("current_conv_id") == conv.id
    cols = st.columns([0.7, 0.15, 0.15])

    with cols[0]:
        label = f"**{conv.title}**" if is_active else conv.title
        if st.button(label, key=f"open_{conv.id}", use_container_width=True):
            st.session_state.current_conv_id = conv.id
            st.session_state.pdf_mode = False
            st.rerun()

    with cols[1]:
        if st.button("✏️", key=f"rename_{conv.id}", help="Rename"):
            st.session_state[f"renaming_{conv.id}"] = True

    with cols[2]:
        if st.button("🗑️", key=f"delete_{conv.id}", help="Delete"):
            history.delete_conversation(conv.id)
            if st.session_state.get("current_conv_id") == conv.id:
                st.session_state.current_conv_id = None
            st.rerun()

    if st.session_state.get(f"renaming_{conv.id}"):
        new_title = st.text_input(
            "New title", value=conv.title, key=f"new_title_{conv.id}"
        )
        confirm_cols = st.columns(2)
        if confirm_cols[0].button("Save", key=f"save_title_{conv.id}"):
            history.rename_conversation(conv.id, new_title)
            st.session_state[f"renaming_{conv.id}"] = False
            st.rerun()
        if confirm_cols[1].button("Cancel", key=f"cancel_title_{conv.id}"):
            st.session_state[f"renaming_{conv.id}"] = False
            st.rerun()


# --------------------------------------------------------------------------
# PDF manager
# --------------------------------------------------------------------------
def _render_pdf_manager(pdf_engine: PDFChatEngine) -> None:
    with st.expander("📄 Uploaded PDFs", expanded=False):
        uploaded_files = st.file_uploader(
            "Drag and drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )

        if uploaded_files:
            for uploaded in uploaded_files:
                dest = UPLOADS_DIR / uploaded.name
                if dest.exists():
                    st.success(f"✅ {uploaded.name} already indexed.")
                    continue  # already uploaded
                try:
                    dest.write_bytes(uploaded.getbuffer())
                    with st.spinner(f"Processing {uploaded.name}…"):
                        n_chunks = pdf_engine.ingest_pdf(dest)
                    st.success(f"✅ {uploaded.name} indexed ({n_chunks} chunks).")
                except PDFProcessingError as exc:
                    st.error(str(exc))
                    dest.unlink(missing_ok=True)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unexpected error processing {uploaded.name}: {exc}")
                    dest.unlink(missing_ok=True)

        existing_pdfs = sorted(UPLOADS_DIR.glob("*.pdf"))
        if not existing_pdfs:
            st.caption("No PDFs uploaded yet.")
        else:
            for pdf_path in existing_pdfs:
                size = human_readable_size(pdf_path.stat().st_size)
                col1, col2 = st.columns([0.8, 0.2])
                col1.markdown(f"📄 {pdf_path.name}  \n<span style='color:#8B949E;font-size:0.8em'>{size}</span>",
                               unsafe_allow_html=True)
                if col2.button("🗑️", key=f"del_pdf_{pdf_path.name}"):
                    try:
                        pdf_engine.remove_document(pdf_path.name)
                    except PDFProcessingError as exc:
                        st.error(str(exc))
                    pdf_path.unlink(missing_ok=True)
                    st.rerun()

            if st.button("💬 Chat with PDFs", use_container_width=True):
                st.session_state.pdf_mode = True
                st.session_state.trigger_pdf_analysis = True
                st.rerun()


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def _render_export_panel(history: HistoryManager) -> None:
    with st.expander("⬇️ Export", expanded=False):
        fmt = st.selectbox("Format", EXPORT_FORMATS, key="export_format")

        current_conv = None
        conv_id = st.session_state.get("current_conv_id")
        if conv_id:
            current_conv = history.get_conversation(conv_id)

        if current_conv and current_conv.messages:
            filename, data = EXPORTERS[fmt](current_conv)
            st.download_button(
                "Export Current Chat", data=data, file_name=filename,
                use_container_width=True,
            )
        else:
            st.caption("No active chat to export yet.")

        all_convs = history.list_conversations()
        if all_convs:
            filename, data = export_all_history(all_convs, fmt)
            st.download_button(
                "Export Entire History", data=data, file_name=filename,
                use_container_width=True,
            )


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
def _render_settings_panel(ollama_client: OllamaClient) -> None:
    with st.expander("⚙️ Settings", expanded=False):
        settings = load_settings()
        available_models = ollama_client.list_models()

        model = st.selectbox(
            "Model", available_models,
            index=available_models.index(settings["model"]) if settings["model"] in available_models else 0,
        )
        temperature = st.slider("Temperature", 0.0, 1.5, float(settings["temperature"]), 0.05)
        max_tokens = st.slider("Max Tokens", 128, 8192, int(settings["max_tokens"]), 128)
        top_p = st.slider("Top P", 0.0, 1.0, float(settings["top_p"]), 0.05)
        theme = st.selectbox("Theme", ["Dark", "Light (coming soon)"],
                              index=0 if settings["theme"] == "Dark" else 0, disabled=True)
        typing_speed = st.slider("Typing Speed (ms/word)", 5, 100, int(settings["typing_speed_ms"]), 5)
        font_size = st.selectbox("Font Size", ["Small", "Medium", "Large"],
                                  index=["Small", "Medium", "Large"].index(settings["font_size"]))

        if st.button("Save Settings", use_container_width=True):
            save_settings({
                "model": model, "temperature": temperature, "max_tokens": max_tokens,
                "top_p": top_p, "theme": "Dark", "typing_speed_ms": typing_speed,
                "font_size": font_size,
            })
            st.success("Settings saved.")
            st.rerun()


# --------------------------------------------------------------------------
# Status + about
# --------------------------------------------------------------------------
def _render_status_and_about() -> None:
    settings = load_settings()
    server_ok = None
    try:
        server_ok = OllamaClient().is_server_available()
    except Exception:  # noqa: BLE001
        server_ok = False

    status_dot = "🟢" if server_ok else "🔴"
    st.caption(f"{status_dot} Model: **{settings['model']}**")

    storage_bytes = get_directory_size(DATABASE_DIR) + get_directory_size(UPLOADS_DIR) + get_directory_size(EXPORTS_DIR)
    st.caption(f"💾 Storage used: {human_readable_size(storage_bytes)}")

    with st.expander("ℹ️ About", expanded=False):
        st.markdown(
            """
            **AI Assistant** is a local, privacy-friendly chatbot powered by
            [Ollama](https://ollama.com) and Streamlit, with document
            question-answering (RAG) over your own PDFs.

            All conversations and documents stay on your machine.
            """
        )
