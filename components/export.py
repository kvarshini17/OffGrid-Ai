"""
export.py
=========
Handles exporting a single conversation, or the entire chat history, to
TXT, Markdown, PDF, or JSON. Used by the sidebar's Export section.
"""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

from core.history import Conversation
from utils.file_utils import timestamped_filename


def export_to_txt(conv: Conversation) -> tuple[str, bytes]:
    """Export a single conversation as plain text."""
    lines = [f"Conversation: {conv.title}", f"Model: {conv.model}",
              f"Created: {conv.created_at}", "-" * 40, ""]
    for msg in conv.messages:
        speaker = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content']}\n")
    content = "\n".join(lines)
    filename = timestamped_filename(conv.title, "txt")
    return filename, content.encode("utf-8")


def export_to_markdown(conv: Conversation) -> tuple[str, bytes]:
    """Export a single conversation as a Markdown document with headings."""
    lines = [f"# {conv.title}", "", f"**Model:** {conv.model}  ",
              f"**Created:** {conv.created_at}", "", "---", ""]
    for msg in conv.messages:
        heading = "### 🧑 You" if msg["role"] == "user" else "### 🤖 Assistant"
        lines.append(heading)
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
    content = "\n".join(lines)
    filename = timestamped_filename(conv.title, "md")
    return filename, content.encode("utf-8")


def export_to_json(conv: Conversation) -> tuple[str, bytes]:
    """Export a single conversation as JSON, preserving all metadata."""
    payload = {
        "id": conv.id,
        "title": conv.title,
        "model": conv.model,
        "created_at": conv.created_at,
        "updated_at": conv.updated_at,
        "messages": conv.messages,
    }
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    filename = timestamped_filename(conv.title, "json")
    return filename, content.encode("utf-8")


def export_to_pdf(conv: Conversation) -> tuple[str, bytes]:
    """Export a single conversation as a formatted PDF document."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], alignment=TA_LEFT
    )
    user_style = ParagraphStyle(
        "UserStyle", parent=styles["Normal"], spaceAfter=10,
        backColor="#E8F0FE", borderPadding=6,
    )
    assistant_style = ParagraphStyle(
        "AssistantStyle", parent=styles["Normal"], spaceAfter=10,
        backColor="#F1F1F1", borderPadding=6,
    )
    meta_style = ParagraphStyle("MetaStyle", parent=styles["Normal"], textColor="#666666")

    story = [
        Paragraph(_markdown_to_reportlab(conv.title), title_style),
        Paragraph(f"Model: {conv.model} | Created: {conv.created_at}", meta_style),
        Spacer(1, 0.25 * inch),
    ]
    for msg in conv.messages:
        speaker = "You" if msg["role"] == "user" else "Assistant"
        style = user_style if msg["role"] == "user" else assistant_style
        text = _markdown_to_reportlab(msg["content"])
        story.append(Paragraph(f"<b>{speaker}:</b><br/>{text}", style))

    doc.build(story)
    filename = timestamped_filename(conv.title, "pdf")
    return filename, buffer.getvalue()


def export_text_to_pdf(text: str) -> bytes:
    """Generate a simple PDF from a string of text."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    story = []
    for paragraph in text.split('\n\n'):
        if paragraph.strip():
            safe_text = _markdown_to_reportlab(paragraph.strip())
            story.append(Paragraph(safe_text, styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
    doc.build(story)
    return buffer.getvalue()


def export_all_history(conversations: list[Conversation], fmt: str) -> tuple[str, bytes]:
    """Export the entire chat history as a single file in the given format."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if fmt == "JSON":
        payload = [
            {
                "id": c.id, "title": c.title, "model": c.model,
                "created_at": c.created_at, "updated_at": c.updated_at,
                "messages": c.messages,
            }
            for c in conversations
        ]
        content = json.dumps(payload, indent=2, ensure_ascii=False)
        return f"chat-history_{stamp}.json", content.encode("utf-8")

    if fmt == "Markdown":
        lines = [f"# Full Chat History", "", f"_Exported {stamp}_", ""]
        for c in conversations:
            lines.append(f"## {c.title}")
            lines.append(f"**Model:** {c.model}  ")
            lines.append(f"**Created:** {c.created_at}")
            lines.append("")
            for msg in c.messages:
                heading = "**You:**" if msg["role"] == "user" else "**Assistant:**"
                lines.append(f"{heading} {msg['content']}")
                lines.append("")
            lines.append("---")
            lines.append("")
        content = "\n".join(lines)
        return f"chat-history_{stamp}.md", content.encode("utf-8")

    if fmt == "PDF":
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=LETTER)
        styles = getSampleStyleSheet()
        story = [Paragraph("Full Chat History", styles["Title"]), Spacer(1, 0.2 * inch)]
        for c in conversations:
            story.append(Paragraph(_markdown_to_reportlab(c.title), styles["Heading2"]))
            for msg in c.messages:
                speaker = "You" if msg["role"] == "user" else "Assistant"
                text = _markdown_to_reportlab(msg["content"])
                story.append(Paragraph(f"<b>{speaker}:</b><br/>{text}", styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))
        doc.build(story)
        return f"chat-history_{stamp}.pdf", buffer.getvalue()

    # Default: TXT
    lines = [f"Full Chat History (exported {stamp})", "=" * 40, ""]
    for c in conversations:
        lines.append(f"Conversation: {c.title}")
        lines.append(f"Model: {c.model} | Created: {c.created_at}")
        lines.append("-" * 30)
        for msg in c.messages:
            speaker = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {msg['content']}")
        lines.append("")
    content = "\n".join(lines)
    return f"chat-history_{stamp}.txt", content.encode("utf-8")


import re

def _markdown_to_reportlab(text: str) -> str:
    """Parse basic Markdown into ReportLab's mini-HTML."""
    text = (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
    
    # Headers
    text = re.sub(r'^###\s+(.*)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    
    # Bold and Italic
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)
    text = re.sub(r'\*([^\*\n]+)\*', r'<i>\1</i>', text)
    text = re.sub(r'_([^\_\n]+)_', r'<i>\1</i>', text)
    
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
    
    # Lists
    lines = text.split('\n')
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('* ') or stripped.startswith('- '):
            lines[i] = '&bull; ' + stripped[2:]
            
    return "<br/>".join(lines)



EXPORTERS = {
    "TXT": export_to_txt,
    "Markdown": export_to_markdown,
    "PDF": export_to_pdf,
    "JSON": export_to_json,
}
