"""
constants.py
============
Centralized configuration and constant values for the application.
Keeping these in one place avoids hardcoded values scattered across modules
and makes the app easy to reconfigure.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Base paths (all relative to project root, no hardcoded absolute paths)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_DIR = BASE_DIR / "database"
UPLOADS_DIR = BASE_DIR / "uploads"
EXPORTS_DIR = BASE_DIR / "exports"
VECTORSTORE_DIR = BASE_DIR / "database" / "vectorstore"
SETTINGS_FILE = BASE_DIR / "database" / "settings.json"

DATABASE_PATH = DATABASE_DIR / "chats.db"

# Ensure required directories exist at import time
for _dir in (DATABASE_DIR, UPLOADS_DIR, EXPORTS_DIR, VECTORSTORE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Ollama configuration
# --------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("DEFAULT_OLLAMA_MODEL", "llama3.2:latest")

# --------------------------------------------------------------------------
# RAG / PDF chat configuration
# --------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model
MAX_PDF_SIZE_MB = 50
TOP_K_RESULTS = 4
NO_ANSWER_MESSAGE = (
    "I couldn't find relevant information inside the uploaded documents."
)

# --------------------------------------------------------------------------
# UI / theme configuration
# --------------------------------------------------------------------------
APP_TITLE = "AI Assistant"
APP_ICON = "🤖"

COLOR_BACKGROUND = "#0E1117"
COLOR_SIDEBAR = "#161B22"
COLOR_CARD = "#1E1E1E"
COLOR_USER_BUBBLE = "#2563EB"
COLOR_ASSISTANT_BUBBLE = "#262B33"
COLOR_ACCENT_GRADIENT = "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)"
COLOR_TEXT = "#E6EDF3"
COLOR_MUTED_TEXT = "#8B949E"
COLOR_BORDER = "#30363D"

DEFAULT_TYPING_SPEED_MS = 25  # milliseconds per word
DEFAULT_FONT_SIZE = "Medium"

# --------------------------------------------------------------------------
# Default generation settings
# --------------------------------------------------------------------------
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 0.9

# --------------------------------------------------------------------------
# History grouping labels
# --------------------------------------------------------------------------
GROUP_TODAY = "Today"
GROUP_YESTERDAY = "Yesterday"
GROUP_LAST_7_DAYS = "Last 7 Days"
GROUP_OLDER = "Older"

# --------------------------------------------------------------------------
# Export formats
# --------------------------------------------------------------------------
EXPORT_FORMATS = ["TXT", "Markdown", "PDF", "JSON"]

# --------------------------------------------------------------------------
# Fallback model list (used if Ollama isn't reachable when populating the
# settings dropdown for the first time)
# --------------------------------------------------------------------------
FALLBACK_MODELS = ["llama3.2", "mistral", "gemma", "phi", "deepseek"]
