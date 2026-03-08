"""
VoiceSlide — Configuration
All server configuration lives here. Never hardcode paths or secrets in app.py.
"""

import os

# Base directory is the voiceslide project root (parent of backend/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Server settings
HOST = os.getenv("VOICESLIDE_HOST", "0.0.0.0")
PORT = int(os.getenv("VOICESLIDE_PORT", "5000"))
DEBUG = os.getenv("VOICESLIDE_DEBUG", "true").lower() == "true"

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
SLIDES_JSON_PATH = os.path.join(DATA_DIR, "slides.json")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
