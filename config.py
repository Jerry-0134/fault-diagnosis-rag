"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.txt")

DATA_FILE = BASE_DIR / "data.txt"
EMBEDDING_FILE = BASE_DIR / "case_embeddings.pkl"
TOP_K = 5

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
