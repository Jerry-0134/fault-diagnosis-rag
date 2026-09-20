import os
from dotenv import load_dotenv

load_dotenv()

DATA_FILE = "data.txt"
TOP_K = 5
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
