from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

# ===============================
# FILE PATHS
# ===============================
REPO_ROOT = Path(__file__).parent
DATA_ROOT = REPO_ROOT / "data"
PARTICIPANTS_FILE_NAME = "TDBRAIN_participants_V3.xlsx"
PROMPTS_ROOT = REPO_ROOT / "prompts"

# ===============================
# API INFORMATION
# ===============================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"