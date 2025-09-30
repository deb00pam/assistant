import json
import os
from typing import Dict, Any

USER_DATA_FILE = "user_data.json"

def load_user_data() -> Dict[str, Any]:
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_user_data(data: Dict[str, Any]):
    with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
