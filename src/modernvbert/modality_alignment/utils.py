import os
import json
from pathlib import Path
import sys


def get_class_file_path(cls):
    module = sys.modules[cls.__module__]
    return Path(os.path.abspath(module.__file__))

def read_text_file(file_path: str) -> str:
    """Read the content of a text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text()

def read_json_file(file_path: str) -> dict:
    """Read the content of a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def write_json_file(file_path: str, data: dict) -> None:
    """Write a dictionary to a JSON file."""
    path = Path(file_path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)