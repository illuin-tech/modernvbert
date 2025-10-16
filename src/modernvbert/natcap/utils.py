import json
from typing import List, Dict, Any, Union
from pathlib import Path

from pydantic import BaseModel, Field

# ────────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ────────────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = Path("prompts/default_prompt.txt").read_text()
TASK_AWARE_PROMPT = Path("prompts/task_aware_prompt.txt").read_text()

class CaptioningOutput(BaseModel):
    caption: str = Field(..., description="Descriptive caption produced by Gemini")
    class_tags: List[str] = Field(..., description="Tags helpful to identify the class")
    other_tags: List[str] = Field(..., description="Tags that set it apart from others in the same class")
    is_image_class_explicit: bool = Field(..., description="Whether the image class is understandable by the model")

# ────────────────────────────────────────────────────────────────────────────────

def save_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

def collect_dict_first_value_in_list(dict, keys):
    """Collect the first value from a list of keys in a dictionary.
    Returns the first value found for any of the keys, or None if none are found.
    """
    for key in keys:
        if key in dict:
            return dict[key]
    return None

def load_jsonl_mapping(path: str, key: str = "dataset_name", value: str = "task_scope") -> Dict[str, str]:
    """Load a JSONL file into a dictionary mapping dataset names to task scopes."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "_error" in record:
                continue
            mapping[record[key]] = record[value]
    return mapping


def strip_unwanted_keys(obj):
    """Recursively drop keys Gemini doesn't accept."""
    if isinstance(obj, dict):
        obj.pop("title", None)
        obj.pop("$schema", None)
        obj.pop("definitions", None)
        for v in obj.values():
            strip_unwanted_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            strip_unwanted_keys(item)
    return obj

def append_jsonl(path: Path, records: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    """Append `records` (list of dict) to *path* one‑per‑line."""
    if isinstance(records, dict):
        records = [records]  # Convert single dict to list for consistency
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_annotation(
    client,
    sample: Dict[str, Any],
    task_info: str | None,
    model: str,
) -> Dict[str, Any]:
    prompt = (
        TASK_AWARE_PROMPT.format(task_info=task_info, label=sample["label"]) if task_info else DEFAULT_PROMPT.format(label=sample["label"])
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt, sample["image"]],
            config={
                "response_mime_type": "application/json",
                "response_schema": CaptioningOutput,
            },
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.splitlines() if not l.startswith("```") )

        return json.loads(text)
    except Exception as err:
        return {"_error": f"Error: {err}"}