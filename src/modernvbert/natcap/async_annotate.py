from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from torch.utils.data import DataLoader  # type: ignore

from datasets import load_dataset
from pqdm.threads import pqdm  # type: ignore
from tqdm import tqdm  

# Vertex AI
from google import genai  # type: ignore

from utils import append_jsonl, load_jsonl_mapping, run_annotation

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_done_ids(path: Path) -> Set[int]:
    """Return IDs already present in an existing JSONL file."""
    if not path.exists():
        return set()
    done: Set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue  # ignore malformed lines
    return done

def _worker(item: Tuple[int, Dict[str, Any]], *, client: genai.Client, task_info: str | None, model: str) -> Dict[str, Any]:
    idx, sample = item
    out = run_annotation(client, sample, task_info, model)
    out.update({"id": idx, "label": sample.get("label")})
    return out

def load_task_info(dataset_id: str) -> str | None:
    mapping_path = Path("task_metadata.jsonl")
    if not mapping_path.exists():
        print("ℹ️ No task metadata found – falling back to default prompt.")
        return None
    from utils import load_jsonl_mapping  # local helper

    task_info = load_jsonl_mapping(mapping_path).get(dataset_id)
    print(f"ℹ️ Using task‑aware prompt: {task_info}")

    return task_info


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel, resumable Gemini inference via Vertex AI (saves per batch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="HuggingFace dataset path or identifier")
    parser.add_argument("--split", default="train", help="Dataset split to process")
    parser.add_argument("--output", default="./results", help="Folder for JSONL outputs")
    parser.add_argument("--model", default="gemini-2.5-flash-lite-preview-06-17", help="Vertex AI model name")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="Max tokens per response")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of parallel worker threads")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    print("⏳ Initialising Vertex AI client …")
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    print(f"Loading dataset '{args.dataset}' (split: {args.split}) …")
    ds = load_dataset(args.dataset, split=args.split)

    task_info = load_task_info(args.dataset)

    output_path = Path(args.output) / Path(args.dataset).name / f"annotations_{args.model}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(output_path)
    if done_ids:
        print(f"↩️  Resuming – {len(done_ids):,} samples ({len(done_ids)/args.batch_size:,}) already completed.")
    start_id = max(done_ids) + 1 if done_ids else 0
    if start_id > 0:
        print(f"↩️  Resuming from ID {start_id} (skipping {len(done_ids)} done samples).")
        ds = ds.skip(start_id)  # skip already processed samples

    dataloader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=lambda x: x,  # no special collating needed
        num_workers=0,  # pqdm handles parallelism
        pin_memory=False,  # not needed for CPU-only processing
    )

    for i, batch in tqdm(enumerate(dataloader), desc="Processing batches", unit="batch"):
        # batch = [
        #     (i*args.batch_size+k, elem) for k, elem in enumerate(batch) 
        #     if i*args.batch_size+k not in done_ids
        # ]
        # if not batch:
        #     continue  # skip if entire chunk already done
        overall_index = i * args.batch_size + start_id
        batch = [
            (overall_index + k, elem) for k, elem in enumerate(batch)
        ]

        results = pqdm(
            batch,
            lambda item: _worker(item, client=client, task_info=task_info, model=args.model),
            n_jobs=min(args.batch_size, len(batch)),
            argument_type="iterable",
        )

        append_jsonl(output_path, results)  # save once per completed batch
        done_ids.update(r["id"] for r in results)

    print(f"✅ Finished!  All predictions written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
