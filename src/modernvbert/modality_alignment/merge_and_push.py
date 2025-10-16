"""Merge a PEFT/LoRA adapter with its base model and push the merged
checkpoint to the Hugging Face Hub.

Example
-------
python merge_adapter_and_push.py \
    --base_dir ./unwrapped_model \
    --adapter_dir ./unwrapped_adapter \
    --tokenizer_dir ./tokenizer \
    --repo_id your-username/merged-model-name
"""

import argparse
from pathlib import Path
import os

from huggingface_hub import HfApi, upload_folder
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig

from models import get_model_classes, get_model_auto_map
from utils import get_class_file_path, write_json_file, read_json_file

DEFAULT_CONFIG_DIR = "/lustre/fswork/projects/rech/nwd/uyn61im/visual_encoder/assets/default_config"
CHAT_TEMPLATE = read_json_file(f"{DEFAULT_CONFIG_DIR}/chat_template.json")
PROCESSOR_CONFIG = read_json_file(f"{DEFAULT_CONFIG_DIR}/processor_config.json")
PREPROCESSOR_CONFIG = read_json_file(f"{DEFAULT_CONFIG_DIR}/preprocessor_config.json")

def get_model_type(base_dir: str) -> str:
    """Determine the model type based on the base directory name."""
    if "vbert-" in base_dir:
        return "vbert"
    elif "vlm-" in base_dir:
        return "vllama"
    else:
        raise ValueError(
            f"Unknown model type in base_dir: {base_dir}. Expected 'vbert-' or 'vlm-' prefix."
        )
def merge_and_save(base_dir: str,
                   adapter_dir: str,
                   tokenizer_dir: str,
                   output_dir: str) -> None:
    """Load base model + adapter, merge them, and save to *output_dir*."""
    print(f"Loading PEFT config from {adapter_dir} …", flush=True)
    peft_cfg = PeftConfig.from_pretrained(adapter_dir)

    model_type = get_model_type(base_dir)

    print(f"Using model type: {model_type}", flush=True)
    config_class, base_model_class, model_class = get_model_classes(model_type)

    print(f"Loading base model from {base_dir} …", flush=True)
    base_model = model_class.from_pretrained(base_dir)

    print("Loading adapter weights …", flush=True)
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        config=peft_cfg,
        is_trainable=False,
    )

    print("Merging adapter into base model …", flush=True)
    merged_model = peft_model.merge_and_unload()

    print("Model size:", flush=True)
    print(f"Total number of parameters: {merged_model.num_parameters()}", flush=True)

    print("Updating config auto_map …", flush=True)
    merged_model.config.auto_map = get_model_auto_map(model_type)
    if not hasattr(merged_model.config, 'max_position_embeddings'):
        merged_model.config.max_position_embeddings = (
            merged_model.config.text_model.max_position_embeddings 
            if hasattr(merged_model.config, 'max_position_embeddings') 
            else 8192
        )

    print(f"Saving merged model to {output_dir} …", flush=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)\
    
    print("Saving chat template …", flush=True)
    chat_template_dest = Path(output_dir) / "chat_template.json"
    write_json_file(chat_template_dest, CHAT_TEMPLATE)

    print("Saving tokenizer …", flush=True)
    tok_dir = tokenizer_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    tokenizer.chat_template = CHAT_TEMPLATE["chat_template"]
    tokenizer.model_max_length = merged_model.config.max_position_embeddings
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.model_input_names = ["input_ids", "attention_mask", "pixel_values", "pixel_attention_mask"]
    tokenizer.save_pretrained(output_dir)

    print("Saving preprocessor config …", flush=True)
    processor_dest = Path(output_dir) / "processor_config.json"
    write_json_file(processor_dest, PROCESSOR_CONFIG)

    print("Saving preprocessor config …", flush=True)
    preprocessor_dest = Path(output_dir) / "preprocessor_config.json"
    write_json_file(preprocessor_dest, PREPROCESSOR_CONFIG)

    # add a copy of the modeling script
    print("Saving model script …", flush=True)
    model_script_path = get_class_file_path(model_class)
    model_script_dest = Path(output_dir) / model_script_path.name
    model_script_dest.write_text(model_script_path.read_text())

    config_script_path = get_class_file_path(config_class)
    config_script_dest = Path(output_dir) / config_script_path.name
    config_script_dest.write_text(config_script_path.read_text())

def push_to_hub(output_dir: str,
                repo_id: str,
                private: bool = False) -> None:
    """Create/overwrite *repo_id* and push *output_dir* to the Hub."""
    print(f"Creating repository {repo_id} (private={private}) …", flush=True)
    api = HfApi()
    api.create_repo(
        repo_id=repo_id, 
        private=private, 
        repo_type="model",
        exist_ok=True
    )

    print("Uploading files …", flush=True)
    upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        commit_message="Add merged model",
        allow_patterns=["*.json", "*.bin", "*.safetensors", "*.py"],
    )
    print("Upload finished!", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge PEFT adapter with base model and push to the Hub")
    parser.add_argument("checkpoint_dir", help="Path to the base model (e.g. ./unwrapped_model)")
    parser.add_argument("--repo_id",
                        help="Destination repo on the Hub, e.g. username/model-name")
    parser.add_argument("--private", action="store_true",
                        help="Create the repo as private")
    parser.add_argument("--overwrite", action="store_true",)
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure the user has logged in
    if "HF_TOKEN" not in os.environ:
        print("⚠️  Environment variable HF_TOKEN not found. Run `huggingface-cli login` first.", flush=True)

    # Normalize the checkpoint directory
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    base_dir = f"{args.checkpoint_dir}/unwrapped_model"
    adapter_dir = f"{args.checkpoint_dir}/unwrapped_adapter"
    tokenizer_dir = f"{args.checkpoint_dir}/tokenizer"
    output_dir = f"{args.checkpoint_dir}__merged"

    if os.path.exists(output_dir) and not args.overwrite:
        print(f"Output directory {output_dir} already exists and not asked to overwrite.", flush=True)
    else:
        merge_and_save(
            base_dir=base_dir,
            adapter_dir=adapter_dir,
            tokenizer_dir=tokenizer_dir,
            output_dir=output_dir
        )
    
    if args.repo_id:
        push_to_hub(
            output_dir=output_dir,
            repo_id=args.repo_id,
            private=args.private
        )


if __name__ == "__main__":
    main()
