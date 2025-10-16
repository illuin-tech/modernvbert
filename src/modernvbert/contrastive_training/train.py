import os

from pathlib import Path
from colpali_engine.utils.gpu_stats import print_summary
import argparse

from loaders import *
from config import load_config

import logging

SMOLVLM_PROCESSOR_CONFIG_PATH = Path("/lustre/fswork/projects/rech/nwd/uyn61im/visual_encoder/assets/default_config/processor_config.json")

logging.getLogger("datasets").setLevel(logging.ERROR)
    
def save(trainer, is_smolvlm_processor=False):
    """
    Save the trained model and processor to the specified output directory.
    """
    output_dir = Path(trainer.args.output_dir) / "final"
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    trainer.data_collator.processor.save_pretrained(output_dir)
    if is_smolvlm_processor:
        processor_config_dest = output_dir / "processor_config.json"
        processor_config_dest.write_text(SMOLVLM_PROCESSOR_CONFIG_PATH.read_text())
    print(f"Model and processor saved to {output_dir}!")

def parse_args():
    parser = argparse.ArgumentParser(description="Train ColBERT model using a YAML config.")
    parser.add_argument("-c", "--config-file", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = load_config(args.config_file)

    if os.path.exists(f"{cfg.tr_args.output_dir}/final"):
        print(f"Output directory {cfg.tr_args.output_dir}/final already exists. Please remove it or choose a different output directory.")
        return

    trainer = cfg.build_trainer()

    result = trainer.train()
    print_summary(result)

    save(trainer, is_smolvlm_processor=True)


if __name__ == "__main__":
    main()