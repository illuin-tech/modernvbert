#!/usr/bin/env python
import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import mteb
from mteb.benchmarks import Benchmark
from mteb.overview import get_tasks
from mteb.models import coleurovbert_models, colmodernvbert_models, colvllama_models
from mteb.model_meta import ModelMeta

from config import load_config

MODELS_MODULES = [coleurovbert_models, colmodernvbert_models, colvllama_models]
smolmieb = Benchmark(
    name="MIEB(smol)",
    tasks=get_tasks(
        tasks=[
        # Document retrieval
        "VidoreArxivQARetrieval",
        "VidoreDocVQARetrieval",
        "VidoreInfoVQARetrieval",
        "VidoreTabfquadRetrieval",
        "VidoreTatdqaRetrieval",
        "VidoreShiftProjectRetrieval",
        "VidoreSyntheticDocQAAIRetrieval",
        "VidoreSyntheticDocQAEnergyRetrieval",
        "VidoreSyntheticDocQAGovernmentReportsRetrieval",
        "VidoreSyntheticDocQAHealthcareIndustryRetrieval",
        "Vidore2ESGReportsRetrieval",
        "Vidore2EconomicsReportsRetrieval",
        "Vidore2BioMedicalLecturesRetrieval",
        "Vidore2ESGReportsHLRetrieval",
        # Caption retrieval.
        "MSCOCOT2IRetrieval",
        "MSCOCOI2TRetrieval",
        "Flickr30kT2IRetrieval",
        "Flickr30kI2TRetrieval",
        # Image class.
        "Caltech101",
        "Caltech101ZeroShot",
        "DTD",
        "DTDZeroShot",
        "FER2013",
        "FER2013ZeroShot",
        "EuroSAT",
        "EuroSATZeroShot",
        "OxfordFlowersClassification",
        "OxfordPets",
        "StanfordCars",
        "Food101Classification",
        ],
    ),
    description="""MIEB(lite) is a comprehensive image embeddings benchmark, spanning 10 task types, covering 51 tasks.
    This is a lite version of MIEB(Multilingual), designed to be run at a fraction of the cost while maintaining
    relative rank of models.""",
    reference="https://arxiv.org/abs/2504.10471",
    contacts=["gowitheflow-1998", "isaac-chung"],
    citation=r"""
@article{xiao2025mieb,
  author = {Chenghao Xiao and Isaac Chung and Imene Kerboua and Jamie Stirling and Xin Zhang and Márton Kardos and Roman Solomatin and Noura Al Moubayed and Kenneth Enevoldsen and Niklas Muennighoff},
  doi = {10.48550/ARXIV.2504.10471},
  journal = {arXiv preprint arXiv:2504.10471},
  publisher = {arXiv},
  title = {MIEB: Massive Image Embedding Benchmark},
  url = {https://arxiv.org/abs/2504.10471},
  year = {2025},
}
""",
)

# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MTEB evaluation with a config file.")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to a YAML or JSON config file.",
    )
    return parser.parse_args()

def str_to_torch_dtype(dtype_str: str):
    # e.g. "float16" -> torch.float16
    try:
        return getattr(torch, dtype_str)
    except AttributeError as e:
        raise ValueError(f"Unknown torch dtype: {dtype_str}") from e
    
def str_to_model_class(model_cls_str: str):
    for module in MODELS_MODULES:
        if hasattr(module, model_cls_str):
            return getattr(module, model_cls_str)
    raise ValueError(f"Unknown model class: {model_cls_str}. Available classes: {[m for mod in MODELS_MODULES for m in dir(mod) if not m.startswith('_')]}.")
# ---------------------------
# Main
# ---------------------------
def main(cfg) -> None:
    # --- Logging ---
    logging.getLogger("mteb").setLevel(logging.INFO)

    # --- Model metadata / loading ---
    model_class = str_to_model_class(cfg.eval_config.wrapper_cls)
    if cfg.eval_config.model_name_or_path:
        model_name_or_path = cfg.eval_config.model_name_or_path
    else:
        model_name_or_path = cfg.tr_args.output_dir + "/final"

    if len(model_name_or_path.split("/")) > 2:
        name = f"SmolVEncoder/{model_name_or_path.split('/')[-2]}"
    else:
        name = model_name_or_path.split("/")[-1]

    print(f"Model class: {model_class}")
    print(f"Model name:  {name}")

    custom_model_meta = ModelMeta(
        loader=model_class,
        name=name,
        modalities=["image", "text"],
        framework=["ColPali"],
        similarity_fn_name="max_sim",
        use_instructions=True,
        revision=None,
        release_date=None,
        languages=None,
        n_parameters=None,
        memory_usage_mb=None,
        max_tokens=None,
        embed_dim=128,
        license="apache-2.0",
        open_weights=True,
        public_training_code=None,
        public_training_data=None,
        training_datasets=None,
    )


    custom_model = custom_model_meta.load_model(
        model_name=model_name_or_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

    custom_model.processor.image_processor.size["longest_edge"] = cfg.eval_config.encode_kwargs.pop("max_image_size", 2048)
    custom_model.processor.image_processor.do_resize = cfg.eval_config.encode_kwargs.pop("do_resize", True)

    # --- Load tasks ---
    # tasks = mteb.get_tasks(tasks=cfg.eval_config.tasks)
    tasks = smolmieb
    print(f"Tasks loaded: {tasks}")
    evaluator = mteb.MTEB(tasks=smolmieb)

    # --- Run evaluation ---
    encode_kwargs = {"batch_size": cfg.eval_config.batch_size}
    encode_kwargs.update(cfg.eval_config.encode_kwargs or {})

    evaluator.run(
        model=custom_model,
        verbosity=cfg.eval_config.verbosity,
        encode_kwargs=encode_kwargs,
        output_folder=cfg.eval_config.output_folder,
    )


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    main(cfg)
