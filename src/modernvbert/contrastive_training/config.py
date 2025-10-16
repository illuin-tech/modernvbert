import torch
from transformers import TrainingArguments, EarlyStoppingCallback

from peft import LoraConfig, get_peft_model

from typing import Any, Union, List, Literal, Dict

from dataclasses import dataclass, field

from datasets import load_dataset

from pathlib import Path

import mteb
import yaml
from dacite import from_dict, Config

from colpali_engine import loss as colpali_losses
from colpali_engine import models as colpali_models
from colpali_engine.data import ColPaliEngineDataset
from colpali_engine.collators import VisualRetrieverCollator
from colpali_engine.trainer import ContrastiveTrainer

from loaders import *

@dataclass
class DatasetArgs:
    dataset_name_or_path: str
    loading_kwargs: dict = field(default_factory=dict)

    @property
    def has_negs(self):
        return self.loading_kwargs.get("num_negs", 0) > 0

    def load_data(self):
        if "manu/colpali" in self.dataset_name_or_path.lower():
            dataset = load_colpali_train_set(self.dataset_name_or_path, **self.loading_kwargs)
        elif "imagenet" in self.dataset_name_or_path.lower():
            dataset = load_imagenet_train_set(self.dataset_name_or_path, **self.loading_kwargs)
        elif "mmeb" in self.dataset_name_or_path.lower():
            dataset = load_mmeb_subset(self.dataset_name_or_path, **self.loading_kwargs)
        elif "mscoco" in self.dataset_name_or_path.lower():
            modality = self.loading_kwargs.pop("modality", "t2i")
            if modality == "i2t":
                dataset = load_coco_train_set_i2t(self.dataset_name_or_path, **self.loading_kwargs)
            else:
                dataset = load_coco_train_set_t2i(self.dataset_name_or_path, **self.loading_kwargs)
        elif "natcap" in self.dataset_name_or_path.lower():
            dataset = load_natcap_train_set(self.dataset_name_or_path, **self.loading_kwargs)
        elif "rlhn" in self.dataset_name_or_path.lower():
            dataset = load_rlhn_100K(self.dataset_name_or_path, **self.loading_kwargs)
        else:
            dataset = load_dataset(self.dataset_name_or_path, **self.loading_kwargs)
            dataset = [ColPaliEngineDataset(dataset, query_column_name="query", pos_target_column_name="image")]
        return dataset
    
@dataclass
class ModelArgs:
    model_name_or_path: str
    model_type: str
    loss_type: str
    loading_kwargs: dict = field(default_factory=dict)
    processor_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.model_name_or_path = Path(self.model_name_or_path)

    @property
    def processor_type(self):
        if self.model_type == "ColQwen2_5":
            return "ColQwen2_5_Processor"
        else:
            return self.model_type + "Processor"

    def load_model(self, resume_path=None):
        model_class = getattr(colpali_models, self.model_type)
        torch_dtype = getattr(torch, self.loading_kwargs.pop("torch_dtype", "bfloat16"))
        device = self.loading_kwargs.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
        return model_class.from_pretrained(
            resume_path or self.model_name_or_path,
            torch_dtype=torch_dtype,
            **self.loading_kwargs
        ).to(torch_dtype).to(device)

    def load_processor(self):
        processor_class = getattr(colpali_models, self.processor_type)
        processor = processor_class.from_pretrained(self.model_name_or_path)
        custom_max_image_size = self.processor_kwargs.pop("max_image_size", None)
        if custom_max_image_size is not None:
            processor.image_processor.size["longest_edge"] = custom_max_image_size
        processor.image_processor.do_resize = self.processor_kwargs.pop("do_resize", True)
        return processor
    
@dataclass
class EvalConfig:
    wrapper_cls: str
    tasks: List[str]
    batch_size: int = 32
    output_folder: str = "./results"
    verbosity: int = 2
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_name_or_path: str = None

@dataclass
class ColbertTrainingArguments:
    """
    Custom training arguments for Colbert.
    """
    # Add any custom arguments here if needed
    model_args: ModelArgs = None
    train_dataset_args: Union[DatasetArgs, List[DatasetArgs]] = None
    tr_args: TrainingArguments = None
    lora_config: LoraConfig = None
    eval_dataset_args: Union[DatasetArgs, List[DatasetArgs]] = None
    eval_config: EvalConfig = None

    def build_trainer(self):
        # Load the training data
        train_datasets = self.load_data(split="train")
        if self.eval_dataset_args is not None:
            eval_datasets = self.load_data(split="eval")
            callbacks = [EarlyStoppingCallback(
                early_stopping_patience=3,    # stop after 3 evals with no improvement
                early_stopping_threshold=0.0  # minimal change required
            )]
        else:
            eval_datasets = None
            callbacks = None


        # Load the model
        chkpts = list(Path(self.tr_args.output_dir).glob("checkpoint-*"))
        if len(chkpts) > 0:
            print("Resume from last checkpoint")
            resume_path = max(chkpts, key=os.path.getctime)
            model = self.model_args.load_model(resume_path=resume_path)
        else:
            print("New model, initializing the LoRAs")
            model = self.model_args.load_model()
            # Wrap the model with LoRA
            model = get_peft_model(model, self.lora_config)
            model.print_trainable_parameters()

        # Load the processor
        processor = self.model_args.load_processor()

        # Resolve the loss func
        loss_fn = getattr(colpali_losses, self.model_args.loss_type)()

        # Build the trainer
        trainer = ContrastiveTrainer(
            model=model,
            train_dataset=train_datasets,
            eval_dataset=eval_datasets,
            args=self.tr_args,
            data_collator=VisualRetrieverCollator(processor),
            loss_func=loss_fn,
            is_vision_model=True,
            compute_symetric_loss=False,
            callbacks=callbacks,
        )

        return trainer
    
    def load_data(self, split: Literal["train", "eval"] = "train"):
        """
        Load the dataset based on the provided dataset_name_or_path.
        """
        if split == "train":
            dataset_args = self.train_dataset_args
        elif split == "eval":
            dataset_args = self.eval_dataset_args
        if isinstance(dataset_args, DatasetArgs):
            dataset_args = [dataset_args]
        all_datasets = []
        for dataset_arg in dataset_args:
            datasets = dataset_arg.load_data()
            all_datasets.extend(datasets)
        print(f"✅  Datasets loaded ({split})!")
        print(f"    → # Datasets: {len(all_datasets)}")
        print(f"    → # Samples: {sum(len(ds) for ds in all_datasets)}")

        if split == "eval":
            # concatenate all eval datasets
            if len(all_datasets) > 1:
                print(f"ℹ️   More than one eval dataset loaded, concatenating them.")
                all_datasets = torch.utils.data.ConcatDataset(all_datasets)
            return all_datasets[0]
        if len(all_datasets) == 1:
            print(f"ℹ️   Only one dataset loaded, returning it directly. No sampler needed.")
            return all_datasets[0]

        return all_datasets
    
def load_config(path, return_dict=False) -> Union[ColbertTrainingArguments, Dict[str, Any]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if return_dict:
        return data
    return from_dict(data=data, data_class=ColbertTrainingArguments)