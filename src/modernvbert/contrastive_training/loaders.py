import os

from PIL import Image

from pathlib import Path

from typing import Any, Union, List

from datasets import load_dataset
from colpali_engine.data import Corpus, ColPaliEngineDataset as T2IColPaliEngineDataset

MMEB_CORPUS_PATH = "/lustre/fsn1/projects/rech/nwd/uyn61im/hf_home/hub/datasets--TIGER-Lab--MMEB-train/snapshots/76dd0a440b6d4c02776830a804443fffbb2d0bfa"
COCO_IMAGE_PATH = "/lustre/fsn1/projects/rech/nwd/uyn61im/hf_home/hub/coco/images/"
COLPALI_SUBSETS = [
        "tatdqa",
        "pdf",
        "arxiv_qa",
        "docvqa",
        "Infographic_VQA"
    ]

IMAGENET_LABELS = Path("/lustre/fswork/projects/rech/nwd/uyn61im/repos/mteb-vlm/mteb/tasks/Image/ZeroShotClassification/eng/templates/Imagenet1k_labels.txt").read_text().splitlines()

dataset_labels = {
    "imagenet": IMAGENET_LABELS
}


class LocalCorpus:
    def __init__(
        self,
        corpus_path: str,
    ):
        """
        Initialize the corpus with the provided data.
        """
        self.corpus_path = corpus_path

        assert os.path.exists(self.corpus_path), f"Corpus path {self.corpus_path} does not exist."

    def __len__(self) -> int:
        """
        Return the number of docs in the corpus.

        Returns:
            int: The number of docs in the corpus.
        """
        return len(os.listdir(self.corpus_path))

    def retrieve(self, docid: Any) -> Image.Image:
        """
        Get the corpus row from the given Doc ID.

        Args:
            docid (str): The id of the document.

        Returns:
            Document: The document retrieved from the corpus.
        """
        path = os.path.join(self.corpus_path, docid)
        with Image.open(path) as img:
            image = img.convert('RGB')
        return image

class I2TColPaliEngineDataset(T2IColPaliEngineDataset):
    def __getitem__(self, idx):
        sample = self.data[idx]

        query = sample[self.query_column_name]

        # If an external document corpus is provided, retrieve the documents from it.
        if self.corpus is not None:
            query = self.corpus.retrieve(query)
            

        pos_targets = sample[self.pos_target_column_name]
        if not isinstance(pos_targets, list):
            pos_targets = [pos_targets]

        if self.neg_target_column_name is not None:
            neg_targets = sample[self.neg_target_column_name]
            if not isinstance(neg_targets, list):
                neg_targets = [neg_targets]
        else:
            neg_targets = None

        return {
            self.QUERY_KEY: query,
            self.POS_TARGET_KEY: pos_targets,
            self.NEG_TARGET_KEY: neg_targets,
        }



def load_mmeb_subset(
        dataset_name_or_path, 
        subsets: List[str]
    ) -> Union[T2IColPaliEngineDataset, I2TColPaliEngineDataset]:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    corpus = LocalCorpus(MMEB_CORPUS_PATH)

    dataset_list = []
    for subset in subsets:
        dataset = load_dataset(dataset_name_or_path, subset, split="diverse_instruction")

        if "qry_image_path" in dataset.column_names:
            train_dataset = I2TColPaliEngineDataset(
                data=dataset,
                corpus=corpus,
                query_column_name="qry_image_path",
                pos_target_column_name="pos_text",
                # neg_target_column_name="neg_text",
            )
        else:
            train_dataset = T2IColPaliEngineDataset(
                data=dataset,
                corpus=corpus,
                query_column_name="qry",
                pos_target_column_name="pos_image_path",
                # neg_target_column_name="neg_image_path",
            )
        dataset_list.append(train_dataset)

    return dataset_list

def load_colpali_train_set(
        dataset_name_or_path, 
        num_negs=0,
        **kwargs
    ) -> T2IColPaliEngineDataset:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    corpus_data = load_dataset("manu/colpali-corpus", split="train")
    corpus = Corpus(corpus_data=corpus_data, doc_column_name="image")

    dataset_list = []
    dataset = load_dataset(dataset_name_or_path, **kwargs)

    # filter to keep only the number of negatives
    if num_negs > 0:
        dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:num_negs]})

    train_dataset = T2IColPaliEngineDataset(
        data=dataset,
        corpus=corpus,
        pos_target_column_name="positive_passages",
        neg_target_column_name="negative_passages" if num_negs > 0 else None,
    )
    dataset_list.append(train_dataset)

    return dataset_list

def load_colpali_train_set_source(
        dataset_name_or_path, 
        subsets=COLPALI_SUBSETS, 
        num_negs=0,
        **kwargs
    ) -> T2IColPaliEngineDataset:
    """Returns the query dataset, then the anchor dataset with the documents, then the dataset type"""
    corpus_data = load_dataset("manu/colpali-corpus", split="train")
    corpus = Corpus(corpus_data=corpus_data, doc_column_name="image")

    dataset_list = []
    for subset in subsets:
        dataset = load_dataset(dataset_name_or_path, split=subset, **kwargs)

        # filter to keep only the number of negatives
        if num_negs > 0:
            dataset = dataset.map(lambda x: {"negative_passages": x["negative_passages"][:num_negs]})

        train_dataset = T2IColPaliEngineDataset(
            data=dataset,
            corpus=corpus,
            pos_target_column_name="positive_passages",
            neg_target_column_name="negative_passages" if num_negs > 0 else None,
        )
        dataset_list.append(train_dataset)

    return dataset_list

def load_imagenet_train_set(
    dataset_name_or_path, 
) -> I2TColPaliEngineDataset:
    dataset = load_dataset(dataset_name_or_path, split="train").shuffle().take(100_000)
    labels = dataset_labels["imagenet"]
    dataset = dataset.map(
        lambda x: {
            "label": labels[int(x["cls"])],
        },
        num_proc=16,
        remove_columns=["cls"]
    )
    train_dataset = I2TColPaliEngineDataset(
        data=dataset,
        query_column_name="jpg",
        pos_target_column_name="label",
    )
    return [train_dataset]

def load_coco_train_set_i2t(
    dataset_name_or_path, 
    **kwargs
) -> I2TColPaliEngineDataset:
    dataset = load_dataset(dataset_name_or_path, **kwargs)
    corpus = LocalCorpus(COCO_IMAGE_PATH)

    dataset = dataset.map(
        lambda x: {
            "path": "/".join(x["url"].split("/")[-2:]),
        },
        remove_columns=["url"]
    )

    train_dataset = I2TColPaliEngineDataset(
        data=dataset,
        corpus=corpus,
        query_column_name="path",
        pos_target_column_name="caption",
    )
    return [train_dataset]

def load_coco_train_set_t2i(
    dataset_name_or_path, 
    **kwargs
) -> T2IColPaliEngineDataset:
    dataset = load_dataset(dataset_name_or_path, **kwargs)
    corpus = LocalCorpus(COCO_IMAGE_PATH)

    dataset = dataset.map(
        lambda x: {
            "path": "/".join(x["url"].split("/")[-2:]),
        },
        remove_columns=["url"]
    )

    train_dataset = T2IColPaliEngineDataset(
        data=dataset,
        corpus=corpus,
        query_column_name="caption",
        pos_target_column_name="path",
    )
    return [train_dataset]

# ---------------------------- NATCAP ----------------------------

NATCAP_SUBSETS = [
    "caltech101",
    "caltech256",
    "cars",
    "country211",
    "dtd",
    "eurosat",
    "fer2013",
    "fgcv_aircraft",
    "food101",
    "pets",
    "resisc45",
    "voc2007",
    "sun397",
]

def load_natcap_train_set(
    dataset_name_or_path,
    subsets: List[str] = NATCAP_SUBSETS,
) -> T2IColPaliEngineDataset:
    datasets = []
    for subset in subsets:
        dataset = load_dataset(dataset_name_or_path, subset, split="train")
        # filter for the column "is_image_class_explicit"
        dataset = dataset.filter(lambda x: x["is_image_class_explicit"], num_proc=16)
        train_dataset = T2IColPaliEngineDataset(
            data=dataset,
            query_column_name="caption",
            pos_target_column_name="image",
        )
        # Combine all datasets into a single dataset
        datasets.append(train_dataset)

    return datasets

def load_rlhn_100K(
        dataset_name_or_path: str = "rlhn/rlhn-100K",
) -> List[T2IColPaliEngineDataset]:
    """
    Load the RLHN 100K dataset and preprocess it to extract positive and negative passages.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name_or_path, split="train")
    dataset = dataset.map(lambda x: {"positive_passages": [p["text"] for p in x["positive_passages"]]})
    dataset = dataset.map(lambda x: {"negative_passages": [p["text"] for p in x["negative_passages"]]})
    # Create the T2IColPaliEngineDataset
    train_dataset = T2IColPaliEngineDataset(
        data=dataset,
        query_column_name="query",
        pos_target_column_name="positive_passages",
        # neg_target_column_name="negative_passages",
    )
    return [train_dataset]