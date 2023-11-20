import cv2
import numpy as np
import os
import re
import torch

from abc import ABC, abstractmethod

from datasets import Dataset, DatasetDict
from itertools import chain
from os.path import join
from PIL import Image
from tifffile import tifffile
from transformers import SamProcessor
from torch.utils.data import Dataset as TorchDataset

from src.utils import RegistryMixin


class PreprocessingStrategy(ABC, RegistryMixin):

    file_reader = "default"

    @staticmethod
    def train_valid_test_split(
        dataset: Dataset, valid_size: float, test_size: float) -> dict:

        dataset = dataset.train_test_split(test_size=test_size)
        train_valid = dataset["train"].train_test_split(test_size=valid_size)
        dataset["train"] = train_valid["train"]
        dataset["valid"] = train_valid["test"]
        return dataset

    @abstractmethod
    def preprocess(
        self, processor: SamProcessor, valid_size: float = 0.1, test_size: float = 0.1, **kwargs):
        pass


@PreprocessingStrategy.register_subclass("busi")
class BUSIPreprocessingStrategy(PreprocessingStrategy):

    def preprocess(
        self, 
        processor: SamProcessor, 
        valid_size: float = 0.1, 
        test_size: float = 0.1,
        use_bounding_box: bool = True
    ) -> DatasetDict:

        dataset_path = "data/Dataset_BUSI_with_GT"
        paths = {}  # dictionary matching ids to image path and label path key-value pairs
        chains = [
            os.listdir(join(dataset_path, "benign")), 
            os.listdir(join(dataset_path, "malignant"))
        ]
        for file in chain(*chains):
            # Each file string contains a number, use re to extract this
            idx = re.search(r"\d+", file).group(0)
            if idx not in paths:
                paths[idx] = {}
            if "mask" in file and ".png" in file:
                paths[idx]["label"] = file
            elif ".png" in file:
                paths[idx]["image"] = file
        image = []
        label = []
        for idx, path in paths.items():
            if "benign" in path["image"]:
                p = os.path.join(dataset_path, "benign")
            else:
                p = os.path.join(dataset_path, "malignant")
            image.append(os.path.join(p, path["image"]))
            label.append(os.path.join(p, path["label"]))

        assert len(image) == len(label), "Number of images and labels do not match."
        
        dataset = Dataset.from_dict({"image": image, "label": label})

        dataset = self.train_valid_test_split(dataset, valid_size, test_size)
        for key in dataset.keys():
            dataset[key] = PathDataset(
                dataset[key], processor, dataset_path, use_bounding_box=use_bounding_box)
        
        return dataset


@PreprocessingStrategy.register_subclass("cvc")
class CVCPreprocessingStrategy(PreprocessingStrategy):

    file_reader = "tif"

    def preprocess(
        self, 
        processor: SamProcessor, 
        valid_size: float = 0.1, 
        test_size: float = 0.1, 
        use_bounding_box: bool = True,
    ) -> DatasetDict:

        dataset_path = "data/CVC-ClinicDB"
        paths = {}
        for file in os.listdir(os.path.join(dataset_path, "Original")):
            paths[file] = {"image": os.path.join(dataset_path, "Original", file)}
        for file in os.listdir(os.path.join(dataset_path, "Ground Truth")):
            paths[file]["label"] = os.path.join(dataset_path, "Ground Truth", file)
        image = []
        label = []
        for path in paths.values():
            image.append(path["image"])
            label.append(path["label"])
        dataset = Dataset.from_dict({"image": image, "label": label})

        dataset = self.train_valid_test_split(dataset, valid_size, test_size)
        for key in dataset.keys():
            dataset[key] = PathDataset(
                dataset[key], processor, dataset_path, 
                file_reader=self.file_reader, use_bounding_box=use_bounding_box)

        return dataset


@PreprocessingStrategy.register_subclass("isic")
class ISICPreprocessingStrategy(PreprocessingStrategy):

    def preprocess(
        self, 
        processor: SamProcessor, 
        valid_size: float = 0.1, 
        test_size: float = 0.1, 
        use_bounding_box: bool = True,
    ) -> DatasetDict:

        dataset_path = "data/ISIC2016"
        train_paths = {}
        test_paths = {}
        for folder in os.listdir(dataset_path):
            for file in os.listdir(os.path.join(dataset_path, folder)):
                idx = re.search(r"\d+", file).group(0)

                if idx not in train_paths and "Training" in folder:
                    train_paths[idx] = {}
                if idx not in test_paths and "Test" in folder:
                    test_paths[idx] = {}

                if folder.endswith("Test_Data"):
                    test_paths[idx]["image"] = join(dataset_path, folder, file)
                elif folder.endswith("Test_GroundTruth"):
                    test_paths[idx]["label"] = join(dataset_path, folder, file)
                elif folder.endswith("Training_Data"):
                    train_paths[idx]["image"] = join(dataset_path, folder, file)
                elif folder.endswith("Training_GroundTruth"):
                    train_paths[idx]["label"] = join(dataset_path, folder, file)
                else:
                    raise ValueError("Unknown folder name.")
                
        train_image = []
        train_label = []
        for idx, path in train_paths.items():
            train_image.append(path["image"])
            train_label.append(path["label"])
        
        test_image = []
        test_label = []
        for idx, path in test_paths.items():
            test_image.append(path["image"])
            test_label.append(path["label"])

        dataset = Dataset.from_dict({"image": train_image, "label": train_label})
        train_valid_dataset = dataset.train_test_split(test_size=valid_size)
        test_dataset = Dataset.from_dict({"image": test_image, "label": test_label})
        
        dataset = DatasetDict({
            "train": train_valid_dataset["train"],
            "valid": train_valid_dataset["test"],
            "test": test_dataset
        })

        for key in dataset.keys():
            dataset[key] = PathDataset(
                dataset[key], processor, dataset_path, 
                use_bounding_box=use_bounding_box)

        return dataset


def get_bounding_box(ground_truth_map: np.array, add_perturbation: bool = False):
    # get bounding box from mask
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        # Return default value
        return [0, 0, ground_truth_map.shape[1], ground_truth_map.shape[0]]
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    if add_perturbation:
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox


class FileReader(ABC, RegistryMixin):

    @abstractmethod
    def __call__(self, path: str):
        pass


@FileReader.register_subclass("tif")
class TIFFileReader(FileReader):
    
    def __call__(self, path: str):
        image = tifffile.imread(path)
        image = cv2.resize(image, (256, 256)) 
        return image


@FileReader.register_subclass("default")
class DefaultFileReader(FileReader):

    def __call__(self, path: str):
        image = np.array(Image.open(path).resize((256, 256)))
        return image


class PathDataset(TorchDataset):
    """
    A Dataset where the image and label are paths to the image and label respectively.
    """
    def __init__(
        self, 
        dataset: Dataset, 
        processor: SamProcessor, 
        dataset_path: str, 
        file_reader: str = "default", 
        use_bounding_box: bool = True
    ) -> None:

        self.dataset = dataset
        self.processor = processor
        self.dataset_path = dataset_path
        self.file_reader = FileReader.create(file_reader)()
        self.use_bounding_box = use_bounding_box

    def __len__(self) -> int:
        return len(self.dataset)

    def preprocess(self, item):
        # Check file type
        image = self.file_reader(item["image"])
        if type(item["label"]) == list:
            ground_truth_mask = [self.file_reader(l) for l in item["label"]]
        else:
            ground_truth_mask = self.file_reader(item["label"])

        # Convert ground_truth_mask to binary
        prompt = None
        if type(ground_truth_mask) == list:
            prompt = [[get_bounding_box(np.max(ground_truth_mask[0] > 0, axis=-1))]]
            ground_truth_mask = [(np.max(M, axis=-1) > 0) for M in ground_truth_mask]
        elif ground_truth_mask.ndim == 3:
            ground_truth_mask = (ground_truth_mask > 0).max(axis=-1)

        # get bounding box prompt
        if prompt is None:
            prompt = [[get_bounding_box(ground_truth_mask)]]

        # Remove bounding box if not using it
        if not self.use_bounding_box:
            prompt = None
        
        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=prompt, return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs["labels"] = torch.tensor(np.array(ground_truth_mask).astype(bool)).to(inputs["pixel_values"].device)
        inputs["prompt"] = prompt

        inputs["image"] = image

        if "input_points" in item:
            inputs["input_points"] = torch.tensor(item["input_points"]).to(inputs["pixel_values"].device)
            inputs["input_labels"] = torch.tensor(item["input_labels"]).to(inputs["pixel_values"].device)

        return inputs

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        inputs = self.preprocess(item)

        return inputs
