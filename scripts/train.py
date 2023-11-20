"""
Script to fine-tune SAM on a dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))  # add root folder to sys.path

from datasets import set_caching_enabled    
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser, 
    SamProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging

from src import constants
from src.corpora import PreprocessingStrategy
from src.metrics import compute_metrics
from src.modeling import SamBaseline
from src.utils import set_seed

set_seed()
logger = logging.get_logger()
logging.set_verbosity_info()
set_caching_enabled(False)

@dataclass
class ModelArguments:
    model_load_path: str = field(
        default="facebook/sam-vit-base",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    processor_load_path: str = field(
        default="facebook/sam-vit-base",
        metadata={"help": "Path to the pretrained processor or processor identifier from huggingface.co/models"}
    )

    model_save_path: str = field(
        default="data/cvc_baseline_1",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    dataset: str = field(
        default="cvc",
        metadata={"help": "Path to the dataset or dataset identifier from huggingface.co/datasets",
                    "choices": ["busi", "cvc", "isic"]}
    )

    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"}
    )

    num_train_epochs: int = field(
        default=10,
        metadata={"help": "Number of training epochs"}
    )

    use_bounding_box: bool = field(
        default=True,
        metadata={"help": "Whether to use bounding boxes"}
    )


def _main(args: ModelArguments):
    # Load dataset
    processor = SamProcessor.from_pretrained(args.processor_load_path)
    preprocessing = PreprocessingStrategy.create(args.dataset)()
    dataset = preprocessing.preprocess(
        processor, 
        valid_size=constants.VALID_SIZE, 
        test_size=constants.TEST_SIZE, 
        use_bounding_box=args.use_bounding_box
    )

    # Load model
    model = SamBaseline.from_pretrained(
        args.model_load_path,
        processor=processor,
        multimask_output=False,
    )

    # Print number of parameters
    print(f"Number of parameters: {model.num_parameters()}")

    # Make sure we only compute gradients for mask decoder
    for name, param in model.named_parameters():
        if name.startswith("sam.vision_encoder") or name.startswith("sam.prompt_encoder"):
            param.requires_grad_(False)
    
    # Set up trainer
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        dataloader_drop_last=False,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained(args.model_save_path)


def main():
    parser = HfArgumentParser((ModelArguments,))
    args, = parser.parse_args_into_dataclasses()
    _main(args)


if __name__ == "__main__":
    main()
