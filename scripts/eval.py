import os
import sys
import torch
sys.path.append(os.path.dirname(sys.path[0]))  # add root folder to sys.path

from dataclasses import dataclass, field
from datasets import set_caching_enabled
from functools import partial
from transformers import ( 
    Trainer, 
    TrainingArguments,
    HfArgumentParser,
)
from transformers import SamProcessor
from transformers.utils import logging

from src import constants
from src.corpora import PreprocessingStrategy
from src.metrics import compute_metrics
from src.modeling import SamBaseline, SimSAM
from src.utils import set_seed

set_caching_enabled(False)

set_seed()
torch.set_grad_enabled(False)
logger = logging.get_logger()
logging.set_verbosity_info()


@dataclass
class Arguments:
    model_load_path: str = field(
        default="facebook/sam-vit-base",
        metadata={"help": "Path to the pretrained model or model identifier from huggingface.co/models"}
    )

    processor_load_path: str = field(
        default="facebook/sam-vit-base",
        metadata={"help": "Path to the pretrained processor or processor identifier from huggingface.co/models"}
    )

    dataset: str = field(
        default="busi",
        metadata={"help": "Path to the dataset or dataset identifier from huggingface.co/datasets",
                    "choices": ["busi", "cvc", "isic"]}
    )

    model_type: str = field(
        default="simsam",
        metadata={"help": "Model type", "choices": ["simsam", "baseline"]}
    )

    ablation: str = field(
        default="none",
        metadata={"help": "Ablation type", "choices": ["k=1", "random", "pixel_aggregation", "none"]}
    )

    write_path: str = field(
        default="data/results.json",
        metadata={"help": "Path to write results"}
    )


def _evaluate(model, dataset, write_path="data/results.json"):
    evaluation_arguments = TrainingArguments(
        output_dir="data/", per_device_eval_batch_size=1
    )

    evaluator = Trainer(
        model=model,
        args=evaluation_arguments,
        eval_dataset=dataset,
        compute_metrics=partial(compute_metrics, write_path=write_path),
    )

    results = evaluator.evaluate()

    return results

def main():
    parser = HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()
    processor = SamProcessor.from_pretrained(args.processor_load_path)

    if args.model_type == "baseline":
        model = SamBaseline.from_pretrained(
            args.model_load_path, processor=processor, multimask_output=False)
    
    elif args.model_type == "simsam":
        model = SimSAM.from_pretrained(
            "facebook/sam-vit-base", 
            processor=processor, 
            num_simulations=constants.NUM_SIMULATIONS if args.ablation != "k=1" else 1,
            pixel_aggregation=args.ablation == "pixel_aggregation",
            click_strategy=args.ablation if args.ablation == "random" else "topk",
        )
    
    else:
        raise ValueError("Invalid model type")

    preprocessing = PreprocessingStrategy.create(args.dataset)()
    dataset = preprocessing.preprocess(
        processor, valid_size=constants.VALID_SIZE, test_size=constants.TEST_SIZE)

    # Evaluate
    results = _evaluate(model, dataset["test"], write_path=args.write_path)
    logger.info(results)
    print(f"Latency: {model.total_time / len(dataset['test']):.3f}s")


if __name__ == "__main__":
    main()
