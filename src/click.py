import numpy as np
import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from typing import Optional

from src.utils import RegistryMixin

class ClickStrategy(ABC, RegistryMixin):

    @abstractmethod
    def get_click(
        self, 
        input_mask: torch.Tensor, 
        binary_input_mask: torch.Tensor,
        label: Optional[torch.Tensor] = None, 
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None
    ):
        pass



@ClickStrategy.register_subclass("random")
class RandomClickStrategy(ClickStrategy):
    """
    Ablation that just randomly samples clicks without strategy
    """

    def get_click(
        self, 
        input_mask: torch.Tensor, 
        binary_input_mask: torch.Tensor,
        label: Optional[torch.Tensor] = None, 
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None
    ):

        error_mask = torch.ones(input_mask.shape).to("cuda")
        probs = (error_mask / error_mask.sum()).reshape(-1)

        binary_input_mask = binary_input_mask.squeeze().cpu().numpy()
        H, W = binary_input_mask.shape[-2:]

        idxs = torch.multinomial(probs, num_samples, replacement=True).cpu().numpy()

        clicks = [[idx % W, idx // W] for idx in idxs]
        labels = 1.0 - binary_input_mask.reshape(-1)[idxs]

        return clicks, labels


@ClickStrategy.register_subclass("topk")
class SamplingClickStrategy(ClickStrategy):

    def get_click(
        self, 
        input_mask: torch.Tensor, 
        binary_input_mask: torch.Tensor,
        label: Optional[torch.Tensor] = None, 
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None
    ):

        fp_mask = 1.0 - F.sigmoid(input_mask).squeeze().cpu().numpy()
        fn_mask = F.sigmoid(input_mask).squeeze().cpu().numpy()

        # Set fp_mask to 0.0 where binary_input_mask is 0
        fp_mask = np.where(binary_input_mask.squeeze().cpu().numpy() == 0, 0.0, fp_mask)
        # Set fn_mask to 0.0 where binary_input_mask is 1
        fn_mask = np.where(binary_input_mask.squeeze().cpu().numpy() == 1, 0.0, fn_mask)

        error_mask = torch.tensor(fp_mask + fn_mask).to("cuda")
        topk = torch.topk(error_mask.reshape(-1), num_samples, largest=True)
        idxs = topk.indices.cpu().numpy()

        binary_input_mask = binary_input_mask.squeeze().cpu().numpy()
        H, W = binary_input_mask.shape[-2:]

        clicks = [[idx % W, idx // W] for idx in idxs]
        labels = 1.0 - binary_input_mask.reshape(-1)[idxs]
        
        
        return clicks, labels
