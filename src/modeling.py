import monai
import numpy as np
import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from dataclasses import dataclass
from transformers.models.sam.modeling_sam import (
    SamModel, 
    SamPreTrainedModel,
    SamImageSegmentationOutput,
)
from typing import Optional, Callable

from src.click import ClickStrategy
from src.metrics import iou
from src.utils import RegistryMixin


@dataclass
class SamMultimaskOutput(SamImageSegmentationOutput):
    loss: torch.FloatTensor = None
    input_points: torch.Tensor = None
    iou_targets: torch.Tensor = None
    iou_pred: torch.Tensor = None
    input_labels: torch.Tensor = None
    initial_pred_masks: torch.Tensor = None
    union: torch.Tensor = None
    intersection: torch.Tensor = None


class SimilarityScorer(ABC, RegistryMixin):

    @abstractmethod
    def score(self, A: np.array, B: np.array) -> float:
        pass


@SimilarityScorer.register_subclass("iou")
class IOUScorer(SimilarityScorer):

    def score(self, A: np.array, B: np.array) -> float:
        return iou(A, B)


class Model(ABC, SamPreTrainedModel):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.sam = SamModel(config)

    @staticmethod
    def compute_loss(
        pred_masks: torch.Tensor, 
        labels: torch.Tensor, 
        loss_fn: Callable,
        return_dict: bool = False
    ):
        """
        pred_masks: (bsz, 1, num_multimask_outputs, H, W)
        labels: (bsz, H, W)
        """
        bsz, _, num_preds, H, W = pred_masks.size()

        loss = loss_fn(
            pred_masks.reshape(-1, H, W), 
            labels.repeat_interleave(num_preds, dim=0)
        ).reshape(bsz, num_preds, -1)

        loss = loss.mean(dim=2)

        if not return_dict:
            return loss.min(dim=1)[0].mean()

        _min = loss.min(dim=1)

        return {
            "loss": _min[0].mean(),
            "indices": _min[1],
        }

    @staticmethod
    def iou_loss(pred_masks, iou_pred, labels, return_iou=False):
        bsz, _, num_preds, H, W = pred_masks.size()
        iou_targets = torch.zeros(bsz, num_preds).to(pred_masks.device)
        pred_masks = pred_masks.squeeze(1).detach().cpu().numpy() > 0.5
        labels = labels.detach().cpu().numpy()
        for i in range(bsz):
            for j in range(num_preds):
                iou_targets[i, j] = iou(pred_masks[i, j], labels[i])
        iou_loss = F.mse_loss(iou_pred.squeeze(), iou_targets.squeeze())

        if return_iou:
            return iou_loss, iou_targets

        return iou_loss


class SamBaseline(Model):

    def __init__(self, config, processor, multimask_output: bool = True):
        super().__init__(config)
        self.sam = SamModel(config)
        self.processor = processor
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
        self.multimask_output = multimask_output

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder

    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        outputs = self.sam(
            image_embeddings=image_embeddings, 
            input_boxes=input_boxes,
            multimask_output=self.multimask_output,
        )

        loss = self.compute_loss(
            pred_masks=outputs.pred_masks, 
            labels=labels, 
            loss_fn=self.seg_loss, 
            return_dict=True
        )
         
        _iou_loss= self.iou_loss(outputs.pred_masks, outputs.iou_scores, labels)

        loss = loss["loss"] + _iou_loss

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=outputs.iou_scores,
            pred_masks=outputs.pred_masks,
        )


class SimSAM(Model):

    def __init__(
        self, 
        config, 
        processor, 
        num_simulations: int = 10, 
        
        click_strategy: str = "topk",
        scorer: str = "iou",
        pixel_aggregation: bool = False,
    ):
        super().__init__(config)
        self.sam = SamModel(config)
        self.seg_loss = monai.losses.DiceLoss(
            sigmoid=True, squared_pred=True, reduction="none")
        self.processor = processor
        self.num_simulations = num_simulations

        self.num_preds = 1
        self.click_strategy = ClickStrategy.create(click_strategy)()
        self.scorer = SimilarityScorer.create(scorer)()
        self.pixel_aggregation = pixel_aggregation


    def _get_clicks(
        self,
        pred_masks: torch.Tensor,
        binary_input_masks: torch.Tensor,
        original_sizes: torch.Tensor,
        reshaped_input_sizes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
    ):

        input_masks = self.processor.image_processor.post_process_masks(
            pred_masks.cpu(), original_sizes.cpu(), 
            reshaped_input_sizes.cpu(),
            binarize=False,
        )
        input_masks = torch.stack(input_masks, dim=0).to(self.device)

        input_points, input_labels = zip(*map(lambda i: self.click_strategy.get_click(
            input_mask=input_masks[i],
            binary_input_mask=binary_input_masks[i],
            label=labels[i] if labels is not None else None,
            num_samples=num_samples,
            image_embeddings=image_embeddings[i] if image_embeddings is not None else None,
            input_boxes=input_boxes[i] if input_boxes is not None else None,
        ), range(pred_masks.size(0))))

        input_points = torch.tensor(input_points).to(self.device).unsqueeze(1)
        input_labels = torch.tensor(input_labels).to(self.device).unsqueeze(1)

        ratios = reshaped_input_sizes / original_sizes
        ratios = ratios.unsqueeze(1).unsqueeze(1).expand_as(input_points)

        return input_points * ratios, input_labels

    def _aggregation(
        self,
        pred_masks,
        return_idxs=False,
    ):

        p_pred = (F.sigmoid(pred_masks) > 0.5).squeeze(1).cpu().numpy()
        shape = (p_pred.shape[0], p_pred.shape[1], p_pred.shape[1])
        similarity_matrix = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    similarity_matrix[i, j, k] = self.scorer.score(p_pred[i, j], p_pred[i, k])
        
        chosen_preds = []
        all_chosen_idxs = []
        for i in range(pred_masks.size(0)):
            chosen_idxs = torch.topk(torch.tensor(similarity_matrix[i]).mean(-1), self.num_preds).indices
            chosen_preds.append(pred_masks[i, :, chosen_idxs])
            all_chosen_idxs.append(chosen_idxs)
        chosen_preds = torch.stack(chosen_preds, dim=0)
        try:
            all_chosen_idxs = torch.stack(all_chosen_idxs).to(self.device)
        except:
            all_chosen_idxs = torch.tensor(all_chosen_idxs).to(self.device)

        if return_idxs:
            return chosen_preds, all_chosen_idxs, np.mean(similarity_matrix, axis=-1)

        return chosen_preds

    def _simulation(
        self,
        pred_masks,
        image_embeddings: torch.Tensor,
        input_boxes,
        original_sizes,
        reshaped_input_sizes,
        i=0
    ):

        input_masks = self.processor.image_processor.post_process_masks(
                pred_masks.cpu(), original_sizes.cpu(), 
                reshaped_input_sizes.cpu(),
                binarize=False
            )
        input_masks = torch.stack(input_masks, dim=0)
        binary_input_masks = (F.sigmoid(input_masks) > 0.5).float()

        input_points, input_labels = self._get_clicks(
            pred_masks=pred_masks, 
            binary_input_masks=binary_input_masks,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
            num_samples=self.num_simulations if i == 0 else 1,
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            labels=None,
        )

        bsz = input_points.size(0)
        CHUNK_SIZE = min(10, self.num_simulations)

        chunk_image_embeddings = image_embeddings.repeat_interleave(CHUNK_SIZE, dim=0)
        if input_boxes is not None:
            input_boxes = input_boxes.repeat_interleave(CHUNK_SIZE, dim=0)
        
        input_masks = (F.sigmoid(input_masks) > 0.5).repeat_interleave(CHUNK_SIZE, dim=0).float()
        all_pred_masks = []
        all_pred_iou = []

        for j in range(0, self.num_simulations, CHUNK_SIZE):

            chunk_input_points = input_points[:, :, j:j+CHUNK_SIZE].reshape(
                bsz * CHUNK_SIZE, 1, 1, 2)
            chunk_input_labels = input_labels[:, :, j:j+CHUNK_SIZE].reshape(
                bsz * CHUNK_SIZE, 1, 1)

            outputs = self.sam(
                image_embeddings=chunk_image_embeddings,
                input_points=chunk_input_points,
                input_labels=chunk_input_labels,
                input_boxes=input_boxes,
                input_masks=input_masks.squeeze(2).to(self.device),
                multimask_output=False
            )
            pred_masks = outputs.pred_masks.reshape(-1, 1, CHUNK_SIZE, *pred_masks.shape[-2:])
            pred_iou = outputs.iou_scores.reshape(-1, CHUNK_SIZE)
            all_pred_masks.append(pred_masks)
            all_pred_iou.append(pred_iou)
        
        new_pred_masks = torch.cat(all_pred_masks, dim=2)
        new_pred_iou = torch.cat(all_pred_iou, dim=1)

        return new_pred_masks, new_pred_iou

    def _forward_eval(
        self,
        pred_masks,
        image_embeddings: torch.Tensor,
        input_boxes,
        original_sizes,
        reshaped_input_sizes,
        labels
    ):
        assert pred_masks.size(0) == 1, "Only batch size 1 is supported for evaluation"
        
        new_pred_masks, _ = self._simulation(
            pred_masks=pred_masks,
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
        )
            
        assert new_pred_masks.size(2) == self.num_simulations

        loss = self.compute_loss(new_pred_masks, labels, self.seg_loss)
        original_sizes = original_sizes.repeat_interleave(self.num_simulations, dim=0)
        reshaped_input_sizes = reshaped_input_sizes.repeat_interleave(self.num_simulations, dim=0)

        union = (F.sigmoid(new_pred_masks) > 0.5).sum(dim=2, keepdim=True)
        
        # Find intersection between all masks along dim 2
        intersection = torch.ones_like(union)
        for i in range(self.num_simulations):
            intersection *= (F.sigmoid(new_pred_masks[:, :, i]) > 0.5).int().unsqueeze(2)

        pred_masks, idxs, pred_iou = self._aggregation(new_pred_masks, return_idxs=True)
        iou_scores = torch.gather(torch.tensor(pred_iou).to(idxs.device), 1, idxs).unsqueeze(1)

        # Pixel aggregation ablation
        if self.pixel_aggregation:
            pred_masks = new_pred_masks.mean(dim=2, keepdim=True)
            iou_scores = torch.tensor(pred_iou).unsqueeze(1).mean(dim=2, keepdim=True)

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=iou_scores,
            pred_masks=pred_masks,
            union=union.squeeze(),
            intersection=intersection.squeeze(),
        )

    def forward(
        self, 
        pixel_values=None,
        image_embeddings=None,
        input_boxes=None,
        labels=None,
        original_sizes=None,
        reshaped_input_sizes=None,
    ):
        
        if self.training:
            raise NotImplementedError("Training not supported yet")
        
        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        outputs = self.sam(
            image_embeddings=image_embeddings, 
            input_boxes=input_boxes, 
            multimask_output=False
        )
        
        outputs = self._forward_eval(
            pred_masks=outputs.pred_masks,
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
            labels=labels,
        )

        return outputs
