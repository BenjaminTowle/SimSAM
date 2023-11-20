import monai
import numpy as np
import torch
import torch.nn.functional as F

from abc import ABC
from dataclasses import dataclass
from transformers import SamProcessor
from transformers.models.sam.modeling_sam import (
    SamModel, 
    SamPreTrainedModel,
    SamImageSegmentationOutput,
    SamConfig,
)
from typing import Optional, Callable, Union, Tuple
from time import perf_counter

from src.click import ClickStrategy
from src.metrics import iou


@dataclass
class SamMultimaskOutput(SamImageSegmentationOutput):
    loss: torch.Tensor = None
    union: torch.Tensor = None
    intersection: torch.Tensor = None


class Model(ABC, SamPreTrainedModel):

    def __init__(self, config: SamConfig) -> None:
        super().__init__(config)
        self.sam = SamModel(config)

    @staticmethod
    def compute_loss(
        pred_masks: torch.Tensor, 
        labels: torch.Tensor, 
        loss_fn: Callable,
        return_dict: bool = False
    ) -> dict:
        
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
    def iou_loss(
        pred_masks: torch.Tensor, 
        iou_pred: torch.Tensor, 
        labels: torch.Tensor, 
        return_iou: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        
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

    def __init__(
        self, 
        config: SamConfig, 
        processor: SamProcessor, 
    ) -> None:

        super().__init__(config)
        self.sam = SamModel(config)
        self.processor = processor
        self.seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="none")
        
        self.total_time = 0.0

    @property
    def mask_decoder(self):
        return self.sam.mask_decoder

    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SamMultimaskOutput:

        t = perf_counter()

        if image_embeddings is None:
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        outputs = self.sam(
            image_embeddings=image_embeddings, 
            input_boxes=input_boxes,
            multimask_output=False,
        )

        loss = self.compute_loss(
            pred_masks=outputs.pred_masks, 
            labels=labels, 
            loss_fn=self.seg_loss, 
            return_dict=True
        )
         
        _iou_loss= self.iou_loss(outputs.pred_masks, outputs.iou_scores, labels)

        loss = loss["loss"] + _iou_loss

        self.total_time += perf_counter() - t

        return SamMultimaskOutput(
            loss=loss,
            iou_scores=outputs.iou_scores,
            pred_masks=outputs.pred_masks,
        )


class SimSAM(Model):

    def __init__(
        self, 
        config: SamConfig, 
        processor: SamProcessor, 
        num_simulations: int = 10, 
        click_strategy: str = "topk",
        pixel_aggregation: bool = False,
        output_union_and_intersection: bool = False,
        chunk_size: int = 50 # Enables batching of simulations to save memory
    ) -> None:
    
        super().__init__(config)
        self.sam = SamModel(config)
        self.seg_loss = monai.losses.DiceLoss(
            sigmoid=True, squared_pred=True, reduction="none")
        self.processor = processor
        self.num_simulations = num_simulations

        self.num_preds = 1
        self.click_strategy = ClickStrategy.create(click_strategy)()
        self.pixel_aggregation = pixel_aggregation
        self.output_union_and_intersection = output_union_and_intersection
        self.chunk_size = chunk_size

        self.total_time = 0.0


    def _get_clicks(
        self,
        pred_masks: torch.Tensor,
        binary_input_masks: torch.Tensor,
        original_sizes: torch.Tensor,
        reshaped_input_sizes: torch.Tensor,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        input_masks = self.processor.image_processor.post_process_masks(
            pred_masks.cpu(), original_sizes.cpu(), 
            reshaped_input_sizes.cpu(),
            binarize=False,
        )
        input_masks = torch.stack(input_masks, dim=0).to(self.device)

        input_points, input_labels = zip(*map(lambda i: self.click_strategy.get_click(
            input_mask=input_masks[i],
            binary_input_mask=binary_input_masks[i],
            num_samples=num_samples,
        ), range(pred_masks.size(0))))

        input_points = torch.tensor(input_points).to(self.device).unsqueeze(1)
        input_labels = torch.tensor(np.array(input_labels)).to(self.device).unsqueeze(1)

        ratios = reshaped_input_sizes / original_sizes
        ratios = ratios.unsqueeze(1).unsqueeze(1).expand_as(input_points)

        return input_points * ratios, input_labels


    def _aggregation(
        self,
        pred_masks: torch.Tensor,
        return_idxs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, np.array]]:

        b_masks = (F.sigmoid(pred_masks) > 0.5).squeeze(1).squeeze(0)  # (num_simulations, H, W)
        p_pred = b_masks.unsqueeze(1).expand(
            self.num_simulations, self.num_simulations, *pred_masks.shape[-2:])

        intersection = torch.logical_and(p_pred, p_pred.transpose(0, 1)).float()
        union = torch.logical_or(p_pred, p_pred.transpose(0, 1)).float()

        similarity_matrix = torch.sum(intersection, dim=(-1, -2)) / torch.sum(union, dim=(-1, -2))
        # set nan values to 0
        similarity_matrix[similarity_matrix != similarity_matrix] = 0.0
        scores = similarity_matrix.unsqueeze(0).mean(-1)

        chosen_preds = []
        all_chosen_idxs = []
        for i in range(pred_masks.size(0)):
            chosen_idxs = torch.topk(scores[i], self.num_preds).indices
            chosen_preds.append(pred_masks[i, :, chosen_idxs])
            all_chosen_idxs.append(chosen_idxs)
        chosen_preds = torch.stack(chosen_preds, dim=0)
        all_chosen_idxs = torch.stack(all_chosen_idxs).to(self.device)

        if return_idxs:
            return chosen_preds, all_chosen_idxs, scores.cpu().numpy()

        return chosen_preds

    def _simulation(
        self,
        pred_masks: torch.Tensor,
        image_embeddings: torch.Tensor,
        input_boxes: torch.Tensor,
        original_sizes: torch.Tensor,
        reshaped_input_sizes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

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
            num_samples=self.num_simulations,
        )

        bsz = input_points.size(0)
        chunk_size = min(self.chunk_size, self.num_simulations)

        chunk_image_embeddings = image_embeddings.repeat_interleave(chunk_size, dim=0)
        if input_boxes is not None:
            input_boxes = input_boxes.repeat_interleave(chunk_size, dim=0)
        
        input_masks = (F.sigmoid(input_masks) > 0.5).repeat_interleave(chunk_size, dim=0).float().squeeze(2).to(self.device)
        all_pred_masks = []
        all_pred_iou = []

        for j in range(0, self.num_simulations, chunk_size):

            chunk_input_points = input_points[:, :, j:j+chunk_size].reshape(
                bsz * chunk_size, 1, 1, 2)
            chunk_input_labels = input_labels[:, :, j:j+chunk_size].reshape(
                bsz * chunk_size, 1, 1)

            outputs = self.sam(
                image_embeddings=chunk_image_embeddings,
                input_points=chunk_input_points,
                input_labels=chunk_input_labels,
                input_boxes=input_boxes,
                input_masks=input_masks,
                multimask_output=False
            )
            
            pred_masks = outputs.pred_masks.reshape(-1, 1, chunk_size, *pred_masks.shape[-2:])
            pred_iou = outputs.iou_scores.reshape(-1, chunk_size)
            all_pred_masks.append(pred_masks)
            all_pred_iou.append(pred_iou)
        
        new_pred_masks = torch.cat(all_pred_masks, dim=2)
        new_pred_iou = torch.cat(all_pred_iou, dim=1)

        return new_pred_masks, new_pred_iou

    def _forward_eval(
        self,
        pred_masks: torch.Tensor,
        image_embeddings: torch.Tensor,
        input_boxes: torch.Tensor,
        original_sizes: torch.Tensor,
        reshaped_input_sizes: torch.Tensor,
        labels: torch.Tensor,
    ) -> SamMultimaskOutput:
        
        assert pred_masks.size(0) == 1, "Only batch size 1 is supported for evaluation"
        
        new_pred_masks, _ = self._simulation(
            pred_masks=pred_masks,
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
        )

        loss = self.compute_loss(new_pred_masks, labels, self.seg_loss)

        # Find union and intersection between all masks along dim 2
        union, intersection = None, None
        if self.output_union_and_intersection:
            union = (F.sigmoid(new_pred_masks) > 0.5).sum(dim=2, keepdim=True)
            intersection = torch.ones_like(union)
            for i in range(self.num_simulations):
                intersection *= (F.sigmoid(new_pred_masks[:, :, i]) > 0.5).int().unsqueeze(2)
            union = union.squeeze()
            intersection = intersection.squeeze()

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
            union=union,
            intersection=intersection,
        )

    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        original_sizes: Optional[torch.Tensor] = None,
        reshaped_input_sizes: Optional[torch.Tensor] = None
    ) -> SamMultimaskOutput:
        
        t = perf_counter()
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
        self.total_time += perf_counter() - t

        return outputs
