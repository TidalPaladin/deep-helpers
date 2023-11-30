from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor
from torchvision.ops import box_convert, clip_boxes_to_image, remove_small_boxes
from torchvision.tv_tensors import BoundingBoxes

from .base import LoggerIntegration
from .image import overlay_heatmap


try:
    import wandb
except ImportError:
    raise ImportError(
        "Unable to import `wandb`. Please install the 'wandb' extra of `deep-helpers` to use this logger integration."
    )


class WandBLoggerIntegration(LoggerIntegration):
    logger_type: ClassVar[Type[Logger]] = WandbLogger

    @staticmethod
    @torch.no_grad()
    def boxes_to_wandb(
        boxes: BoundingBoxes,
        scores: Dict[str, Tensor] = {},
        captions: List[str] = [],
        class_ids: Optional[Tensor] = None,
        class_labels: Dict[int, str] = {},
        min_size: float = 1,
    ) -> Dict[str, Any]:
        r"""Converts bounding boxes to a format that can be logged to WandB.

        Args:
            boxes: The bounding boxes to log in absolute coordinates with shape :math:`(N, 4)`.
            scores: A dictionary of scores for each bounding box. Keys should identify the metric
                name and values should be tensors with shape :math:`(N,)`.
            captions: A list of captions for each bounding box with length :math:`N`.
            class_ids: A tensor of class IDs for each bounding box with shape :math:`(N,)`.
            class_labels: A dictionary mapping class IDs to class labels.
            min_size: The minimum size of a bounding box to log. Boxes smaller than this will be
                ignored. Units are in pixels, and removal will be determined prior to converting
                to relative coordinates.

        Returns:
            The converted bounding boxes.
        """
        # Validate boxes and determine the number of boxes
        if not isinstance(boxes, BoundingBoxes):
            raise TypeError(f"Expected boxes to be of type BoundingBoxes, got {type(boxes)}")
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError(f"Expected boxes to have shape (N, 4), got {boxes.shape}")

        # Validate other inputs against expected number of boxes
        for k, v in scores.items():
            if len(v) != len(boxes):
                raise ValueError(f"Expected scores `{k}` to have length {len(boxes)}, got {len(v)}")
        if len(captions) != len(boxes):
            raise ValueError(f"Expected captions to have length {len(boxes)}, got {len(captions)}")
        if class_ids is not None and len(class_ids) != len(boxes):
            raise ValueError(f"Expected class_ids to have length {len(boxes)}, got {len(class_ids)}")

        # Convert boxes to xyxy format and clip them to the image
        H, W = boxes.canvas_size
        input_format = boxes.format
        processed_boxes = box_convert(boxes, input_format.value.lower(), "xyxy")
        processed_boxes = clip_boxes_to_image(processed_boxes, (H, W))

        # Get mask for boxes that meet the size requirement
        meets_size_requirement = boxes.new_full((len(boxes),), False, dtype=torch.bool)
        meets_size_requirement[remove_small_boxes(processed_boxes, min_size)] = True
        assert len(meets_size_requirement) == len(boxes)

        # Convert boxes to relative coordinates
        processed_boxes = processed_boxes.float() / processed_boxes.new_tensor([W, H, W, H])
        assert ((0 <= processed_boxes) & (processed_boxes <= 1)).all()

        box_data: List[Dict[str, Any]] = []
        for i, (box, valid_size) in enumerate(zip(boxes, meets_size_requirement)):
            if not valid_size:
                continue

            # Build the box data
            metadata = {
                "position": {
                    "minX": float(box[0]),
                    "minY": float(box[1]),
                    "maxX": float(box[2]),
                    "maxY": float(box[3]),
                },
                "class_id": int(class_ids[i]) if class_ids is not None else None,
                "box_caption": captions[i] if captions else None,
                "scores": {k: float(v[i]) for k, v in scores.items()} or None,
            }

            # Pop any fields that are None
            metadata = {k: v for k, v in metadata.items() if v is not None}
            box_data.append(metadata)

        return {
            "box_data": box_data,
            "class_labels": class_labels,
        }

    @staticmethod
    @torch.no_grad()
    def image_to_wandb(
        img: Tensor,
        caption: Optional[str] = None,
        grouping: Optional[int] = None,
        classes: Optional[Sequence[Dict]] = None,
        boxes: Optional[Dict[str, Any]] = None,
        masks: Optional[Dict[str, Any]] = None,
        heatmap: Optional[Tensor] = None,
        heatmap_alpha: float = 0.5,
        max_size: Optional[Tuple[int, int]] = None,
    ) -> wandb.Image:
        """
        Converts an image to a wandb.Image object.

        Args:
            img: The image to convert. If the image is floating point, it will be converted to
                uint8 and scaled to the range [0, 255]. It is assumed that RGB images are in
                channels-first format. Floating point and byte images are supported.
            caption: The caption for the image.
            grouping: The grouping for the image.
            classes: The classes for the image.
            boxes: The bounding boxes for the image.
            masks: The masks for the image.
            heatmap: Optional heatmap to overlay on the image. Should be a single-channel image
                with floating point values in the range [0, 1].
            heatmap_alpha: The alpha value to use for blending with the heatmap.
            max_size: The maximum size for the image. If the image is larger than this, it will
                resized to this size using bilinear interpolation.

        Shapes:
            * ``img`` - :math:`(C, H, W)` or :math:`(H, W)`
            * ``heatmap`` - :math:`(1, H, W)` or :math:`(H, W)`

        Returns:
            The converted image object.
        """
        H, W = img.shape[-2:]
        prepared_img = img.clone()

        # Resize image
        if max_size is not None:
            prepared_img = F.interpolate(prepared_img.view(1, -1, H, W), size=max_size, mode="bilinear").squeeze_(0)
            heatmap = (
                F.interpolate(heatmap.view(1, -1, H, W), size=max_size, mode="bilinear").squeeze_(0)
                if heatmap is not None
                else None
            )
        H, W = img.shape[-2:]

        # Overlay heatmap if provided
        if heatmap is not None:
            prepared_img = overlay_heatmap(
                heatmap.view(1, 1, H, W),
                prepared_img.view(1, -1, H, W),
                alpha=heatmap_alpha,
            ).view(-1, H, W)

        # If float, convert to uint8
        if prepared_img.is_floating_point():
            prepared_img = prepared_img.mul_(255).clamp_(0, 255).byte()

        # If RGB, convert to channels-last
        H, W = prepared_img.shape[-2:]
        prepared_img = prepared_img.view(-1, H, W)
        if prepared_img.shape[0] == 3:
            prepared_img = prepared_img.movedim(0, -1)

        return wandb.Image(
            prepared_img.cpu().numpy(),
            caption=caption,
            grouping=grouping,
            classes=classes,
            boxes=boxes,
            masks=masks,
        )
