from typing import Any, ClassVar, Dict, Final, List, Optional, Type, TypeVar, Union

import torch
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor
from torchvision.ops import box_convert, clip_boxes_to_image, remove_small_boxes
from torchvision.tv_tensors import BoundingBoxes

from ..structs import Mode
from ..tasks import Task
from .base import LoggerIntegration


ALL_MODES: Final = [Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT]
T = TypeVar("T")
L = TypeVar("L", bound=Logger)
TaskIdentifier = Union[str, int, Type[Task]]


class WandBLoggerIntegration(LoggerIntegration):
    r"""Abstraction for integrating with a logger. Implementations of this class
    should be able to take a logging target prepared by the :class:`LoggingCallback`
    and log it to the appropriate logger.
    """
    logger_type: ClassVar[Type[Logger]] = WandbLogger

    @staticmethod
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
