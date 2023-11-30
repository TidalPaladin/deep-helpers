from typing import Tuple

import pytest
import torch
import wandb
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image

from deep_helpers.callbacks import WandBLoggerIntegration


def test_boxes_to_wandb():
    boxes = BoundingBoxes(
        torch.tensor([[0, 0, 10, 10], [5, 5, 12, 12]]),
        format=BoundingBoxFormat.XYXY,
        canvas_size=(10, 10),
    )

    scores = {"score": torch.tensor([0.2, 0.8])}
    captions = ["img1", "img2"]
    class_ids = torch.tensor([0, 1])
    class_labels = {0: "test", 1: "test"}
    min_size = 1

    actual = WandBLoggerIntegration.boxes_to_wandb(
        boxes=boxes,
        scores=scores,
        captions=captions,
        class_ids=class_ids,
        class_labels=class_labels,
        min_size=min_size,
    )

    expected = {
        "box_data": [
            {
                "position": {"minX": 0.0, "minY": 0.0, "maxX": 10.0, "maxY": 10.0},
                "class_id": 0,
                "box_caption": "img1",
                # Floating point precision is lost when converting to json so reconvert when comparing
                "scores": {"score": float(scores["score"][0])},
            },
            {
                "position": {"minX": 5.0, "minY": 5.0, "maxX": 12.0, "maxY": 12.0},
                "class_id": 1,
                "box_caption": "img2",
                "scores": {"score": float(scores["score"][1])},
            },
        ],
        "class_labels": class_labels,
    }
    assert actual == expected


@pytest.mark.parametrize(
    "dtype,size,heatmap",
    [
        (torch.float32, (3, 32, 32), False),
        (torch.float32, (32, 32), False),
        (torch.uint8, (3, 32, 32), False),
    ],
)
def test_image_to_wandb(dtype: torch.dtype, size: Tuple[int, ...], heatmap: bool):
    img = torch.rand(*size) if dtype.is_floating_point else torch.randint(0, 255, size, dtype=dtype)
    img = Image(img)
    heatmap_tensor = torch.rand(1, *size) if heatmap else None
    result = WandBLoggerIntegration.image_to_wandb(img, heatmap=heatmap_tensor)
    assert isinstance(result, wandb.Image)
