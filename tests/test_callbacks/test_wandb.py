import torch
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

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
