from typing import Any, Dict, List, Union

import cv2
import numpy as np
import torch


def vis_annotations(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    vis_image = image.copy()

    for annotation in annotations:
        landmarks = annotation["landmarks"]

        colors = [(255, 0, 0), (128, 255, 0), (255, 178, 102), (102, 128, 255), (0, 255, 255)]

        for landmark_id, (x, y) in enumerate(landmarks):
            vis_image = cv2.circle(vis_image, (int(x), int(y)), radius=3, color=colors[landmark_id], thickness=3)

        x_min, y_min, x_max, y_max = (int(tx) for tx in annotation["bbox"])

        x_min = np.clip(x_min, 0, x_max - 1)
        y_min = np.clip(y_min, 0, y_max - 1)

        vis_image = cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    return vis_image


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


def process_predictions(
        batch_num: int,
        batches: torch.Tensor,
        boxes: torch.Tensor,
        landmarks: torch.Tensor,
        scores: torch.Tensor,
) -> List[List[Dict[str, Union[List, float]]]]:
    results = []
    for batch_id in range(batch_num):
        idx = torch.where(batches == batch_id)
        _boxes = boxes[idx]
        _scores = scores[idx]
        _landmarks = landmarks[idx]
        if _boxes.shape[0] > 0:
            annotations = [
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "score": score,
                    "landmarks": landmark.tolist(),
                }
                for score, (x_min, y_min, x_max, y_max), landmark \
                in zip(_scores, _boxes, _landmarks) \
                if x_min < x_max and y_min < y_max
            ]
        else:
            annotations = []
        results.append(annotations)
    return results
