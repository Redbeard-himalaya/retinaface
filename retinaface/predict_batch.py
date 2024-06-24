"""There is a lot of post processing of the predictions."""
from collections import OrderedDict
from typing import Dict, List, Union

import torch
from torch.nn import functional as F
from torchvision.ops import nms

from retinaface.box_utils import decode, decode_landm
from retinaface.network import RetinaFace
from retinaface.prior_box import priorbox
from retinaface.transform import Transformer, clip_boxes

ROUNDING_DIGITS = 2


class Model:
    def __init__(self, max_size: int = 960, device: str = "cpu") -> None:
        self.model = RetinaFace(
            name="Resnet50",
            pretrained=False,
            return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
            in_channels=256,
            out_channels=256,
        ).to(device)
        self.device = device
        self.transform = Transformer(max_size=max_size)
        self.max_size = max_size
        self.variance = [0.1, 0.2]

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        self.model.load_state_dict(state_dict)

    def eval(self) -> None:  # noqa: A003
        self.model.eval()

    def predict_jsons(
        self, image: torch.tensor, confidence_threshold: float = 0.7, nms_threshold: float = 0.4
    ) -> List[Dict[str, Union[List, float]]]:
        with torch.no_grad():
            original_height, original_width = image.shape[1:]

            transformed_image = self.transform(image=image)

            transformed_height, transformed_width = transformed_image.shape[1:]
            transformed_size = (transformed_width, transformed_height)

            scale_landmarks = torch.tensor(transformed_size).tile((5,)).to(self.device)
            scale_bboxes = torch.tensor(transformed_size).tile((2,)).to(self.device)

            prior_box = priorbox(
                min_sizes=[[16, 32], [64, 128], [256, 512]],
                steps=[8, 16, 32],
                clip=False,
                image_size=(transformed_height, transformed_width),
            ).to(self.device)

            torched_image = transformed_image

            loc, conf, land = self.model(torched_image.unsqueeze(0))  # pylint: disable=E1102

            conf = F.softmax(conf, dim=-1)

            boxes = decode(loc.data[0], prior_box, self.variance)

            boxes *= scale_bboxes
            scores = conf[0][:, 1]

            landmarks = decode_landm(land.data[0], prior_box, self.variance)
            landmarks *= scale_landmarks

            # ignore low scores
            valid_index = torch.where(scores > confidence_threshold)[0]
            boxes = boxes[valid_index]
            landmarks = landmarks[valid_index]
            scores = scores[valid_index]

            # do NMS
            keep = nms(boxes, scores, nms_threshold)
            boxes = boxes[keep, :]

            if boxes.shape[0] == 0:
                return [{"bbox": [], "score": -1, "landmarks": []}]

            landmarks = landmarks[keep]

            resize_coeff = original_height / transformed_height
            scores = scores[keep].round(decimals=ROUNDING_DIGITS).cpu()
            landmarks = (landmarks.reshape(-1, 10) * resize_coeff)\
                .round(decimals=ROUNDING_DIGITS)\
                .reshape(-1, 5, 2)\
                .cpu()
            boxes = clip_boxes(boxes.cpu(),
                               image_width=original_width,
                               image_height=original_height,
                               resize_coeff=resize_coeff).round(decimals=ROUNDING_DIGITS)

            annotations: List[Dict[str, Union[List, float]]] = []

            # import pdb; pdb.set_trace()
            for score, bbox, landmark in zip(scores, boxes, landmarks):
                x_min, y_min, x_max, y_max = bbox
                if x_min >= x_max or y_min >= y_max:
                    continue
                annotations += [
                    {
                        "bbox": bbox.tolist(),
                        "score": score,
                        "landmarks": landmark.tolist(),
                    }
                ]
            return annotations
