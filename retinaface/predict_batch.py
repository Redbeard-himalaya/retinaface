"""There is a lot of post processing of the predictions."""
from collections import OrderedDict
from typing import Dict, List, Union

import torch
from torch.nn import functional as F
from torchvision.ops import nms

from retinaface.box_utils import decode_batch, decode_landm_batch
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
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.transform = Transformer(max_size=max_size, device=self.device)
        self.max_size = max_size
        self.variance = [0.1, 0.2]

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        self.model.load_state_dict(state_dict)

    def eval(self) -> None:  # noqa: A003
        self.model.eval()

    def predict_jsons(
        self, image: torch.tensor, confidence_threshold: float = 0.7, nms_threshold: float = 0.4
    ) -> List[Dict[str, Union[List, float]]]:
        assert len(image.shape) == 4, f"image tensor {image.shape} is not in BxCxHxW dimension"
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            original_height, original_width = image.shape[-2:]

            transformed_image = self.transform(image=image)

            transformed_height, transformed_width = transformed_image.shape[-2:]
            transformed_size = (transformed_width, transformed_height)

            scale_landmarks = torch.tensor(transformed_size)\
                                   .tile((5,)).reshape(5, 2).to(self.device)
            scale_bboxes = torch.tensor(transformed_size).tile((2,)).to(self.device)
            resize_coeff = original_height / transformed_height

            # prior_box shape Px4
            prior_box = priorbox(
                min_sizes=[[16, 32], [64, 128], [256, 512]],
                steps=[8, 16, 32],
                clip=False,
                image_size=(transformed_height, transformed_width),
            ).to(self.device)

            # Due to CUDA memory limit, can not infer in batch
            locs, confs, lands = [], [], []
            for torched_image in transformed_image:
                # shapes loc 1xPx4, confs 1xPx2, lands 1xPx10
                loc, conf, land = self.model(torched_image.unsqueeze(0))
                locs.append(loc)
                confs.append(conf)
                lands.append(land)

            # shapes batch_boxes BxPx4, batch_landmarks BxPx10, batch_scores BxP
            batch_boxes = decode_batch(torch.cat(locs, dim=0), prior_box, self.variance)
            batch_landmarks = decode_landm_batch(
                torch.cat(lands, dim=0),
                prior_box,
                self.variance,
            ).reshape(-1, prior_box.shape[0], 5, 2)
            batch_scores = F.softmax(torch.cat(confs, dim=0), dim=-1)[:,:,1]
            batch_ids, valid_indeces = torch.where(batch_scores > confidence_threshold)

            results = []
            for batch_id, (boxes, scores, landmarks) in enumerate(
                    zip(batch_boxes,
                        batch_scores,
                        batch_landmarks)
            ):
                # valid_index = torch.where(scores > confidence_threshold)[0]
                valid_index = valid_indeces[torch.where(batch_ids == batch_id)[0]]

                # get low score filter-outed data
                scores = scores[valid_index]
                boxes = boxes[valid_index] * scale_bboxes

                # do NMS
                keep = nms(boxes, scores, nms_threshold)
                boxes = boxes[keep]

                if boxes.shape[0] > 0:
                    scores = scores[keep].round(decimals=ROUNDING_DIGITS)#.cpu()
                    landmarks = (
                        landmarks[valid_index][keep] * scale_landmarks * resize_coeff
                    ).round(decimals=ROUNDING_DIGITS)#.cpu()
                    boxes = clip_boxes(
                        boxes,#.cpu(),
                        image_width=original_width,
                        image_height=original_height,
                        resize_coeff=resize_coeff,
                    ).round(decimals=ROUNDING_DIGITS)

                    annotations = [
                        {
                            "bbox": [x_min, y_min, x_max, y_max],
                            "score": score,
                            "landmarks": landmark.tolist(),
                        }
                        for score, (x_min, y_min, x_max, y_max), landmark \
                        in zip(scores, boxes, landmarks) \
                        if x_min < x_max and y_min < y_max
                    ]
                else:
                    annotations = [{"bbox": [], "score": -1, "landmarks": []}]
                results.append(annotations)

            return results
