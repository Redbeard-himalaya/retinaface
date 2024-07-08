"""There is a lot of post processing of the predictions."""
from collections import OrderedDict
from typing import Dict, List, Union

import torch
from torch.nn import functional as F
from torchvision.ops import nms, batched_nms

from retinaface.box_utils import decode_batch, decode_landm_batch
from retinaface.network import RetinaFace
from retinaface.prior_box import priorbox
from retinaface.transform import Transformer, clip_boxes

ROUNDING_DIGITS = 2


class Model:
    def __init__(self,
                 max_size: int = 960,
                 batch_width: int = None,
                 batch_height: int = None,
                 device: str = None,
    ) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = RetinaFace(
            name="Resnet50",
            pretrained=False,
            return_layers={"layer2": 1, "layer3": 2, "layer4": 3},
            in_channels=256,
            out_channels=256,
        ).to(device=self.device)
        self.transform = Transformer(max_size=max_size, device=self.device)
        self.variance = torch.tensor([0.1, 0.2], device=self.device)
        self.original_width = torch.tensor(batch_width, device=self.device)
        self.original_width_cpu = self.original_width.cpu()
        self.original_height = torch.tensor(batch_height, device=self.device)
        self.original_height_cpu = self.original_height.cpu()
        transformed_height, transformed_width = self.transform(
            image=torch.empty((3, batch_height, batch_width), device=self.device),
        ).shape[-2:]
        transformed_size = (transformed_width, transformed_height)
        self.scale_landmarks = torch.tensor(transformed_size, device=self.device)\
                               .tile((5,)).reshape(5, 2)
        self.scale_landmarks_cpu = self.scale_landmarks.cpu()
        self.scale_bboxes = torch.tensor(transformed_size, device=self.device).tile((2,))
        self.scale_bboxes_cpu = self.scale_bboxes.cpu()
        self.resize_coeff = self.original_height / transformed_height
        self.resize_coeff_cpu = self.resize_coeff.cpu()
        self.batch_prior_box = priorbox(
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
            image_size=(transformed_height, transformed_width),
        ).to(self.device)

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        self.model.load_state_dict(state_dict)

    def eval(self) -> None:  # noqa: A003
        self.model.eval()


    def predict_jsons_origin_cpu(
            self,
            image: torch.tensor,
            confidence_threshold: float = 0.7,
            nms_threshold: float = 0.4,
    ) -> List[Dict[str, Union[List, float]]]:
        # torch cuda time measure
        # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        # how does torch cuda behavior
        # https://discuss.pytorch.org/t/escaping-if-statement-synchronization/130263/5
        if image.dim() != 4:
            raise ValueError(f"image tensor {image.shape} is not in BxCxHxW dimension")

        # import pdb; pdb.set_trace()
        with torch.inference_mode():
            transformed_image = self.transform(image=image)

            # Due to CUDA memory limit, can not infer in batch
            locs, confs, lands = [], [], []
            for torched_image in transformed_image:
                # shapes loc 1xPx4, confs 1xPx2, lands 1xPx10
                loc, conf, land = self.model(torched_image.unsqueeze(0))
                locs.append(loc)
                confs.append(conf)
                lands.append(land)

            # shapes batch_boxes BxPx4, batch_landmarks BxPx10, batch_scores BxP
            batch_boxes = decode_batch(torch.cat(locs, dim=0),
                                       self.batch_prior_box,
                                       self.variance)
            batch_landmarks = decode_landm_batch(
                torch.cat(lands, dim=0),
                self.batch_prior_box,
                self.variance,
            ).reshape(-1, self.batch_prior_box.shape[0], 5, 2)
            batch_scores = F.softmax(torch.cat(confs, dim=0), dim=-1)[:,:,1]
            batch_scores_cpu = batch_scores.cpu()
            batch_ids, valid_indeces = torch.where(batch_scores_cpu > confidence_threshold)

            results = []
            for batch_id, (boxes, scores, landmarks) in enumerate(
                    zip(batch_boxes.cpu(),
                        batch_scores_cpu,
                        batch_landmarks.cpu())
            ):
                valid_index = valid_indeces[torch.where(batch_ids == batch_id)]

                # get low score filter-outed data
                scores = scores[valid_index]
                boxes = boxes[valid_index] * self.scale_bboxes_cpu

                # do NMS
                keep = nms(boxes, scores, nms_threshold)
                boxes = boxes[keep]

                if boxes.shape[0] > 0:
                    scores = scores[keep].round(decimals=ROUNDING_DIGITS)
                    landmarks = (
                        landmarks[valid_index][keep] * self.scale_landmarks_cpu * self.resize_coeff_cpu
                    ).round(decimals=ROUNDING_DIGITS)#.cpu()
                    boxes = clip_boxes(
                        boxes=boxes,#.cpu(),
                        image_width=self.original_width_cpu,
                        image_height=self.original_height_cpu,
                        resize_coeff=self.resize_coeff_cpu,
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
                    annotations = []
                results.append(annotations)

            return results


    def predict_jsons_cpu(
            self,
            image: torch.tensor,
            confidence_threshold: float = 0.7,
            nms_threshold: float = 0.4,
    ) -> List[Dict[str, Union[List, float]]]:
        # torch cuda time measure
        # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        # how does torch cuda behavior
        # https://discuss.pytorch.org/t/escaping-if-statement-synchronization/130263/5
        if image.dim() != 4:
            raise ValueError(f"image tensor {image.shape} is not in BxCxHxW dimension")

        # import pdb; pdb.set_trace()
        with torch.inference_mode():
            transformed_image = self.transform(image=image)

            # Due to CUDA memory limit, can not infer in batch
            locs, confs, lands = [], [], []
            for torched_image in transformed_image:
                # shapes loc 1xPx4, confs 1xPx2, lands 1xPx10
                loc, conf, land = self.model(torched_image.unsqueeze(0))
                locs.append(loc)
                confs.append(conf)
                lands.append(land)

            # shapes batch_boxes BxPx4, batch_landmarks BxPx10, batch_scores BxP
            batch_boxes = decode_batch(torch.cat(locs, dim=0),
                                       self.batch_prior_box,
                                       self.variance)
            batch_landmarks = decode_landm_batch(
                torch.cat(lands, dim=0),
                self.batch_prior_box,
                self.variance,
            ).reshape(-1, self.batch_prior_box.shape[0], 5, 2)
            batch_scores = F.softmax(torch.cat(confs, dim=0), dim=-1)[:,:,1]

            # filter out valid scores, boxes, and landmarks
            batch_boxes_cpu = batch_boxes.cpu()
            batch_landmarks_cpu = batch_landmarks.cpu()
            batch_scores_cpu = batch_scores.cpu()
            batch_num = batch_scores.shape[0]
            highscore_batches, highscore_indeces = torch.where(
                batch_scores_cpu > confidence_threshold
            )
            # highscores.shape: Nx1
            highscores = batch_scores_cpu[highscore_batches, highscore_indeces]
            # highscore_boxes.shape: Nx4
            highscore_boxes = batch_boxes_cpu[highscore_batches, highscore_indeces]
            keep = batched_nms(boxes=highscore_boxes,
                               scores=highscores,
                               idxs=highscore_batches,
                               iou_threshold=nms_threshold)
            valid_batches = highscore_batches[keep]
            valid_scores = highscores[keep].round(decimals=ROUNDING_DIGITS)
            valid_boxes = clip_boxes(
                boxes=highscore_boxes[keep] * self.scale_bboxes_cpu,
                image_width=self.original_width_cpu,
                image_height=self.original_height_cpu,
                resize_coeff=self.resize_coeff_cpu,
            ).round(decimals=ROUNDING_DIGITS)
            valid_landmarks = (
                batch_landmarks_cpu[highscore_batches, highscore_indeces][keep] \
                * self.scale_landmarks_cpu * self.resize_coeff_cpu
            ).round(decimals=ROUNDING_DIGITS)

            results = []
            for batch_id in range(batch_num):
                idx = torch.where(valid_batches == batch_id)
                boxes = valid_boxes[idx]
                scores = valid_scores[idx]
                landmarks = valid_landmarks[idx]
                if boxes.shape[0] > 0:
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
                    annotations = []
                results.append(annotations)
            return results


    def predict_jsons(
            self,
            image: torch.tensor,
            confidence_threshold: float = 0.7,
            nms_threshold: float = 0.4,
    ) -> List[Dict[str, Union[List, float]]]:
        # test against 2048 max_size, 180 batch_size, achive 0.10s/f, vRAM usage 14.7G
        # torch cuda time measure
        # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        # how does torch cuda behavior
        # https://discuss.pytorch.org/t/escaping-if-statement-synchronization/130263/5
        if image.dim() != 4:
            raise ValueError(f"image tensor {image.shape} is not in BxCxHxW dimension")

        # import pdb; pdb.set_trace()
        with (torch.autocast(device_type=self.device, enabled=(self.device=="cuda")),
              torch.inference_mode(),
        ):
            transformed_image = self.transform(image=image)

            # Due to CUDA memory limit, can not infer in batch
            locs, confs, lands = [], [], []
            for torched_image in transformed_image:
                # shapes loc 1xPx4, confs 1xPx2, lands 1xPx10
                loc, conf, land = self.model(torched_image.unsqueeze(0))
                locs.append(loc)
                confs.append(conf)
                lands.append(land)

            # shapes batch_boxes BxPx4, batch_landmarks BxPx10, batch_scores BxP
            batch_boxes = decode_batch(torch.cat(locs, dim=0),
                                       self.batch_prior_box,
                                       self.variance)
            batch_landmarks = decode_landm_batch(
                torch.cat(lands, dim=0),
                self.batch_prior_box,
                self.variance,
            ).reshape(-1, self.batch_prior_box.shape[0], 5, 2)
            batch_scores = F.softmax(torch.cat(confs, dim=0), dim=-1)[:,:,1]

            # filter out valid scores, boxes, and landmarks
            batch_num = batch_scores.shape[0]
            highscore_batches, highscore_indeces = torch.where(
                batch_scores > confidence_threshold
            )
            # highscores.shape: Nx1
            highscores = batch_scores[highscore_batches, highscore_indeces]
            # highscore_boxes.shape: Nx4
            highscore_boxes = batch_boxes[highscore_batches, highscore_indeces]
            keep = batched_nms(boxes=highscore_boxes,
                               scores=highscores,
                               idxs=highscore_batches,
                               iou_threshold=nms_threshold)
            valid_batches = highscore_batches[keep]
            valid_scores = highscores[keep].round(decimals=ROUNDING_DIGITS)
            valid_boxes = clip_boxes(
                boxes=highscore_boxes[keep] * self.scale_bboxes,
                image_width=self.original_width,
                image_height=self.original_height,
                resize_coeff=self.resize_coeff,
            ).round(decimals=ROUNDING_DIGITS)
            valid_landmarks = (
                batch_landmarks[highscore_batches, highscore_indeces][keep] \
                * self.scale_landmarks * self.resize_coeff
            ).round(decimals=ROUNDING_DIGITS)

            results = []
            for batch_id in range(batch_num):
                idx = torch.where(valid_batches == batch_id)
                boxes = valid_boxes[idx]
                scores = valid_scores[idx]
                landmarks = valid_landmarks[idx]
                if boxes.shape[0] > 0:
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
                    annotations = []
                results.append(annotations)
            return results
