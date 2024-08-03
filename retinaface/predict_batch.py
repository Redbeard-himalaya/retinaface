"""There is a lot of post processing of the predictions."""
from collections import OrderedDict
from typing import Dict, Tuple, Union

import torch
from torch.nn import functional as F
from torchvision.ops import nms, batched_nms

from retinaface.box_utils import decode_batch, decode_landm_batch
from retinaface.network import RetinaFace
from retinaface.prior_box import priorbox
from retinaface.transform import Transformer, clip_boxes
from retinaface.extract import Extractor

ROUNDING_DIGITS = 2


class Model:
    def __init__(self,
                 max_size: int = 960,
                 face_size: int = 112,
                 margin: int = 0,
                 device: str = None,
    ) -> None:
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = RetinaFace(
            name="Resnet50",
            pretrained=False,
            return_layers={"layer2": '1', "layer3": '2', "layer4": '3'},
            in_channels=256,
            out_channels=256,
        ).to(device=self.device)
        self.transform = Transformer(max_size=max_size)
        self.extract = Extractor(resize=face_size, margin=margin)
        self.variance = (0.1, 0.2)
        self.set_params()

    def set_params(self,
                   batch_height: int = None,
                   batch_width: int = None,
                   transformed_height: int = None,
                   transformed_width: int = None,
    ):
        if batch_height is None or batch_width is None:
            self.original_width = None
            self.original_height = None
            self.scale_landmarks = None
            self.scale_bboxes = None
            self.resize_coeff = None
            self.batch_prior_box = None
        elif (self.original_height, self.original_width) != (batch_height, batch_width):
            self.original_height, self.original_width = batch_height, batch_width
            transformed_size = (transformed_width, transformed_height)
            self.scale_landmarks = torch.tensor(transformed_size, device=self.device)\
                                        .tile((5,)).reshape(5, 2)
            self.scale_bboxes = torch.tensor(transformed_size, device=self.device).tile((2,))
            self.resize_coeff = self.original_height / transformed_height
            self.batch_prior_box = priorbox(
                min_sizes=[[16, 32], [64, 128], [256, 512]],
                steps=[8, 16, 32],
                clip=False,
                image_size=(transformed_height, transformed_width),
                device=self.device,
            )

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        self.model.load_state_dict(state_dict)

    def eval(self) -> None:  # noqa: A003
        if self.device == "cpu":
            self.model = torch.jit.optimize_for_inference(torch.jit.script(self.model.eval()))
        else:
            # torch.jit.optimize_for_inference fails on GPU
            # https://discuss.pytorch.org/t/using-optimize-for-inference-on-torchscript-model-causes-error/196384/1
            self.model = torch.jit.script(self.model.eval())


    @torch.inference_mode
    @torch.jit.optimized_execution(True)
    def predict_jsons(
            self,
            image: torch.Tensor,
            confidence_threshold: float = 0.7,
            nms_threshold: float = 0.4,
    ) -> Tuple[torch.Tensor]:
        # test against 2048 max_size, 180 batch_size, achive 0.10s/f, vRAM usage 14.7G
        # torch cuda time measure
        # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        # how does torch cuda behavior
        # https://discuss.pytorch.org/t/escaping-if-statement-synchronization/130263/5
        if image.dim() != 4:
            raise ValueError(f"image tensor {image.shape} is not in BxCxHxW dimension")

        # import pdb; pdb.set_trace()
        with torch.autocast(device_type=str(self.device), enabled=(self.device=="cuda")):
            transformed_image = self.transform(image=image)

            # Due to CUDA memory limit, can not infer in batch
            locs, confs, lands = [], [], []
            for torched_image in transformed_image:
                # shapes loc 1xPx4, confs 1xPx2, lands 1xPx10
                loc, conf, land = self.model(torched_image.unsqueeze(0))
                locs.append(loc)
                confs.append(conf)
                lands.append(land)
            self.set_params(*image.shape[-2:], *transformed_image.shape[-2:])

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

            # batches is used as index so good on cpu
            # boxes is used to crop and tag targate so good on cpu
            return valid_batches.cpu(), valid_boxes, valid_landmarks, valid_scores


    @torch.inference_mode
    def predict(
            self,
            image: torch.Tensor,
            confidence_threshold: float = 0.7,
            nms_threshold: float = 0.4,
            
    ) -> Tuple[torch.Tensor]:
        """This is a wrapper of predict_jsons and extract
        Params: image - torch.Tensor: a batch of images in shape BxCxHxW
        """
        batches, boxes, landmarks, scores = self.predict_jsons(image)
        faces, bboxes = self.extract(images=image,
                                     batch_ids=batches,
                                     bboxes=boxes,
                                     landmarks=landmarks)
        return faces, batches, boxes, landmarks, scores

