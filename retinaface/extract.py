from typing import Tuple

import torch
from torch.nn import functional as F
from torchvision.transforms.v2 import ConvertBoundingBoxFormat
from torchvision import tv_tensors
from torchvision.transforms.functional import resized_crop

class Extractor:
    def __init__(self, width: int, height: int, resize: Tuple[int]):
        self.bbox_convert = ConvertBoundingBoxFormat(format="XYWH")
        self.canvas_size = (height, width)
        self.resize = resize

    def __call__(self,
                 images: torch.Tensor,
                 bboxes: torch.Tensor,
                 landmarks: torch.Tensor,
                 batch_ids: torch.Tensor,
    ):
        """Extract faces from images located in bounding box bboxes, then resize and align faces
        Parameters:
        images - tensor in shape Bx3xHxW
        bboxes - tensor in Nx4 in format XYXY
        landmarks - tensor in Nx5x2
        batch_ids - tensor in N, it index bboxes and landmarks to images
        Usage:
        from retinaface.extract import Extractor
        e = Extractor(width=imgs.shape[-1], height=imgs.shape[-2], resize=(160,160))
        c = e(images=imgs, bboxes=model.valid_boxes, batch_ids=model.valid_batches)
        for i in range(len(imgs)):
        for j, crop in enumerate(c[torch.where(model.valid_batches == i)]):
            to_pil_image(crop).save(f"retinaface/test/marked/9_{i}-{j}.jpg")
        """
        if images.dim() != 4:
            raise ValueError(f"images {images.shape} is not in NxCxHxW shape")
        if bboxes.shape[0] != batch_ids.shape[0]:
            raise ValueError(f"bboxes {bboxes.shape} num does not equal to batch_ids {batch_ids.shape} num")
        bboxes = self.bbox_convert(
            tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=self.resize,
                dtype=torch.int,
            )
        )
        # resized_crop is faster against float32 images
        images = images.to(torch.float32)
        # crops in shape: BATCH_IDS.NUM x RESIZE x RESIZE, dtype: float32
        crops = images.new_empty((bboxes.shape[0], 3) + self.resize)
        empty_crop = images.new_empty((0, 3) + self.resize)
        # import pdb; pdb.set_trace()
        for i, image in enumerate(images):
            _crops = [empty_crop]
            idx = torch.where(batch_ids == i)
            for x, y, w, h in bboxes[idx]:
                # croped image is in dtype.uint8
                _crops.append(
                    resized_crop(img=image,
                                 left=x, top=y, width=w, height=h,
                                 size=self.resize).unsqueeze(0)
                )
            # torch.cat autocast uint8 to float32 because empty_crop.dtype==float32
            crops[idx] = torch.cat(_crops, dim=0)
        # alignment require crops.dtype in torch.float32, otherwise fails at affine_grid
        return self.alignment(crops, landmarks)


    def alignment(self,
                  img: torch.Tensor,
                  landmark: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Alignma given face with respect to the left and right eye coordinates.
        Left eye is the eye appearing on the left (right eye of the person). Left top point is (0, 0)
        Args:
            img (torch.Tensor): an image 3D or a batch of images 4D
            landmark (torch.Tensor): a landmark of a face in 2D or a batch of landmarks of faces in 3D
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"param img ({type(img)}) is not torch.Tensor")
        if not isinstance(landmark, torch.Tensor):
            raise TypeError(f"param landmark ({type(landmark)}) is not torch.Tensor")
        img_dim = img.dim()
        lm_dim = len(landmark.shape)
        if img_dim < 3 or 4 < img_dim:
            raise TypeError(f"param img ({img_dim}D) is not 3D or 4D")
        if lm_dim < 2 or 3 < lm_dim:
            raise TypeError(f"param img ({img_dim}D) is not 3D or 4D")

        # change signle image and landmark to batch
        if img_dim == 3:
            img = img.unsqueeze(0)
        if lm_dim == 2:
            landmark = landmark.unsqueeze(0)

        # calculate angle in batch
        left_eye_x = landmark[:,0,0]  # batch in shape N
        left_eye_y = landmark[:,0,1]  # batch in shape N
        right_eye_x = landmark[:,1,0] # batch in shape N
        right_eye_y = landmark[:,1,1] # batch in shape N
        angle = torch.complex(right_eye_x - left_eye_x,
                              right_eye_y - left_eye_y).angle() * 180 / torch.pi
        # rotate in batch
        img = self.batch_rotate(img, angle)

        # -----------------------
        if img_dim == 3:
            return img[0]#, -angle[0]
        else:
            return img#, -angle


    def batch_rotate(self,
                     x: torch.Tensor,
                     degree: torch.Tensor,
    ) -> torch.Tensor:
        """
        https://discuss.pytorch.org/t/how-to-rotate-batch-of-images-by-a-batch-of-angles/187482
        Rotate batch of images [B, C, W, H] by a specific angles [B] 
        (counter clockwise)

        Parameters
        ----------
        x : torch.Tensor
        batch of images
        angle : torch.Tensor
        batch of angles

        Returns
        -------
        torch.Tensor
            batch of rotated images
        """
        if x.dim() != 4:
            raise TypeError(f"param x ({img.dim()}D) is not in 4D")
        angle = degree / 180 * torch.pi
        s = torch.sin(angle)
        c = torch.cos(angle)
        rot_mat = torch.stack((torch.stack([c, -s], dim=1),
                               torch.stack([s, c], dim=1)), dim=1)
        zeros = torch.zeros(rot_mat.size(0), 2).unsqueeze(2).to(rot_mat.device)
        aff_mat = torch.cat((rot_mat, zeros), 2)
        grid = F.affine_grid(aff_mat, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x
