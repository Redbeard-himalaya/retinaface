import torch
import torchvision.transforms as T

class Transformer:
    """This transformer replace albumentations equivalent:
    import albumentations as A
    transform = A.Compose([A.LongestMaxSize(max_size=2048, p=1), A.Normalize(p=1)])
    transformed_image = transform(image=image)["image"]
    """

    def __init__(self, max_size: int = 960, device: str = "cpu"):
        # refer: https://albumentations.ai/docs/api_reference/full_reference/?h=normali#albumentations.augmentations.transforms.Normalize
        max_pixel_value = 255.0
        mean = torch.tensor((0.485, 0.456, 0.406)) # in RGB
        std = torch.tensor((0.229, 0.224, 0.225))  # in RGB
        self.mean = (mean * max_pixel_value).unsqueeze(-1).unsqueeze(-1).to(device)
        self.rstd = (1 / (std * max_pixel_value)).unsqueeze(-1).unsqueeze(-1).to(device)
        self.rectangle_resizer = T.Resize(size=max_size-1, max_size=max_size)
        self.square_resizer = T.Resize(size=(max_size, max_size))

    def __call__(self, image: torch.tensor) -> torch.tensor:
        assert len(image.shape) == 3 or len(image.shape) == 4, \
            f"image ({image.shape}) is not in shape CxHxW or BxCxHxW"
        # broadcasting operates
        # refer https://stackoverflow.com/questions/51371070/how-does-pytorch-broadcasting-work
        height, width = image.shape[-2:]
        if height == width:
            return (self.square_resizer(image) - self.mean) * self.rstd
        else:
            return (self.rectangle_resizer(image) - self.mean) * self.rstd


def clip_boxes(boxes: torch.tensor,
               image_width: int,
               image_height: int,
               resize_coeff: float,
) -> torch.tensor:
    """This function replace below code by torch.tensor batch operation:
    boxes_np = boxes.cpu().numpy()
    boxes_np *= resize_coeff
    clipped_boxes = torch.empty(boxes.shape, dtype=torch.int)
    for box_id, bbox in enumerate(boxes_np):
        x_min, y_min, x_max, y_max = bbox
        x_min = np.clip(x_min, 0, image_width - 1)
        x_max = np.clip(x_max, x_min + 1, image_width)
        y_min = np.clip(y_min, 0, image_height - 1)
        y_max = np.clip(y_max, y_min + 1, image_height)
        clipped_boxes[box_id] = torch.tensor([x_min, y_min, x_max, y_max])
    return clipped_boxes

    perf test:
from time import perf_counter
boxes = torch.tensor((4, 5, 60, 80), dtype=torch.float32)[None, :].expand(40, -1)
image_width, image_height, resize_coeff = 20, 30, 0.5
ts = perf_counter()
boxes_np = boxes.cpu().numpy()
boxes_np *= resize_coeff
clipped_boxes1 = torch.empty(boxes.shape, dtype=torch.float32)
for box_id, bbox in enumerate(boxes_np):
    x_min, y_min, x_max, y_max = bbox
    x_min = np.clip(x_min, 0, image_width - 1)
    x_max = np.clip(x_max, x_min + 1, image_width)
    y_min = np.clip(y_min, 0, image_height - 1)
    y_max = np.clip(y_max, y_min + 1, image_height)
    clipped_boxes1[box_id] = torch.tensor([x_min, y_min, x_max, y_max])
print(f"cost {perf_counter() - ts}")

from time import perf_counter
boxes = torch.tensor((4, 5, 60, 80), dtype=torch.float32)[None, :].expand(40, -1)
ts = perf_counter()
clipped_boxes2 = clip_boxes(boxes, image_width=20, image_height=30, resize_coeff=0.5)
print(f"cost {perf_counter() - ts}")

(clipped_boxes1 == clipped_boxes2).all()
    """
    boxes *= resize_coeff
    boxes_num = boxes.shape[0]
    x_min_clip_down = torch.zeros(boxes_num, dtype=torch.int)
    x_min_clip_up   = torch.ones(boxes_num, dtype=torch.int) * (image_width - 1)
    x_max_clip_down = boxes[:,0] + 1
    x_max_clip_up   = torch.ones(boxes_num, dtype=torch.int) * image_width

    y_min_clip_down = torch.zeros(boxes_num, dtype=torch.int)
    y_min_clip_up   = torch.ones(boxes_num, dtype=torch.int) * (image_height - 1)
    y_max_clip_down = boxes[:,1] + 1
    y_max_clip_up   = torch.ones(boxes_num, dtype=torch.int) * image_height

    return boxes.clip(min=torch.stack([x_min_clip_down, y_min_clip_down,
                                       x_max_clip_down, y_max_clip_down], dim=1),
                      max=torch.stack([x_min_clip_up, y_min_clip_up,
                                       x_max_clip_up, y_max_clip_up], dim=1),
    )
