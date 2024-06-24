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
