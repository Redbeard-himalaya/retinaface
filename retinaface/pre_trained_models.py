from pathlib import Path

from cached_path import cached_path
from collections import namedtuple
import torch
from torch.utils import model_zoo

from retinaface.predict_single import Model

model = namedtuple("model", ["url", "model"])

models = {
    "resnet50_2020-07-20": model(
        url="https://github.com/Redbeard-himalaya/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20.pth",  # noqa: E501 pylint: disable=C0301
        model=Model,
    )
}


def get_model(model_name: str,
              max_size: int,
              model_dir: Path = None,
              device: str = "cpu",
              quiet: bool = False,
) -> Model:
    model = models[model_name].model(max_size=max_size, device=device)
    weight_file = cached_path(models[model_name].url, cache_dir=model_dir.resolve(), quiet=quiet)
    state_dict = torch.load(weight_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    return model
