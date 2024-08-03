from pathlib import Path

from cached_path import cached_path
from collections import namedtuple
import torch
from torch.utils import model_zoo

from retinaface.predict_single import Model as SingleModel
from retinaface.predict_batch import Model as BatchModel

model = namedtuple("model", ["url", "model"])

models = {
    "resnet50_2020-07-20": model(
        url="https://github.com/Redbeard-himalaya/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20.pth",  # noqa: E501 pylint: disable=C0301
        model={"single": SingleModel, "batch": BatchModel},
    )
}


def get_model(model_name: str,
              max_size: int,
              batch_model: bool = True,
              face_size: int = 112,
              margin: int = 0,
              model_dir: Path = None,
              device: str = None,
              quiet: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch_model:
        model = models[model_name].model["batch"](max_size=max_size,
                                                  face_size=face_size,
                                                  margin=margin,
                                                  device=device)
    else:
        model = models[model_name].model["single"](max_size=max_size,
                                                   face_size=face_size,
                                                   margin=margin,
                                                   device=device)
    if model_dir is None:
        model_dir = Path.home() / ".face_search"
    weight_file = cached_path(models[model_name].url, cache_dir=model_dir.resolve(), quiet=quiet)
    if device == "cpu":
        state_dict = torch.load(
            weight_file,
            map_location=lambda storage, loc: storage,
        )
    else:
        state_dict = torch.load(
            weight_file,
            map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()),
        )
    model.load_state_dict(state_dict)
    return model
