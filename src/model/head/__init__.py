from importlib import import_module
from .image_diffusion import Image_Diffusion
from .image_diffusion_fpn import Image_Diffusion_FPN
from .video_diffusion import Video_Diffusion


def get(args):
    model_name = args.head_name
    module_name = 'model.' + model_name.lower()
    module = import_module(module_name)

    return getattr(module, model_name)
