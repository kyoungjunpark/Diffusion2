from model.minknet.mink_unet import MinkUNet18A
from model.minknet.completionnet import CompletionNet
from model.minknet.mink_unet import MinkUNet34B
from model.minknet.detection import SparseGenerativeFeatureUpsampleNetwork


def get_model(model_name, classes, in_channels=None, multi_scale=False, refractive=False):
    '''Get the 3D model.'''
    if model_name == "mink_unet":
        model = MinkUNet18A(in_channels=3, out_channels=classes, D=3, multi_scale=multi_scale, refractive=refractive)
        # model = MinkUNet34B(in_channels=3, out_channels=classes, D=3)
    elif model_name == "completion":
        model = CompletionNet(channels=classes)
    elif model_name == "detection":
        model = SparseGenerativeFeatureUpsampleNetwork(in_channels=classes, D=3)
    return model
