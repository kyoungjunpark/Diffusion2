


from importlib import import_module


def get(args):
    if args.model_name.startswith("Diffusion_Heatmap"):
        model_name = "Diffusion_Heatmap_"
    else:
        model_name = args.model_name
    metric_name = model_name + 'Metric'
    module_name = 'metric.' + metric_name.lower()
    module = import_module(module_name)

    return getattr(module, metric_name)


class BaseMetric:
    def __init__(self, args):
        self.args = args

    def evaluate(self, sample, output, mode):
        pass
