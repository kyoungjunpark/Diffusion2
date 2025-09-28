


from importlib import import_module


def get(args):
    if args.model_name.startswith("Diffusion_Heatmap"):
        model_name = "Diffusion_Heatmap_Model"
    else:
        model_name = args.model_name + 'Model'
    module_name = 'model.' + model_name.lower()
    # print('Module Name {}'.format(module_name))
    module = import_module(module_name)

    return getattr(module, model_name)
