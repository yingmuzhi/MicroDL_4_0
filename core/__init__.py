# region ABS path 
import sys, os
module_path = os.path.dirname(__file__)
project_path = os.path.dirname(os.path.dirname(module_path))
sys.path.extend([module_path, project_path])
# endregion

# region version
VERSION = 1.0
# endregion

# region importlib
import torch, torch.nn as nn, matplotlib_inline.backend_inline as backend_inline, matplotlib.pyplot as plt # plt is held as the global variable
import \
    core.hyper_parameters as hyper_parameters, \
    core.module as module, core.progress_board as progress_board, \
    core.data_module as data_module, core.trainer as trainer
# endregion

# region progress_board.py
def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')
# endregion

# region module.py
def init_cnn(module):
    """
    intro:
        Initialize weights for CNNs.
        default is `xavier_uniform`
    """
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
# endregion

# region data_module.py
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    intro:
        Plot a list of images.
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = torch.numpy(img)
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
# endregion

# region trainer.py
def cpu():
    """Get the CPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device('cpu')

def gpu(i=0):
    """Get a GPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device(f'cuda:{i}')

def num_gpus():
    """Get the number of available GPUs.

    Defined in :numref:`sec_use_gpu`"""
    return torch.cuda.device_count()

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    return [gpu(i) for i in range(num_gpus())]
# endregion

# region global variables
def add_to_class(Class):
    """Register functions as methods in created class.

    Defined in :numref:`sec_oo-design`"""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
float32 = torch.float32
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
randn = torch.randn
# endregion