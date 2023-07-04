'''
author: yingmuzhi
time: 20230704

intro: Core Components. Module elements, such as Unet and so on.
'''
import torch
import core

class Module(torch.nn.Module, core.hyper_parameters.HyperParameters):
    """
    intro:
        abstract class Module
    
    args:
        :param int plot_train_per_epoch: plot the pics.
        :param int plot_valid_per_epoch: plot.
    """
    
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = core.progress_board.ProgressBoard()