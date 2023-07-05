'''
author: yingmuzhi
time: 20230704

intro: Core Components. Module elements, such as Unet and so on.

    - module include plot on training and validation.
    - forward.
    - layer_summary.
    - *parameters init
'''
import torch, torch.nn.functional as F, torch.nn as nn
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
        super().__init__()  # torch.nn.Module
        self.save_hyperparameters()
        self.board = core.progress_board.ProgressBoard()
    
    def loss(self, y_hat, y):
        raise NotImplementedError
    
    def forward(self, X):
        assert hasattr(self, "net"), "ERROR::Neural network is not defined"
        return self.net(X)

    def plot(self, key, value, train):
        """
        intro:
            plot animation.
        """
        assert hasattr(self, "trainer"), "ERROR::Trainer is not inited"
        self.board.xlabel = "epoch"
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, core.numpy(core.to(value, core.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        """
        intro:
            calculate loss.
        """
        l = self.loss(self(*batch[:-1]), batch[-1]) # same as model(X) -> loss(model(X), y)
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        """Defined in :numref:`sec_classification`"""
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):
        """Defined in :numref:`sec_lazy_init`"""
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)


class Classifier(Module):
    """The base class of classification models.

    Defined in :numref:`sec_classification`"""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions.
    
        Defined in :numref:`sec_classification`"""
        Y_hat = core.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = core.astype(core.argmax(Y_hat, axis=1), Y.dtype)
        compare = core.astype(preds == core.reshape(Y, -1), core.float32)
        return core.reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        Y_hat = core.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = core.reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = core.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)