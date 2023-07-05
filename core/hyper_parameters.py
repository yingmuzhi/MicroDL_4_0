'''
author: yingmuzhi
time: 20230615

intro: Core Components. To get the parameters in () embed in self. automatically.
'''
import inspect


class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """
        intro:
            Must be overloaded.
        """
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """
        intro:
            Save function arguments into class attributes.
        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)