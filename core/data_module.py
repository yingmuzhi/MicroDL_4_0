'''
author: yingmuzhi
time: 20230705

intro: Core Components. To get the data.
'''
import torch, torchvision.transforms as transforms, torchvision
import core


class DataModule(core.hyper_parameters.HyperParameters):
    """
    intro:
        The base class of data.
    """
    def __init__(self, root='/home/yingmuzhi/microDL_4_0/src/data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)


class FashionMNIST(DataModule):
    """
    intro:
        The Fashion-MNIST dataset.
    """
    def __init__(self, batch_size=64, resize=(28, 28), download=False):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=download)     # train_dataset
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=download)    # val_dataset

    def text_labels(self, indices):
        """Return text labels.
    
        Defined in :numref:`sec_fashion_mnist`"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        """Defined in :numref:`sec_fashion_mnist`"""
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                           num_workers=self.num_workers)        # train_dataloader or val_dataloader

    def visualize(self, batch, nrows=1, ncols=8, labels=[]):
        """Defined in :numref:`sec_fashion_mnist`"""
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        core.show_images(X.squeeze(1), nrows, ncols, titles=labels)