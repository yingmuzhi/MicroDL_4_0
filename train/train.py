import torch, torch.nn as nn, sys; sys.path.append("/home/yingmuzhi/microDL_4_0")
import core


# region AlexNet Model
class AlexNet(core.module.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(core.init_cnn)

AlexNet().layer_summary((1, 1, 224, 224))
# endregion

# region Train
model = AlexNet(lr=0.01)
data = core.data_module.FashionMNIST(batch_size=128, resize=(224, 224))
trainer = core.trainer.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
# endregion
