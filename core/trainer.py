'''
author: yingmuzhi
time: 20230705

intro: Core Components. Trainer.
'''
import torch
import core

class Trainer(core.hyper_parameters.HyperParameters):
    """
    intro:
        Trainer.
    """
    def __init__(self,
                 max_epochs,
                 num_gpus=0,
                 gradient_clip_val=0,
                 ) -> None:
        self.save_hyperparameters()
        assert num_gpus != 0, "ERROR::No GPU support yet"
        self.gpus = [core.gpu(i) for i in range(min(num_gpus, core.num_gpus()))]
    
    def prepare_data(self, data):
        """
        intro:
            get data ready.
        
        args:
            :param core.data_module.DataModule data: how to sample your data.
        """
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader) # means how many batches per epoch. -- batch_size means how many pics per batch.
        self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0)
    
    def prepare_model(self, model):
        """
        intro:
            get model ready.
        
        args:
            :param core.module.Module model: how to plot your model while training and validation.
        """
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model
    
    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
    
    def prepare_batch(self, batch):
        if self.gpus:
            batch = [core.to(a, self.gpus[0]) for a in batch]
        return batch
    
    def clip_gradients(self, grad_clip_val, model):
        """Defined in :numref:`sec_rnn-scratch`"""
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm
    
    def fit_epoch(self):
        """
        intro:
            training per epoch
        """
        self.model.train()  # since Module is from nn.Module, it will relate with self.model.net
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1