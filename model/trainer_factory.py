import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR, ExponentialLR, PolynomialLR, LambdaLR, StepLR
from torch.optim import Adam, AdamW, SGD, RMSprop

# from sklearn.metrics import log_loss
from .zoo.convnext import *
from .zoo.convnext_2 import *
from .zoo.convnext import DeepFakeClassifier as ConvNeXtClassifier
from .zoo.convnext_2 import DeepFakeClassifier as ConvNeXtv2Classifier
from .zoo.efficentnet import DeepFakeClassifier as EfficentNetClassifier
from .optim import *
from .loss_fn.loss_functions import BinaryCrossentropy, WeightedLosses
import numpy as np
from torchmetrics.classification import BinaryPrecisionRecallCurve, BinaryAccuracy, BinaryAUROC, BinaryROC
import gc

class TrainerFactory:
    def __init__(self, params):
        self.params = params

    def get_size(self):
        return self.params['size']

    def get_loss_fn(self):
        config = self.params['loss_fn']
        if config == 'BinaryCrossentropy':
            return BinaryCrossentropy()

        raise ValueError('Loss function not registered')

    def get_optim(self, model):
        params = model.parameters()
        optimizer_config = self.params['optimizer']
        optim_type = optimizer_config['type']

        lr = optimizer_config["learning_rate"]
        momentum = optimizer_config["momentum"]
        weight_decay = optimizer_config["weight_decay"]
        nesterov = optimizer_config["nesterov"]

        if optim_type == 'Adam':
            return Adam(params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'AdamW':
            return AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'SGD':
            return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        elif optim_type == 'RMSprop':
            return RMSprop(params, lr=lr, weight_decay=weight_decay)



        raise ValueError('Optimizer not registered! Registered Optimizers: [Adam, AdamW, SGD, RMSprop]')

    def get_base_model(self):
        config = self.params['network']
        encoder = self.params['encoder']
        dropout = self.params['dropout']
        if config == 'efficientnet':
            return EfficentNetClassifier(encoder=encoder, dropout_rate=dropout)
        elif config == 'convnext':
            return ConvNeXtClassifier(encoder=encoder, dropout_rate=dropout)
        elif config == 'convnext_v2':
            return ConvNeXtv2Classifier(encoder=encoder, dropout_rate=dropout)

        raise ValueError('Model not registered')

    def get_scheduler(self, optim):
        config = self.params['optimizer']['schedule']
        schedule_type = config['type']
        optim_params = config['params']

        if schedule_type == 'step':
            return StepLR(optim, **optim_params)
        elif schedule_type == 'clr':
            return CyclicLR(optim, **optim_params)
        elif schedule_type == 'multistep':
            return MultiStepLR(optim, **optim_params)
        elif schedule_type == 'exponential':
            return ExponentialLR(optim, **optim_params)
        elif schedule_type == 'poly':
            return PolynomialLR(optim, **optim_params)
        elif schedule_type == 'constant':
            return LambdaLR(optim, lambda epoch: 1.0)
        elif schedule_type == 'linear':
            return LambdaLR(optim, lambda it: it * optim_params['alpha'] + optim_params['beta'])

        raise ValueError('Scheduler not registered')

    def build_model(self):
        model = self.get_base_model()
        loss_fn = self.get_loss_fn()
        optim = self.get_optim(model)
        scheduler = self.get_scheduler(optim)
        size = self.get_size()

        return GenericModel(model=model, optim=optim, loss_fn=loss_fn, scheduler=scheduler, size=size)


class GenericModel(pl.LightningModule):
    def __init__(self, model, optim, loss_fn, scheduler, size, eps=1e-10):
        super().__init__()
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.size = size
        self.eps = eps
        self.yp_val = []
        self.y_val = []
        #self.acc = BinaryAccuracy()

    def get_base_model(self):
        return self.model

    def get_loss_fn(self):
        return self.loss_fn

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return {
            'optimizer': self.optim,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'val_loss',
            },
        }

    def log_all(self, real_loss, fake_loss, loss, modus="train"):
        self.log_dict({
            f'{modus}_real_loss': real_loss,
            f'{modus}_fake_loss': fake_loss,
            f'{modus}_loss': loss
        }, sync_dist=True, prog_bar=True, on_step=True,logger=True)



    def training_step(self, train_batch, batch_idx):
        x, y = train_batch['x'], train_batch['y']

        size = x.size(0) * x.size(1)

        x = torch.reshape(x, (size, 3, self.size, self.size))
        y = torch.reshape(y, (size, 1))

        yp = torch.add(self.model(x), self.eps)

        loss = self.loss_fn(yp, y)
        #print(yp, y,  loss)
        self.log("train_loss",loss, sync_dist=True, prog_bar=True, on_step=True,logger=True)

        return loss

    def on_train_epoch_end(self):
        gc.collect()

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['x'], val_batch['y']

        size = x.size(0) * x.size(1)

        x = torch.reshape(x, (size, 3, self.size, self.size))
        y = torch.reshape(y, (size, 1))

        yp = self.model(x)

        loss = self.loss_fn(yp, y).detach().cpu()
        #self.yp_val.extend(yp.detach().cpu().tolist())
        #self.y_val.extend(y.detach().cpu().tolist())

        self.log("val_loss",loss, sync_dist=True, prog_bar=True, on_step=True,logger=True)

    def on_validation_epoch_end(self):
        #yp_val = np.array(self.yp_val)
        #y_val = np.array(self.y_val)
        #loss = self.loss_fn(yp_val, y_val)
        #self.log("total_val_loss", loss, sync_dist=True, prog_bar=True, logger=True)
        #self.yp_val, self.y_val = [], []
        gc.collect()


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch['x'], test_batch['y']

        size = x.size(0) * x.size(1)

        x = torch.reshape(x, (size, 3, self.size, self.size))
        y = torch.reshape(y, (size, 1))

        yp = self.model(x)

        loss = self.loss_fn(yp, y).detach().cpu()
        self.log("test_loss",loss, sync_dist=True, prog_bar=True, on_step=True,logger=True)
