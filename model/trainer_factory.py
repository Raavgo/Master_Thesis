import json

import random
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR, ExponentialLR, PolynomialLR, LambdaLR, StepLR
from torch.optim import Adam, AdamW, SGD, RMSprop
import matplotlib.pyplot as plt
# from sklearn.metrics import log_loss
from .zoo.convnext import *
from .zoo.convnext_2 import *
from .zoo.convnext import DeepFakeClassifier as ConvNeXtClassifier
from .zoo.convnext_2 import DeepFakeClassifier as ConvNeXtv2Classifier
from .zoo.efficentnet import DeepFakeClassifier as EfficentNetClassifier
from .optim import *
from .loss_fn.loss_functions import BinaryCrossentropy, WeightedLosses
import numpy as np

import gc
import torchmetrics.classification as torchmetrics
class TrainerFactory:
    def __init__(self, params):
        self.params = params

    def get_path(self):
        return self.params['path']

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
        path = self.get_path()

        return GenericModel(model=model, optim=optim, loss_fn=loss_fn, scheduler=scheduler, size=size, path=path)


class GenericModel(pl.LightningModule):
    def __init__(self, model, optim, loss_fn, scheduler, size, eps=1e-10, path=""):
        super().__init__()
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.size = size
        self.eps = eps
        self.yp = []
        self.y = []
        self.path = path

        # Metrics
        self.accuracy = torchmetrics.BinaryAccuracy(compute_on_step=False)
        self.precision_recall_curve = torchmetrics.BinaryPrecisionRecallCurve(compute_on_step=False)
        self.auroc = torchmetrics.BinaryAUROC(compute_on_step=False)
        self.roc = torchmetrics.BinaryROC(compute_on_step=False)
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

        self.log("val_loss",loss, sync_dist=True, prog_bar=True, on_step=True,logger=True)
        return loss

    def on_validation_epoch_end(self):
        gc.collect()


    def test_step(self, test_batch, batch_idx):
        self.model.eval()
        x, y = test_batch['x'], test_batch['y']

        size = x.size(0) * x.size(1)

        x = torch.reshape(x, (size, 3, self.size, self.size))
        y = torch.reshape(y, (size, 1))

        yp = self.model(x)
        loss = self.loss_fn(yp, y)
        yp_prob = torch.sigmoid(yp)

        # Update metrics
        self.accuracy(yp_prob, y.long())
        torch.use_deterministic_algorithms(False)
        self.precision_recall_curve(yp_prob, y.long())
        self.auroc(yp_prob, y.long())
        self.roc(yp_prob, y.long())
        torch.use_deterministic_algorithms(True)

        self.log("test_loss",loss, sync_dist=True, prog_bar=True, on_step=True,logger=True)
        return loss

    def on_test_end(self):
        torch.use_deterministic_algorithms(False)
        roc_auc = self.auroc.compute()
        acc = self.accuracy.compute()

        precision, recall, _ = self.precision_recall_curve.compute()
        fpr, tpr, _ = self.roc.compute()
        random_value = random.randint(10000, 99999)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr.cpu(), tpr.cpu(), 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(f'{self.path}/roc{random_value}.png')
        torch.use_deterministic_algorithms(True)

        results = {
            "acc": acc,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "fpr":fpr,
            "tpr":tpr
        }

        torch.save(results, f'{self.path}/metrics{random_value}')

        return
