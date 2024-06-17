from functools import partial

import numpy as np
import torch
from timm.models.efficientnet import tf_efficientnet_b6, tf_efficientnet_b7, tf_efficientnet_b4
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch import sigmoid

encoder_params = {
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b6_ns": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7, pretrained=True, drop_path_rate=0.2)
    },
    "tf_efficientnet_b7_ns_04d": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7, pretrained=True, drop_path_rate=0.4)
    },
    "tf_efficientnet_b6_ns_04d": {
        "features": 2304,
        "init_op": partial(tf_efficientnet_b6, pretrained=True, drop_path_rate=0.4)
    },
}

class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.0) -> None:
        super().__init__()
        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params[encoder]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
