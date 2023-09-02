from functools import partial

import numpy as np
import torch
from timm import create_model
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d

encoder_params = {
    "convnext_large_clip_320": {
        "features": 1536,
        "init_op": partial(create_model,'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320', pretrained=True, num_classes=0)
    },
    "convnext_xxlarge_clip_256": {
        "features": 3072,
        "init_op": partial(create_model,'convnext_xxlarge.clip_laion2b_soup_ft_in1k', pretrained=True, num_classes=0)
    },
    "convnext_large_clip_384": {
        "features": 1536,
        "init_op": partial(create_model,'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384', pretrained=True, num_classes=0)
    },
    "convnext_xlarge_fb_384": {
        "features": 2048,
        "init_op": partial(create_model,'convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True, num_classes=0)
    }
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
