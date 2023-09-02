from functools import partial


from timm import create_model
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveAvgPool2d


encoder_params = {
    "convnextv2_huge_384": {
        "features": 2816,
        "init_op": partial(create_model,'convnextv2_huge.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=0)
    },
    "convnextv2_large_384": {
        "features": 1536,
        "init_op": partial(create_model,'convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=0)
    },
    "convnextv2_base_384": {
        "features": 1024,
        "init_op": partial(create_model,'convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=0)
    },
    "convnextv2_large_224": {
        "features": 1536,
        "init_op": partial(create_model,'convnextv2_large.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0)
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
