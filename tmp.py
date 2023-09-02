from tqdm import tqdm

from model.zoo.convnext_2 import DeepFakeClassifier
from dataset.dataset import DeepFakeClassificationDataset
from torch.utils.data import DataLoader
import torch
from dataset.augmentation.augmentation import create_train_transforms, create_val_transforms


"""encoder_params = {
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
}"""
s = 224
train_data =  DeepFakeClassificationDataset("/home/ai21m034/master_project/data/data/validation", samples=4,  transform=create_train_transforms(s))
train_dl = DataLoader(train_data, batch_size=1, num_workers=2, shuffle=True)

#m = DeepFakeClassifier("convnextv2_large_224")
#m = m.eval()

for (i, batch) in tqdm(enumerate(train_dl)):
    x, y = batch['x'], batch['y']
    #size = x.size(0) * x.size(1)
    """    x = torch.reshape(x, (size, 3, s, s))
    y = torch.reshape(y, (size, 1))
    yp = m(x)

    print(yp, y)"""

