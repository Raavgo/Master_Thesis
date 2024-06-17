import argparse
import json
import os.path

from dataset.dataset import DeepFakeClassificationDataset
from dataset.augmentation.augmentation import create_train_transforms, create_val_transforms
from model.trainer_factory import TrainerFactory
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Train Model")
   parser.add_argument("--trainer_config", help="JSON config file for trainer", required=True)
   parser.add_argument("--model_config", help="JSON config file for the model", required=True)
   parser.add_argument("--ckpt", help="Path to Checkpoint", required=True)
   parser.add_argument("--nodes", help="(Optional) to Override train config with slurm config", required=False)
   parser.add_argument("--tasks", help="(Optional) to Override train config with slurm config", required=False)
   args = parser.parse_args()

   with open(args.trainer_config) as config_file:
    trainer_config = json.load(config_file)

    with open(args.model_config) as config_file:
        model_config = json.load(config_file)

    model_name = model_config["network"]

   test_data = DeepFakeClassificationDataset(
       trainer_config["validation_path"],
       samples=model_config["sample_size"],
       transform=create_val_transforms(model_config["size"]),
       mode='test'
   )

   test_dl = DataLoader(
        test_data,
        batch_size=model_config["batch_size"],
        num_workers=trainer_config["num_workers"],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True
   )

   trainer_config["model_checkpoint"]["filename"] += model_name
   callbacks = []
   # callbacks = [
   #     ModelCheckpoint(**trainer_config["model_checkpoint"]),
   #     EarlyStopping(**trainer_config["early_stopping"]),
   # ]

   trainer_config["logger"]["name"] = model_name
   logger = CSVLogger(**trainer_config["logger"])

   trainer_config["trainer"]["callbacks"] = callbacks
   trainer_config["trainer"]["logger"] = logger
   if args.tasks:
       trainer_config["trainer"]["devices"] = args.tasks
   if args.nodes:
       trainer_config["trainer"]['num_nodes'] = args.nodes

   trainer = pl.Trainer(accelerator = "gpu")

   lightning_model = TrainerFactory(model_config).build_model()
   checkpoint_path = args.ckpt

   print("Model checkpoint found:", os.path.exists(checkpoint_path))
   #if os.path.exists(checkpoint_path):
   lightning_model = load_state_dict_from_zero_checkpoint(lightning_model, checkpoint_path)
   lightning_model.eval()
   trainer.test(
       model=lightning_model,
       dataloaders = test_dl
   )