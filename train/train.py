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
    parser.add_argument("--config", help="JSON config file for training", required=True)
    parser.add_argument("--mode", help="Mode for training", required=True)
    parser.add_argument("--nodes", help="(Optional) to Override train config with slurm config", required=False)
    parser.add_argument("--tasks", help="(Optional) to Override train config with slurm config", required=False)
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    project_base = config["project_base"]

    trainer_config = config["trainer_config"]
    trainer_config["train_path"] = project_base + trainer_config["train_path"]
    trainer_config["validation_path"]= project_base + trainer_config["validation_path"]
    trainer_config["test_path"]= project_base + trainer_config["test_path"]
    trainer_config["logger"]["save_dir"]= project_base + trainer_config["logger"]["save_dir"] + args.mode
    mode = args.mode
    model_config = config["model_config"]
    model_config["name"] = config["name"] + " " + mode
    model_config["path"] = project_base + config["path"]+ mode
    trainer_config["model_checkpoint"]["dirpath"] = project_base + config["path"]+ mode

    model_name = model_config["network"]

    if not os.path.exists(model_config["path"]):
        os.makedirs(model_config["path"])

    if not os.path.exists(trainer_config["logger"]["save_dir"]):
        os.makedirs(trainer_config["logger"]["save_dir"])

    f = create_val_transforms(model_config["size"])
    m = 'test'
    b = False

    if 'sampling' in mode:
        b = True

    if 'blackout' in mode:
        m = 'train'

    if 'albu_aug' in mode:
        f = create_train_transforms(model_config["size"])

    train_data = DeepFakeClassificationDataset(
        trainer_config["train_path"],
        samples=model_config["sample_size"],
        transform=f,
        mode=m,
        balance=b
    )

    validation_data = DeepFakeClassificationDataset(
        trainer_config["validation_path"],
        samples=model_config["sample_size"],
        transform=create_val_transforms(model_config["size"]),
        mode='test'
    )

    test_data = DeepFakeClassificationDataset(
        trainer_config["test_path"],
        samples=model_config["sample_size"],
        transform=create_val_transforms(model_config["size"]),
        mode='test'
    )

    train_dl = DataLoader(
        train_data,
        batch_size=model_config["batch_size"],
        num_workers=trainer_config["num_workers"],
        shuffle=True,
        pin_memory=True,

    )

    validation_dl = DataLoader(
        validation_data,
        batch_size=model_config["batch_size"],
        num_workers=trainer_config["num_workers"],
        shuffle=False,
        pin_memory=True,
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
    callbacks = [
        ModelCheckpoint(**trainer_config["model_checkpoint"]),
        EarlyStopping(**trainer_config["early_stopping"]),
    ]

    trainer_config["logger"]["name"] = model_name
    logger = CSVLogger(**trainer_config["logger"])

    trainer_config["trainer"]["callbacks"] = callbacks
    trainer_config["trainer"]["logger"] = logger
    if args.tasks:
        trainer_config["trainer"]["devices"] = args.tasks
    if args.nodes:
        trainer_config["trainer"]['num_nodes'] = args.nodes

    trainer = pl.Trainer(**trainer_config["trainer"])

    lightning_model = TrainerFactory(model_config).build_model()
    checkpoint_path = f'{trainer_config["model_checkpoint"]["dirpath"]}/{trainer_config["model_checkpoint"]["filename"]}.ckpt'
    print("Model checkpoint found:", os.path.exists(checkpoint_path))

    if os.path.exists(checkpoint_path):
        lightning_model = load_state_dict_from_zero_checkpoint(lightning_model, checkpoint_path)

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dl,
        val_dataloaders=validation_dl
    )

    trainer.test(
        model=lightning_model,
        dataloaders=test_dl
    )
