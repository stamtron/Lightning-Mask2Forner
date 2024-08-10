import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import yaml

# from clearml import Task
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import Mask2FormerImageProcessor

from datasets.supervisely_panoptic import SuperviselyPanopticDatasetNew
from models.mask2former_panoptic import LitMask2FormerPanoptic
from utils.helper import collate_fn, get_transforms


def main(param_config_file="params/panoptic_params.yaml"):
    # PARAMS and CLEARML
    # Create an ArgumentParser object
    parser = ArgumentParser()

    # Add the path argument to specify the configuration file
    parser.add_argument("--param_file", help="Path to the configuration file", default="params/params_trayfinder.yaml", type=str)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the YAML configuration file
    if param_config_file is None:
        param_file = args.param_file
    else:
        param_file = param_config_file
    # Read param file
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)

    pl.seed_everything(params["project"]["seed"])

    # Iterate over the configuration values and add them to the argument parser
    for section, section_config in params.items():
        for key, value in section_config.items():
            arg_name = f"--{section}_{key}"
            parser.add_argument(arg_name, default=value)

    # Parse the command-line arguments again to override the defaults
    args = parser.parse_args()
    # task = Task.init(project_name=params["project"]["project_name"], task_name=params["project"]["experiment_name"])

    # DATASETS ####
    if params["model"]["mode"] == "panoptic":
        ignore_index = 255

    processor = Mask2FormerImageProcessor(max_size=params["data"]["img_size"], ignore_index=ignore_index, do_resize=False, do_rescale=False, do_normalize=False)
    train_transform = get_transforms(params, True)

    # Get data paths
    empty_trays_path = params["data"]["empty_trays_path"]
    non_syngenta_data_path = params["data"]["non_syngenta_data_path"]
    synthetic_path = params["data"]["synthetic_path"]

    train_root = params["data"]["train_root"]
    entries = os.listdir(train_root)
    # Filter out subdirectories
    train_dirs = [os.path.join(train_root, entry) for entry in entries if os.path.isdir(os.path.join(train_root, entry))]

    if synthetic_path is not None:
        synth_entries = os.listdir(synthetic_path)
        synth_train_dirs = [os.path.join(synthetic_path, entry) for entry in synth_entries if os.path.isdir(os.path.join(synthetic_path, entry))]
        train_dirs = train_dirs + synth_train_dirs

    if empty_trays_path is not None:
        empty_trays_entries = os.listdir(empty_trays_path)
        empty_trays_dirs = [os.path.join(empty_trays_path, entry) for entry in empty_trays_entries if os.path.isdir(os.path.join(empty_trays_path, entry))]
        train_dirs = train_dirs + empty_trays_dirs

    if non_syngenta_data_path is not None:
        non_syngenta_entries = os.listdir(non_syngenta_data_path)
        non_syngenta_dirs = [os.path.join(non_syngenta_data_path, entry) for entry in non_syngenta_entries if os.path.isdir(os.path.join(non_syngenta_data_path, entry))]
        train_dirs = train_dirs + non_syngenta_dirs

    train_dataset = SuperviselyPanopticDatasetNew(paths=train_dirs, processor=processor, transform=train_transform)
    valid_transform = get_transforms(params, False)

    valid_root = params["data"]["valid_root"]
    entries = os.listdir(valid_root)
    # Filter out subdirectories
    valid_dirs = [os.path.join(valid_root, entry) for entry in entries if os.path.isdir(os.path.join(valid_root, entry))]
    val_dataset = SuperviselyPanopticDatasetNew(paths=valid_dirs, processor=processor, transform=valid_transform)

    # DATALOADERS ####
    train_dataloader = DataLoader(
        train_dataset, batch_size=params["training"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=params["data"]["num_workers"]
    )

    val_dataloader = DataLoader(val_dataset, batch_size=params["training"]["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=params["data"]["num_workers"])

    # CALLBACKS and TRAINER ####

    save_path = params["save"]["save_model_path"]
    if params["model"]["ckpt_path"] is None:
        ckpt = "COCO"
    else:
        ckpt = "Seedling"

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=save_path,
        # filename = save_path + save_path.split('/')[-2] + '_' + ckpt +'-{epoch:02d}-{val_loss:.2f}',
        filename=save_path + save_path.split("/")[-2] + "_" + ckpt + "-epoch{epoch:02d}-val_loss{val/loss:.2f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
        mode="min",
    )
    logs_directory = params["model"]["mode"] + "_logs/"
    tb_logger = TensorBoardLogger(logs_directory, name=save_path.split("/")[-2])
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        devices=1,
        max_steps=params["training"]["max_steps"],
        log_every_n_steps=params["training"]["log_every_n_steps"],
        check_val_every_n_epoch=params["training"]["check_val_every_n_epoch"],
        default_root_dir=logs_directory,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor, EarlyStopping(monitor="val/loss", mode="min", patience=params["training"]["patience"])],
        gradient_clip_val=params["training"]["gradient_clip_val"],
        accumulate_grad_batches=params["training"]["accumulate_grad_batches"],
    )

    # MODEL ####
    module = LitMask2FormerPanoptic(params, processor)

    # TRAINING ####
    if os.path.exists(params["model"]["ckpt_path"]):
        trainer.fit(
            module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=params["model"]["ckpt_path"],
        )
    else:
        print("No checkpoint found")
        trainer.fit(
            module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    # task.close()


if __name__ == "__main__":
    main()
