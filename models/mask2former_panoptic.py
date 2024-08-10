import logging

import lightning.pytorch as pl
import torch
from phenobench.evaluation.auxiliary.panoptic_eval import PanopticQuality
from torch import tensor
from torch_ema import ExponentialMovingAverage
from torchmetrics.classification import MulticlassJaccardIndex  # type: ignore
from torchmetrics.detection import MeanAveragePrecision
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerModel,
)

from utils.helper import AverageMeter


class LitMask2FormerPanoptic(pl.LightningModule):
    def __init__(self, params, processor, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        self.processor = processor
        self.train_stats = {
            "loss": AverageMeter(),
        }
        self.val_stats = {
            "loss": AverageMeter(),
            "PQ": PanopticQuality(),
            "IoU": MulticlassJaccardIndex(num_classes=2, average=None),
            "mAP": MeanAveragePrecision(iou_type="segm"),
        }
        self.pred_stats = {
            "loss": AverageMeter(),
            "PQ": PanopticQuality(),
            "IoU": MulticlassJaccardIndex(num_classes=2, average=None),
            "mAP": MeanAveragePrecision(iou_type="segm"),
        }
        self.model = self.make_model()

        logging.getLogger().setLevel(logging.INFO)

    def make_model(self):
        id2label = {0: "background", 1: "seedling"}
        config = Mask2FormerConfig.from_pretrained(self.params["model"]["pretrained"])
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        config.num_queries = self.params["model"]["num_queries"]

        base_model = Mask2FormerModel.from_pretrained(self.params["model"]["pretrained"], config=config, ignore_mismatched_sizes=True)
        base_model.config.num_queries = self.params["model"]["num_queries"]
        model = Mask2FormerForUniversalSegmentation(config)
        model.model = base_model
        if self.params["model"]["ema_decay"] < 1.0:
            self.ema = ExponentialMovingAverage(model.model.parameters(), decay=self.params["model"]["ema_decay"])

        # FIX THAT ###
        # print(self.params["model"]["pth_path"] is not None)
        # if self.params["model"]["pth_path"] is not None:
        #     print("Loading pth model...")
        #     print(self.params["model"]["pth_path"])
        #     model.load_state_dict(torch.load(self.params["model"]["pth_path"]))
        return model

    def forward(self, batch):
        # print(batch['pixel_values'].shape)
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
        )
        return outputs

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.params["model"]["ema_decay"] < 1.0:
            # workaround, because self.device is not yet set in setup()
            if self.ema.shadow_params[0].device != self.device:
                self.ema.shadow_params = [p.to(self.device) for p in self.ema.shadow_params]
            self.ema.update(self.model.model.parameters())

    def training_step(self, batch, batch_idx):
        # print('training_step')
        outputs = self.forward(batch)
        self.train_stats["loss"].update(outputs.loss.item())
        self.log("train/loss", self.train_stats["loss"].avg, prog_bar=True)  # , on_step=True)
        if (batch_idx + 1) % self.params["training"]["log_every_n_steps"] == 0:
            for v in self.train_stats.values():
                v.reset()
        return outputs.loss

    def on_train_epoch_end(self):
        for v in self.train_stats.values():
            v.reset()

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)

        # Evaluation
        if self.params["training"]["do_val_metrics"]:
            img_size = batch["pixel_values"].shape[2:]  # (self.params['data']['img_size'], self.params['data']['img_size'])
            if self.params["model"]["mode"] == "instance":
                predictions = self.processor.post_process_instance_segmentation(outputs, target_sizes=[img_size for _ in range(len(batch["pixel_values"]))])
            elif self.params["model"]["mode"] == "panoptic":
                predictions = self.processor.post_process_panoptic_segmentation(
                    outputs, target_sizes=[img_size for _ in range(len(batch["pixel_values"]))], label_ids_to_fuse=[0]
                )

            for inst_gt, class_labels, pan_pred in zip(batch["mask_labels"], batch["class_labels"], predictions):
                pred_instance_map = pan_pred["segmentation"]
                pred_semantics = torch.zeros(pred_instance_map.shape, dtype=torch.int64)
                instance_ids = torch.unique(pred_instance_map)
                for instance_id in instance_ids:
                    for class_info in pan_pred["segments_info"]:
                        if class_info["id"] == instance_id:
                            pred_semantics[pred_instance_map == instance_id] = class_info["label_id"]
                            break
                pred_semantics[pred_semantics > 2] = 0
                gt_semantics = torch.zeros_like(pred_semantics)
                for inst, class_label in zip(inst_gt, class_labels):
                    gt_semantics[inst > 0] = class_label
                gt_instance_map = torch.zeros_like(gt_semantics)
                for i, inst in enumerate(inst_gt):
                    gt_instance_map[inst > 0] = i + 1
                if self.params["model"]["mode"] == "panoptic":
                    gt_instance_map -= 1
                self.val_stats["PQ"].compute_pq(pred_semantics.cpu(), gt_semantics.cpu(), pred_instance_map.cpu(), gt_instance_map.cpu())
                self.val_stats["IoU"].update(pred_semantics.cpu(), gt_semantics.cpu())
                preds = [
                    dict(
                        masks=pred_semantics.cpu().unsqueeze(0).to(torch.bool),  # tensor([pred_instance_map.cpu()], dtype=torch.bool),
                        scores=tensor([0.536]),
                        labels=tensor([0]),
                    )
                ]
                target = [
                    dict(
                        masks=gt_semantics.cpu().unsqueeze(0).to(torch.bool),  # tensor([gt_instance_map.cpu()], dtype=torch.bool),
                        labels=tensor([0]),
                    )
                ]
                self.val_stats["mAP"].update(preds, target)

        # Loss
        if self.params["training"]["do_val_loss"]:
            self.val_stats["loss"].update(outputs.loss.item())
        return self.train_stats["loss"].avg

    def on_validation_epoch_end(self):
        if self.params["training"]["do_val_metrics"]:
            # Metrics logging
            iou_per_class = self.val_stats["IoU"].compute()
            self.log("val/IoU/mean", round(float(iou_per_class.mean()), 4))
            # self.log("val/IoU/background", round(float(iou_per_class[0]), 4))
            # self.log("val/IoU/plant", round(float(iou_per_class[1]), 4))

            mAP = self.val_stats["mAP"].compute()
            self.log("val/mAP/mean", round(float(mAP["map_50"]), 4))

            pq_per_class = self.val_stats["PQ"].panoptic_qualities
            pq_avg = self.val_stats["PQ"].average_pq(pq_per_class)
            self.log("val/PQ/mean", round(pq_avg, 4))

            # self.log("val/PQ/background", round(pq_per_class[0]["pq"], 4))
            # self.log("val/PQ/plant", round(pq_per_class[1]["pq"], 4))
        # if self.params['training']['do_val_metrics']:
        #     # Metrics logging
        #     iou_per_class = self.val_stats["IoU"].compute()
        #     self.log("val/IoU/mean",
        #              round(float(iou_per_class.mean()), 4))
        #     self.log("val/IoU/soil",
        #              round(float(iou_per_class[0]), 4))
        #     self.log("val/IoU/crop",
        #              round(float(iou_per_class[1]), 4))
        #     self.log("val/IoU/weed",
        #              round(float(iou_per_class[2]), 4))

        #     pq_per_class = self.val_stats["PQ"].panoptic_qualities
        #     pq_avg = self.val_stats["PQ"].average_pq(pq_per_class)
        #     self.log("val/PQ/mean", round(pq_avg, 4))
        #     self.log("val/PQ/crop", round(pq_per_class[1]["pq"], 4))
        #     if self.params['data']['target_type'] == "plant_instances":
        #         self.log("val/PQ/weed", round(pq_per_class[2]["pq"], 4))

        if self.params["training"]["do_val_loss"]:
            self.log("val/loss", self.val_stats["loss"].avg, prog_bar=True)  # , on_step=True)

        for v in self.val_stats.values():
            v.reset()

    def predict_step(self, batch, batch_idx):
        outputs = self.forward(batch)

        # Evaluation
        img_size = batch["pixel_values"].shape[2:]
        # img_size = (self.params['data']['img_size'], self.params['data']['img_size'])
        if self.params["model"]["mode"] == "instance":
            predictions = self.processor.post_process_instance_segmentation(
                outputs,
                target_sizes=[img_size for _ in range(len(batch["pixel_values"]))],
                threshold=self.params["inference"]["threshold"],
                mask_threshold=self.params["inference"]["mask_threshold"],
                overlap_mask_area_threshold=self.params["inference"]["overlap_mask_area_threshold"],
            )
        elif self.params["model"]["mode"] == "panoptic":
            predictions = self.processor.post_process_panoptic_segmentation(
                outputs,
                target_sizes=[img_size for _ in range(len(batch["pixel_values"]))],
                label_ids_to_fuse=[0],
                threshold=self.params["inference"]["threshold"],
                mask_threshold=self.params["inference"]["mask_threshold"],
                overlap_mask_area_threshold=self.params["inference"]["overlap_mask_area_threshold"],
            )

        semantic_predictions = []
        for inst_gt, class_labels, pan_pred in zip(batch["mask_labels"], batch["class_labels"], predictions):
            pred_instance_map = pan_pred["segmentation"]
            pred_semantics = torch.zeros(pred_instance_map.shape, dtype=torch.int64)
            instance_ids = torch.unique(pred_instance_map)
            for instance_id in instance_ids:
                for class_info in pan_pred["segments_info"]:
                    if class_info["id"] == instance_id:
                        pred_semantics[pred_instance_map == instance_id] = class_info["label_id"]
                        break
            semantic_predictions.append(pred_semantics)
            pred_semantics[pred_semantics > 2] = 0
            gt_semantics = torch.zeros_like(pred_semantics)
            for inst, class_label in zip(inst_gt, class_labels):
                gt_semantics[inst > 0] = class_label
            gt_instance_map = torch.zeros_like(gt_semantics)
            for i, inst in enumerate(inst_gt):
                gt_instance_map[inst > 0] = i + 1
            if self.params["model"]["mode"] == "panoptic":
                gt_instance_map -= 1
            self.pred_stats["PQ"].compute_pq(pred_semantics.cpu(), gt_semantics.cpu(), pred_instance_map.cpu(), gt_instance_map.cpu())
            self.pred_stats["IoU"].update(pred_semantics.cpu(), gt_semantics.cpu())
            preds = [
                dict(
                    masks=pred_semantics.cpu().unsqueeze(0).to(torch.bool),  # tensor([pred_instance_map.cpu()], dtype=torch.bool),
                    scores=tensor([0.536]),
                    labels=tensor([0]),
                )
            ]
            target = [
                dict(
                    masks=gt_semantics.cpu().unsqueeze(0).to(torch.bool),  # tensor([gt_instance_map.cpu()], dtype=torch.bool),
                    labels=tensor([0]),
                )
            ]
            self.pred_stats["mAP"].update(preds, target)

        self.pred_stats["loss"].update(outputs.loss.item())

        return predictions, semantic_predictions, batch["image_name"], outputs.loss.item()

    def on_predict_epoch_end(self):
        # Metrics logging
        metrics = {}
        iou_per_class = self.pred_stats["IoU"].compute()
        metrics["IoU"] = {
            "mean": iou_per_class.mean(),
            # "background": iou_per_class[0],
            # "plant": iou_per_class[1],
        }

        mAP = self.val_stats["mAP"].compute()
        metrics["mAP"] = {"mean", round(float(mAP["map_50"]), 4)}

        pq_per_class = self.pred_stats["PQ"].panoptic_qualities
        metrics["PQ"] = {"mean": self.pred_stats["PQ"].average_pq(pq_per_class)}
        # metrics ={}
        # iou_per_class = self.pred_stats["IoU"].compute()
        # metrics["IoU"] = {"mean": iou_per_class.mean(),
        #                   "soil": iou_per_class[0],
        #                   "crop": iou_per_class[1],
        #                   "weed": iou_per_class[2]}
        # metrics["PQ"] = {
        #     "mean": self.pred_stats["PQ"].average_pq(self.pred_stats["PQ"].panoptic_qualities),
        #     "crop": self.pred_stats["PQ"].panoptic_qualities[1]["pq"],
        # }
        # if self.params['data']['target_type'] == "plant_instances":
        #     metrics["PQ"]["weed"] = self.pred_stats["PQ"].panoptic_qualities[2]["pq"]

        metrics["loss"] = self.pred_stats["loss"].avg
        logging.info(metrics)
        return metrics

    def configure_optimizers(self):
        model_params = [
            {"params": self.model.model.transformer_module.parameters()},
            {"params": self.model.class_predictor.parameters()},
            {"params": self.model.model.pixel_level_module.decoder.parameters()},
            {"params": self.model.model.pixel_level_module.encoder.parameters(), "lr": self.params["optimizer"]["lr"] * self.params["optimizer"]["encoder_lr_factor"]},
        ]
        if self.params["optimizer"]["name"] == "Adam":
            optimizer = torch.optim.Adam(model_params, lr=self.params["optimizer"]["lr"], weight_decay=self.params["optimizer"]["weight_decay"])

        elif self.params["optimizer"]["name"] == "AdamW":
            optimizer = torch.optim.AdamW(model_params, lr=self.params["optimizer"]["lr"], weight_decay=self.params["optimizer"]["weight_decay"])

        schedulers = []
        milestones = []

        if self.params["scheduler"]["warmup_steps"] > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=self.params["scheduler"]["warmup_start_multiplier"], total_iters=self.params["scheduler"]["warmup_steps"]
            )
            schedulers.append(scheduler)
            milestones.append(self.params["scheduler"]["warmup_steps"])

        if self.params["scheduler"]["name"] == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.params["scheduler"]["milestones"], gamma=self.params["scheduler"]["gamma"])
        elif self.params["scheduler"]["name"] == "PolynomialLR":
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.params["training"]["max_steps"] - 1 - self.params["scheduler"]["warmup_steps"])

        elif self.params["scheduler"]["name"] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params["training"]["max_steps"] - 1 - self.params["scheduler"]["warmup_steps"])

        elif self.params["scheduler"]["name"] == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.params["scheduler"]["T_0"], T_mult=self.params["scheduler"]["T_mult"])

        schedulers.append(scheduler)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=milestones)

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        ret = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
        return ret
