import os

import cv2
import numpy as np
import requests
import torch
import yaml
from PIL import Image
from transformers import Mask2FormerImageProcessor

from models.mask2former_panoptic import LitMask2FormerPanoptic
from utils.helper import get_transforms


def main(param_file="params/panoptic_params.yaml"):

    # Load params
    with open(param_file, "r") as f:
        params = yaml.safe_load(f)

    if params["model"]["mode"] == "panoptic":
        ignore_index = 255

    # Load model
    print("Loading model...")
    processor = Mask2FormerImageProcessor(
        longest_edge=params["data"]["img_size"],
        # max_size=params['data']['img_size'],
        ignore_index=ignore_index,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )
    module = LitMask2FormerPanoptic(params, processor)
    # module = LitMask2FormerPanoptic.load_from_checkpoint(params["inference"]["ckpt_path"], map_location="cpu")  # Load from checkpoint
    module.eval()

    # Load image
    print("Loading image...")
    transforms = get_transforms(params, False)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = np.array(Image.open(requests.get(url, stream=True).raw).convert("RGB"))
    # img_path = "test/test_img.jpg"
    # image = np.array(Image.open(img_path).convert("RGB"))
    image = transforms(image=image)["image"]
    image = image.transpose(2, 0, 1)
    inputs = processor([image], return_tensors="pt")

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # inputs.to("cpu")
        inputs["mask_labels"] = None
        inputs["class_labels"] = None
        outputs = module(inputs)

    print("Post-processing...")
    predictions = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[inputs["pixel_values"].shape[2:] for _ in range(len(inputs["pixel_values"]))],
        label_ids_to_fuse=[0],
        threshold=params["inference"]["threshold"],
        mask_threshold=params["inference"]["mask_threshold"],
        overlap_mask_area_threshold=params["inference"]["overlap_mask_area_threshold"],
    )

    pred_instance_map = predictions[0]["segmentation"].cpu().numpy()
    pred_semantics = torch.zeros(pred_instance_map.shape)
    instance_ids = np.unique(pred_instance_map)
    for instance_id in instance_ids:
        for class_info in predictions[0]["segments_info"]:
            if class_info["id"] == instance_id:
                pred_semantics[pred_instance_map == instance_id] = class_info["label_id"]
                break
    pred_semantics[pred_semantics > 2] = 0

    if params["inference"]["save_detections"]:
        print("Saving detections...")
        log_folder = params["inference"]["output_folder"]
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(f"{log_folder}/plant_instances", exist_ok=True)
        os.makedirs(f"{log_folder}/semantics", exist_ok=True)

        cv2.imwrite(f"{log_folder}/plant_instances/inf_img_inst.png", pred_instance_map.astype(np.uint64))
        cv2.imwrite(f"{log_folder}/semantics/inf_img_sem.png", pred_semantics.cpu().numpy().astype(np.uint64))


if __name__ == "__main__":
    main()
