import base64
import json
import zlib
from pathlib import Path
from warnings import warn

import numpy as np
import torch
from cv2 import IMREAD_UNCHANGED, imdecode
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def decode_base64_mask(s):
    """Convert base64 PNG image to boolean mask."""
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = imdecode(n, IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


def combine_masks(img, ann_path, mode):
    # Read image dimensions
    width, height = img.size

    # Read the annotation file
    with open(ann_path, "r") as json_file:
        annotation_data = json.load(json_file)

        # Create an empty pixel mask
        pixel_mask = Image.new("L", (width, height), 0)

        # Create an empty list for class labels
        if mode == "panoptic":
            class_labels = np.array([0], dtype=np.int32)
        elif mode == "instance":
            class_labels = np.array([], dtype=np.int32)

        # Create an empty list to store individual masks
        masks_list = []

        # Loop through objects in the annotation
        for obj in annotation_data.get("objects", []):
            if obj["geometryType"] == "bitmap":
                # Convert base64 string into a boolean mask image
                mask = decode_base64_mask(obj["bitmap"]["data"])
                mask_array = mask.astype(float)

                # Adjust origin to match image coordinates
                xmin, ymin = obj["bitmap"]["origin"]
                xmax = xmin + mask_array.shape[1]
                ymax = ymin + mask_array.shape[0]

                # Create an empty mask array
                mask_array = np.zeros((height, width), dtype=np.float32)

                # print((mask.astype(np.float32)).shape)
                # print(mask_array[ymin:ymax, xmin:xmax].shape)
                # Check if shapes match before assigning
                if mask.astype(np.float32).shape == mask_array[ymin:ymax, xmin:xmax].shape:
                    mask_array[ymin:ymax, xmin:xmax] = mask.astype(np.float32)
                    # # Ensure that the mask and the target area have compatible sizes
                    # mask_array[ymin:ymax, xmin:xmax] = mask.astype(np.float32)

                    # Append the mask array to the list
                    masks_list.append(mask_array)

                    # Paste the mask onto the pixel mask at the specified origin
                    pixel_mask.paste(Image.fromarray((mask * 255).astype(np.uint8)), (xmin, ymin))

                    # Add the corresponding class label to the list
                    class_labels = np.append(class_labels, 1 if obj["classTitle"] == "Plant" else 0)

    # Convert the pixel mask to a NumPy array for plotting
    semantics_array = np.array(pixel_mask, dtype=np.int32)
    # Replace all occurrences of 22 with 1
    semantics_array[semantics_array == 255] = 1

    # Create an empty mask array
    instance_mask_array = np.zeros_like(semantics_array)
    # Assign unique non-zero values to each mask within the combined_mask
    for idx, mask in enumerate(masks_list, start=1):
        instance_mask_array[mask != 0] = idx

    return semantics_array, instance_mask_array, class_labels


class SuperviselyPanopticDatasetNew(Dataset):
    def __init__(self, paths, processor, max_image_size=1333, mode="panoptic", transform=None, overfit=False, blackout=False):
        self.paths = paths
        self.transform = transform
        self.overfit = overfit
        self.mode = mode
        self.max_image_size = max_image_size
        self.processor = processor
        self.blackout = blackout

        # Load file paths and other necessary data
        data = self.get_data()
        self.data = tuple(data.items())  # turn into tuple of tuples so the order is fixed
        if len(self.data) == 0:
            warn("Empty dataset")

    def get_data(self):
        data = {}
        for path in self.paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(path)
            json_files = sorted((path / "ann").glob("*.json"))  # find all json files
            verbose = True
            it = tqdm(json_files, desc=f"Parsing {path.name}") if verbose else json_files
            for json_file in it:
                # image filename is the same as the json filename without the ".json" ending
                image_filename = json_file.name.split(".json")[0]
                image_path = path / "img" / image_filename
                ann_path = path / "ann" / json_file.name

                # targets = get_targets(path = json_file, labels = None)
                data[str(image_path)] = str(ann_path)
        return data

    def __getitem__(self, idx):
        img_path = self.data[idx][0]
        image_name = img_path.split("/")[-1]
        ann_path = self.data[idx][1]
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")  # get image in original size

        semantics_array, instance_mask_array, class_labels = combine_masks(img, ann_path, self.mode)
        # plant masks consecutive
        # instance_ids = np.unique(instance_mask_array)[1:]
        # if self.mode == "panoptic":
        #     instance_ids = np.arange(0, len(instance_ids) + 1)
        # elif self.mode == "instance":
        #     instance_ids = np.arange(1, len(instance_ids) + 1)
        instance_ids = np.unique(instance_mask_array)

        # FIX THIS LATER ####
        inst2class = {}
        for instance_id in instance_ids:
            # Check if the sequence is empty before finding the argmax
            # if len(semantics_array[instance_mask_array == instance_id]) > 0:
            true_false_mask = instance_mask_array == instance_id
            # Check if there is at least one True value
            if np.any(true_false_mask):
                pass  # print("There is at least one True value in the mask.")
            else:
                print("There are no True values in the mask.")
            tf_semantics = semantics_array[true_false_mask]
            bincount = np.bincount(tf_semantics)
            category = np.argmax(bincount)
            # category = np.argmax(np.bincount(semantics_array[instance_mask_array == instance_id]))
            inst2class[instance_id] = category
            #     empty = False
            # else:
            #     empty = True
            #     # Handle the case where the sequence is empty
            #     print("Sequence is empty, cannot find argmax.")

        image = np.array(img)
        # apply transforms (need to be applied on RGB values)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=instance_mask_array)
            image, instance_mask_array = transformed["image"], transformed["mask"]
            if self.blackout:
                image[instance_mask_array == 0] = [0, 0, 0]
            # convert to C, H, W
            image = image.transpose(2, 0, 1)

        if np.sum(instance_mask_array) == 0:  # or empty:
            # If the image has no objects then it is skipped
            inputs = self.processor([image], return_tensors="pt")
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            inputs["class_labels"] = torch.tensor([0])
            inputs["mask_labels"] = torch.zeros((0, inputs["pixel_values"].shape[-2], inputs["pixel_values"].shape[-1]))
        else:
            inputs = self.processor([image], [instance_mask_array], instance_id_to_semantic_id=inst2class, return_tensors="pt")
            inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        inputs["image_name"] = image_name

        return inputs

    def __len__(self):
        if self.overfit:
            return 1
        else:
            return len(self.data)
