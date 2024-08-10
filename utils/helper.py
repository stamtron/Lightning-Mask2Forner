import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import label2rgb


def semantics_as_rgb(mask):
    rgb_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_img[mask == 1] = [0, 0, 255]
    rgb_img[mask == 2] = [255, 0, 0]
    return rgb_img


def instances_as_rgb(mask):
    rgb_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    instance_ids = np.unique(mask)[1:]
    for instance_id in instance_ids:
        rgb_img[mask == instance_id] = np.random.randint(0, 255, size=3)
    return rgb_img


def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    image_name = [example["image_name"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels, "image_name": image_name}


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_transforms(params, train_split):
    DS_MEAN = np.array(params["data"]["ds_mean"]) / 255
    DS_STD = np.array(params["data"]["ds_std"]) / 255

    transforms = []

    if train_split:
        # transforms.append(A.OneOf([A.Resize(width = params['data']['img_size'],
        #                                     height = params['data']['img_size']),
        #                            A.RandomResizedCrop(height = params['data']['img_size'],
        #                                                width = params['data']['img_size'],
        #                                                scale = (0.2, 1.0),
        #                                                p = params['data']['p_randomresizedcrop']),],
        #                           p=1.0),)
        transforms.append(A.LongestMaxSize(params["data"]["img_size"], p=1.0))
        transforms.append(A.HorizontalFlip(p=params["data"]["p_horizontalflip"]))
        transforms.append(A.VerticalFlip(p=params["data"]["p_verticalflip"]))
        transforms.append(A.Rotate(p=params["data"]["p_rotate"]))
        transforms.append(A.RGBShift(p=params["data"]["p_rgbshift"]))
        transforms.append(A.RandomBrightnessContrast(p=params["data"]["p_randombrightnesscontrast"]))
        transforms.append(
            A.OneOf(
                [
                    A.GaussianBlur(p=1.0),
                    A.Blur(p=1.0),
                    A.MedianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                ],
                p=params["data"]["p_blur"],
            )
        )

    else:
        transforms.append(A.LongestMaxSize(params["data"]["img_size"], p=1.0))
        # transforms.append(A.Resize(width=params['data']['img_size'],
        #                            height=params['data']['img_size']))
    transforms.append(A.Normalize(mean=DS_MEAN, std=DS_STD))

    return A.Compose(transforms)


def get_transforms_crops(params, train_split):
    DS_MEAN = np.array(params["data"]["ds_mean"]) / 255
    DS_STD = np.array(params["data"]["ds_std"]) / 255

    transforms = []

    if train_split:
        transforms.append(A.HorizontalFlip(p=params["data"]["p_horizontalflip"]))
        transforms.append(A.VerticalFlip(p=params["data"]["p_verticalflip"]))
        transforms.append(A.ShiftScaleRotate(rotate_limit=90, shift_limit=0.0, scale_limit=0.0, p=params["data"]["p_rotate"])),
        transforms.append(A.RGBShift(p=params["data"]["p_rgbshift"]))
        transforms.append(A.RandomBrightnessContrast(p=params["data"]["p_randombrightnesscontrast"]))
        transforms.append(
            A.OneOf(
                [
                    A.GaussianBlur(p=1.0),
                    A.Blur(p=1.0),
                    A.MedianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                ],
                p=params["data"]["p_blur"],
            )
        )
    transforms.append(A.Normalize(mean=DS_MEAN, std=DS_STD))
    return transforms


def get_transforms_leaf_instance_crops(params, train_split):
    return A.Compose(get_transforms_crops(params, train_split), additional_targets={"mask2": "mask"})


def get_transforms_small_plants(params, train_split):
    return A.Compose(get_transforms_crops(params, train_split))


def blow_up_rect(f, x_min, x_max, y_min, y_max, image_shape):
    h = y_max - y_min
    w = x_max - x_min
    y_c = y_min + h / 2
    x_c = x_min + w / 2
    size = max(h, w) * f
    y_min = max(0, int(y_c - size / 2))
    y_max = min(image_shape[0], int(y_c + size / 2))
    x_min = max(0, int(x_c - size / 2))
    x_max = min(image_shape[1], int(x_c + size / 2))
    w, h = x_max - x_min, y_max - y_min
    return x_min, x_max, y_min, y_max, w, h


def compare_countings(ground_truth, predicted, figsize=(16, 4), fontsize=8):
    # Create an array of indices for the x-axis
    x = np.arange(len(ground_truth))

    # Width of each bar
    width = 0.35

    # Create a figure and two subplots
    _, ax = plt.subplots(figsize=figsize)  # Adjust the width and height as needed

    # Plot the bars for ground truth
    rects1 = ax.bar(x - width / 2, ground_truth, width, label="Ground Truth")  # noqa

    # Plot the bars for predicted countings
    rects2 = ax.bar(x + width / 2, predicted, width, label="Predicted")  # noqa

    # Add labels, title, and legend
    ax.set_xlabel("Index")
    ax.set_ylabel("Count")
    ax.set_title("Comparison of Ground Truth and Predicted Countings")
    ax.legend()

    # Adjust the x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=fontsize)  # Adjust the font size as needed

    # Display the chart
    plt.show()


def denormalize_image(params, image):
    # Define ImageNet statistics for mean and standard deviation
    mean = np.array(params["data"]["ds_mean"]) / 255
    std = np.array(params["data"]["ds_std"]) / 255
    denormalized_image = image.copy()
    for i in range(3):
        denormalized_image[i, :, :] = (denormalized_image[i, :, :] * std[i]) + mean[i]
    return denormalized_image


def visualize_predicted_masks(params, img, masks):
    img = denormalize_image(params, img)
    # masks = combine_masks(pred_masks, mask_threshold=mask_threshold)
    if len(masks.shape) == 2:
        masks = np.expand_dims(masks, axis=0)
    # masks = masks.transpose((1, 2, 0))
    # masks = np.dstack([masks, masks, masks])

    fig, axs = plt.subplots(1, 2, figsize=(14, 14))
    # Set individual titles for each subplot
    titles = ["Original Image", "Predicted Masks"]

    # Plot the original image
    axs[0].imshow(img.transpose(1, 2, 0))
    axs[0].set_title(titles[0])

    masks = np.tile(masks, (3, 1, 1))
    print(masks.shape)
    print(img.shape)
    masks = masks.transpose(1, 2, 0)
    img = img.transpose(1, 2, 0)
    # Plot the masks
    axs[1].imshow(label2rgb(masks[:, :, 0], img, bg_label=0))
    axs[1].set_title(titles[1])

    plt.show()
