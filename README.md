# PyTorch Lightning Panoptic Segmentation with Mask2Forner

This README provides instructions on how to train, evaluate, and run inference for a `Panoptic Segmentation Mask2Former` model using PyTorch Lightning and Hugging Face Transformers.

## Installation

Before running any of the provided scripts or notebooks, make sure you have the required Python packages installed in your environment.
You can use `requirements.txt` file to install these packages.

To install the necessary packages, follow these steps:

```bash
pip install -r requirements.txt
```

## Training a Model

To train a new model, you can use the `lit_train.py` script. The `lit` prefix stands for `Lightning` which is the framework used for training.
The script accepts various arguments that can be configured through a parameters file located in the `params` folder.

Make sure to update the your configuration file with the appropriate paths for your system.

```yaml
data:
  train_root: /path/to/your/train_data/
  valid_root: /path/to/your/valid_data/

save:
  save_model_path: /path/to/your/save_model_directory/

model:
  pretrained: /path/to/your/pretrained_model
  ckpt_path: /path/to/your/checkpoint

logging:
  output_folder: /path/to/your/log_directory/

```

Here's how you can run the training script:

```bash
python lit_train.py --param_file params/panoptic_segmentation.yaml
```


## Evaluating a model:
To evaluate a model you can use the `00-lit-eval.ipynb` notebook. This notebook demonstrates:

- Loading the saved Lightning checkpoint
- Getting the PyTorch model
- Calculating various metrics on the validation set
- Visualizing the results

However, you need to manually set the variables below:

```python
param_file = 'path/to/your/config.yaml'
ckpt_path = 'path/to/your/ckpt.ckpt'
img_path = 'path/to/your/test_image.jpg'
save_path = 'path/to/your/save_directory/model.pth'
log_folder = 'path/to/your/log_folder/'
save_json_path = 'path/to/your/save_directory/metrics.json'
```


## Example Usage:
This a script that can be used to run inference on a single image.

```python
from utils.helper import *
from transformers import Mask2FormerImageProcessor
from models.mask2former_panoptic import LitMask2FormerPanoptic

# Load params
param_file = 'params/panoptic_segmentation.yaml'
with open(param_file, "r") as f:
    params = yaml.safe_load(f)

if params['model']['mode'] == "panoptic":
    ignore_index = 255

# Load model
processor = Mask2FormerImageProcessor(max_size=params['data']['img_size'],
                                    ignore_index=ignore_index,
                                    do_resize=False,
                                    do_rescale=False,
                                    do_normalize=False)
module = LitMask2FormerPanoptic(params, processor)
module = LitMask2FormerPanoptic.load_from_checkpoint('example.ckpt') # Load from checkpoint
module.eval()

# Load image
transforms = get_transforms(params, False)
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = np.array(Image.open(requests.get(url, stream=True).raw).convert("RGB"))
image = transforms(image=image)["image"]
image = image.transpose(2, 0, 1)
inputs = processor([image], return_tensors="pt")

# Run inference
with torch.no_grad():
    #inputs.to("cpu")
    inputs["mask_labels"] = None
    inputs["class_labels"] = None
    outputs = module(inputs)

```

## Inference script:
To run inference on an image with a Lightning , you can use the `lit_infer.py` script by changing the `img_path` for the test image inside the script.
