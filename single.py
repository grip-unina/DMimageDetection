"""
This module is used for running inference on a single image using pre-trained
models. It supports different models and applies necessary transformations to the
input image before feeding it to the models for prediction.

It prints out the logits returned by each model and the final label based on these logits.
"""

import argparse
import numbers
import time
import json
import os
import torch
import psutil
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import networks.networks.resnet_mod as resnet_mod

models_config = {
    "Grag2021_progan": {
        "model_path": "./weights/Grag2021_progan/model_epoch_best.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
        "patch_size": None,
    },
    "Grag2021_latent": {
        "model_path": "./weights/Grag2021_latent/model_epoch_best.pth",
        "arch": "res50stride1",
        "norm_type": "resnet",
        "patch_size": None,
    },
}


def print_memory_usage():
    """
    Prints the current memory usage of the process.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory used: {mem_info.rss / (1024 * 1024):.2f} MB")


def load_model(arch, model_path):
    """
    Loads the specified model architecture with given weights.

    Args:
        arch (str): The architecture of the model.
        model_path (str): Path to the model's weights.

    Returns:
        torch.nn.Module: The loaded model.
    """
    if arch == "res50stride1":
        model = resnet_mod.resnet50(num_classes=1, gap_size=1, stride0=1)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Load the entire checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Extract the state dictionary for the model
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise KeyError("No model state_dict found in the checkpoint file")

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    return model


def center_crop(img, output_size):
    """
    Performs center cropping of the image.

    Args:
        img (PIL.Image): Input image.
        output_size (tuple): Desired output size.

    Returns:
        PIL.Image: Center cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size[1], img.size[0]
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return img.crop(
        (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
    )


class CenterCropNoPad:
    """
    Applies center crop to the image without padding.
    """

    def __init__(self, siz):
        self.siz = siz

    def __call__(self, img):
        return center_crop(img, self.siz)


class PaddingWarp:
    """
    Applies padding to the image.
    """

    def __init__(self, siz):
        self.siz = siz

    def __call__(self, img):
        return padding_wrap(img, self.siz)


def padding_wrap(img, output_size):
    """
    Wraps the image with padding.

    Args:
        img (PIL.Image): Input image.
        output_size (tuple): Output size after padding.

    Returns:
        PIL.Image: Padded image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    new_img = Image.new(img.mode, output_size)
    for x_offset in range(0, output_size[0], img.size[0]):
        for y_offset in range(0, output_size[1], img.size[1]):
            new_img.paste(img, (x_offset, y_offset))
    return new_img


def get_list_norm(norm_type):
    """
    Gets the list of transformations based on the normalization type.

    Args:
        norm_type (str): Type of normalization to apply.

    Returns:
        list: List of transformations.
    """
    transforms_list = list()
    if norm_type == "resnet":
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    elif norm_type == "none":
        transforms_list.append(transforms.ToTensor())
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
    return transforms_list


def run_single_test(image_path, weights_dir, debug):
    """
    Runs inference on a single image using specified models and weights.
    Loads each model individually to optimize memory usage.

    Args:
        image_path (str): Path to the image file for inference.
        weights_dir (str): Directory where the model weights are stored.
        debug (bool): Flag to enable debug mode for additional information.

    Returns:
        dict: JSON object with detection results and execution time.
    """
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logits = {}

    img = Image.open(image_path).convert("RGB")

    for model_name, config in models_config.items():
        # Load and process each model individually
        model = load_model(config["arch"], config["model_path"])
        model = model.to(device).eval()
        transform = transforms.Compose(get_list_norm(config["norm_type"]))

        with torch.no_grad():
            transformed_img = transform(img)
            transformed_img = transformed_img.unsqueeze(0).to(device)
            out_tens = model(transformed_img).cpu().numpy()
            logits[model_name] = np.mean(out_tens, (2, 3)).item()

        # Unload model from memory
        del model
        torch.cuda.empty_cache()

        if debug:
            print(f"Model {model_name} processed.")
            print_memory_usage()

    execution_time = time.time() - start_time
    label = "True" if any(value > 0 for value in logits.values()) else "False"

    # Construct output JSON
    output = {
        "product": "diffusion-model-detector",
        "detection": {
            "logit": logits,
            "IsDiffusionImage?": label,
            "ExecutionTime": execution_time
        }
    }

    return output


def main():
    """
    The main function of the script. It parses command-line arguments and runs the inference test.

    The function expects three command-line arguments:
    - `--weights_dir`: The directory where the model weights are stored.
    - `--image_path`: The path to the image file on which inference is to be performed.
    - `--debug`: Show memory usage or not

    After parsing the arguments, it calls the `run_single_test` function to perform inference
    on the specified image using the provided model weights.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_dir",
        type=str,
        help="The path to the weights of the networks",
        default="./weights",
    )
    parser.add_argument("--image_path", type=str, help="The path to the image file")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to print memory usage"
    )
    args = parser.parse_args()
    output = run_single_test(args.image_path, args.weights_dir, debug=args.debug)
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    main()
