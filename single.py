"""
This module is used for running inference on a single image using pre-trained
models. It supports different models and applies necessary transformations to the
input image before feeding it to the models for prediction.

It prints out the logits returned by each model and the final label based on these logits.
"""

import argparse
import time
import json
import os
import torch
import psutil
import numpy as np
from PIL import Image, UnidentifiedImageError
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

def compress_and_resize_image(image_path, max_size=(1024, 1024)):
    """
    Compresses and resizes an image to a manageable size.

    Args:
        image_path (str): Path to the image file.
        max_size (tuple): Maximum width and height of the resized image.

    Returns:
        str: Path to the processed image.
    """
    try:
        # Validate the file format
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            raise ValueError("Unsupported file format. Accepts only JPEG, PNG, and WebP.")

        # Open and process the image
        with Image.open(image_path) as img:
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                # Resize the image only if it's larger than max_size
                img.thumbnail(max_size, Image.LANCZOS)
            # Save the processed image in a lossless format
            processed_image_path = os.path.splitext(image_path)[0] + "_processed.png"
            img.save(processed_image_path, format='PNG', optimize=True)
            return processed_image_path

    except UnidentifiedImageError as exc:
        # Explicitly re-raising with context from the original exception
        raise ValueError("Invalid image file or path.") from exc

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


def run_single_test(image_path, debug):
    """
    Runs inference on a single image using specified models and weights.
    Loads each model individually to optimize memory usage.

    Args:
        image_path (str): Path to the image file for inference.
        debug (bool): Flag to enable debug mode for additional information.

    Returns:
        dict: JSON object with detection results and execution time.
    """
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logits = {}

    processed_image_path = compress_and_resize_image(image_path)
    img = Image.open(processed_image_path).convert("RGB")

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

    # Calculate if the image is fake or not

    threshold=0.5

    def calculate_sigmoid_probabilities(logits_dict):
        """
        Adds sigmoid probabilities to the logits dictionary.

        Parameters:
        logits_dict (dict): Dictionary containing logits.

        Returns:
        dict: Updated dictionary with sigmoid probabilities.
        """
        sigmoid_probs = {}
        for model, logit in logits_dict.items():
            sigmoid_prob = 1 / (1 + np.exp(-logit))  # Sigmoid function
            sigmoid_probs[model] = sigmoid_prob
        return sigmoid_probs

    sigmoid_probs = calculate_sigmoid_probabilities(logits)

    for prob in sigmoid_probs.values():
        if prob >= threshold:
            isDifussionImage = True  # Image is classified as fake
            break
        else:
            isDifussionImage = False

    detection_output = {
        "model": "gan-model-detector",
        "inferenceResults": {
            "logits": logits,
            "probabilities": sigmoid_probs,
            "isDiffusionImage": isDifussionImage,
            "executionTime": execution_time,
        },
    }

    return detection_output

def main():
    """
    The main function of the script. It parses command-line arguments and runs the inference test.

    The function expects three command-line arguments:
    - `--image_path`: The path to the image file on which inference is to be performed.
    - `--debug`: Show memory usage or not

    After parsing the arguments, it calls the `run_single_test` function to perform inference
    on the specified image using the provided model weights.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="The path to the image file")
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to print memory usage"
    )
    args = parser.parse_args()
    output = run_single_test(args.image_path, debug=args.debug)
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    main()
