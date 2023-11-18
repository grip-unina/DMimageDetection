"""
This module is used for running inference on a single image using pre-trained
models. It supports different models and applies necessary transformations to the
input image before feeding it to the models for prediction.

It prints out the logits returned by each model and the final label based on these logits.
"""

import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from normalization import CenterCropNoPad, get_list_norm
from normalization2 import PaddingWarp
from get_method_here import get_method_here, def_model


def run_single_test(image_path, weights_dir):
    """
    Runs inference on a single image using specified models and weights.

    Args:
        image_path (str): Path to the image file on which inference is to be performed.
        weights_dir (str): Directory where the model weights are stored.

    The function loads the models, applies the necessary transformations to the image,
    and then feeds the image to the models. It prints out the logits from each model
    and the final label (True/False) based on these logits.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # List of models
    models_list = {
        "Grag2021_progan": "Grag2021_progan",
        "Grag2021_latent": "Grag2021_latent",
    }

    models_dict = dict()
    transform_dict = dict()
    for model_name, model_alias in models_list.items():
        _, model_path, arch, norm_type, patch_size = get_method_here(
            model_alias, weights_path=weights_dir
        )

        model = def_model(arch, model_path, localize=False)
        model = model.to(device).eval()

        transform = list()
        if patch_size is not None:
            if isinstance(patch_size, tuple):
                transform.append(transforms.Resize(*patch_size))
            else:
                if patch_size > 0:
                    transform.append(CenterCropNoPad(patch_size))
                else:
                    transform.append(CenterCropNoPad(-patch_size))
                    transform.append(PaddingWarp(-patch_size))
            transform_key = "custom"
        else:
            transform_key = "none"

        transform = transform + get_list_norm(norm_type)
        transform = transforms.Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)

    img = Image.open(image_path).convert("RGB")
    logits = {}
    with torch.no_grad():
        for model_name, (transform_key, model) in models_dict.items():
            transformed_img = transform_dict[transform_key](img)
            transformed_img = transformed_img.unsqueeze(0).to(device)
            out_tens = model(transformed_img).cpu().numpy()

            if out_tens.size == 1:
                logits[model_name] = out_tens.item()
            else:
                logits[model_name] = np.mean(out_tens, (2, 3)).item()

    print(f"Image: {image_path}")
    print("Logits:")
    for model_name, logit in logits.items():
        print(f"{model_name}: {logit}")
    print(
        "Label: " + ("True" if any(value > 0 for value in logits.values()) else "False")
    )


def main():
    """
    Main function to parse arguments and run the test.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_dir",
        type=str,
        help="The path to the weights of the networks",
        default="./weights",
    )
    parser.add_argument("--image_path", type=str, help="The path to the image file")
    args = parser.parse_args()
    run_single_test(args.image_path, args.weights_dir)


if __name__ == "__main__":
    main()
