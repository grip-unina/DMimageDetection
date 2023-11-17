import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse

from normalization import CenterCropNoPad, get_list_norm
from normalization2 import PaddingWarp
from get_method_here import get_method_here, def_model

def run_single_test(image_path, weights_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # List of models
    models_list = {
        'Grag2021_progan': 'Grag2021_progan',
        'Grag2021_latent': 'Grag2021_latent'
    }

    models_dict = dict()
    transform_dict = dict()
    for model_name in models_list:
        _, model_path, arch, norm_type, patch_size = get_method_here(models_list[model_name], weights_path=weights_dir)

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
            transform_key = 'custom'
        else:
            transform_key = 'none'

        transform = transform + get_list_norm(norm_type)
        transform = transforms.Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, model)

    # Process the image
    img = Image.open(image_path).convert('RGB')
    logits = {}
    with torch.no_grad():
        for model_name in models_dict:
            transformed_img = transform_dict[models_dict[model_name][0]](img)
            transformed_img = transformed_img.unsqueeze(0).to(device)
            out_tens = models_dict[model_name][1](transformed_img).cpu().numpy()

            # Check the shape of out_tens and handle accordingly
            print(f"Output tensor shape for {model_name}: {out_tens.shape}")  # Debugging line
            if out_tens.size == 1:
                logits[model_name] = out_tens.item()
            else:
                logits[model_name] = np.mean(out_tens, (2, 3)).item()

    # Print the results
    print(f"Image: {image_path}")
    print("Logits:")
    for model_name, logit in logits.items():
        print(f"{model_name}: {logit}")
    print("Label: " + ("True" if any(value > 0 for value in logits.values()) else "False"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=str, help="The path to the weights of the networks", default="./weights")
    parser.add_argument("--image_path", type=str, help="The path to the image file")
    args = parser.parse_args()
    run_single_test(args.image_path, args.weights_dir)

if __name__ == "__main__":
    main()
