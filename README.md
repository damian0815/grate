# Grate

Make a matrix of images by running the same prompt through multiple Stable Diffusion models. 

![Demo output of a grid of prompts rendered with htree different stable diffusion models](grate-demo.png)

Supports huggingface repo ids, local CompVis-style `.ckpt` files, and paths to local folder hierarchies containing diffusers-format models. 

The above image was rendered using the following commandline on an empty runpod instance:

```commandline
python3 grate.py 
    --prompts \
        "a cat playing with a ball in the forest" \
        "a dog chasing a postman" \
        "a fish singing the blues" \
    --repo_ids_or_paths \
        "stabilityai/stable-diffusion-2-1" \
        "runwayml/stable-diffusion-v1-5" \
        "nitrosocke/Arcane-Diffusion" \
    --output_path ./grate-demo.png
```

Run `python3 grate.py -h` for help:

```commandline
$ python3 grate.py -h
usage: grate [-h] --prompts PROMPTS [PROMPTS ...] --repo_ids_or_paths REPO_IDS_OR_PATHS [REPO_IDS_OR_PATHS ...] --output_path OUTPUT_PATH
             [--device DEVICE]

Generates a grid of images by running a set of prompts through different Stable Diffusion models.

options:
  -h, --help            show this help message and exit
  --prompts PROMPTS [PROMPTS ...]
                        prompts to render, as strings enclosed in "" and separated by spaces. eg --prompts "a cat" "a dog" "a fish"
  --repo_ids_or_paths REPO_IDS_OR_PATHS [REPO_IDS_OR_PATHS ...]
                        repository ids or paths to models in diffusers or ckpt format, as strings enclosed in "" and separated by spaces.
                        eg --repo_ids_or_paths "stablityai/stable-diffusion-2-1" "../models/v1-5-pruned-emaonly.ckpt"
  --output_path OUTPUT_PATH
                        where to save the resulting image
  --device DEVICE       device to use, eg 'cuda', 'mps', 'cpu'. if omitted, will try to pick the best device.
```

Enjoy!