# Grate

Make a matrix of images by running the same prompt through multiple Stable Diffusion models. 

![Demo output of a grid of prompts rendered with htree different stable diffusion models](grate-demo.png)

Supports huggingface repo ids, local CompVis-style `.ckpt` files, and paths to local folder hierarchies containing diffusers-format models. Currently using DDPM++2 sampler at 15 steps.

The above image was rendered using the following commandline on an empty runpod instance:

```commandline
python3 grate.py \
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

## Merging

Supports numerous merge methods.

* Basic merges between two models `a` and `b`, using `weighted_sum`, `sigmoid`, or `inv_sigmoid` weighting with `alpha=0` means use only model `a`, and `alpha=1` means use only model `b`. eg: 
    ```commandline
    grate --prompts "a cat" "a dog" \
        --repo_ids_or_paths stabilityai/stable-diffusion-2-1 IlluminatiAI/Illuminati_Diffusion_v1.0 \
        --merge_alphas 0.2 0.5 0.8
    ```
  will produce a grid of 3 rows, 2 columns showing "a cat" and "a dog" rendered using merges between `stabilityai/stable-diffusion-2-1` and `IlluminatiAI/Illuminati_Diffusion_v1.0` at alphas 0.2, 0.5, and 0.8.
  
* Three-way merge by adding the difference between models `b` and `c` to `a` (`--merge_algorithm add_diff`).
* Merges using a different weight for the unet and text_encoder modules (`--merge_unet_alpha` and `--merge_text_encoder_alpha`).
* Block-weighted merges where a different weight can be used for blocks of layers within the unet - 12 weights for the down blocks, 1 weight for the middle block, 12 weights for the up blocks (`--merge_unet_block_weights`).

For each the advanced `--merge_` arguments, you can specify either one value to apply to all rows, or one value per row, where the number of rows is determined by the number of alpha values passed for the `--merge_alphas` argument. For example, to render a grid showing the effects of using the unet down blocks from `stabilityai/stable-diffusion-2-1` and the unet up blocks from `IlluminatiAI/Illuminati_Diffusion_v1.0`, and vice-versa, use this command:

```commandline
    grate --prompts "a cat" "a dog" \
        --repo_ids_or_paths stabilityai/stable-diffusion-2-1 IlluminatiAI/Illuminati_Diffusion_v1.0 \
        --merge_alphas 0.5 0.5 \
        --merge_unet_block_weights "0,0,0,0,0,0,0,0,0,0,0,0,0.5,1,1,1,1,1,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,1,1,1,1,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0"
```

Note that the number of values passed to `--merge_alphas` matches the number of strings passed to `--merge_unet_block_weights`.

## Full arguments list

Run `python3 grate.py -h` for help:

```commandline
usage: grate [-h] --prompts PROMPTS [PROMPTS ...] --repo_ids_or_paths REPO_IDS_OR_PATHS [REPO_IDS_OR_PATHS ...] --output_path OUTPUT_PATH [--device DEVICE]
             [--batch_size BATCH_SIZE] [--width WIDTH] [--height HEIGHT] [--negative_prompts NEGATIVE_PROMPTS [NEGATIVE_PROMPTS ...]] [--seeds SEEDS [SEEDS ...]]
             [--cfg CFG] [--local_files_only] [--merge_alphas MERGE_ALPHAS [MERGE_ALPHAS ...]] [--merge_algorithm MERGE_ALGORITHM [MERGE_ALGORITHM ...]]
             [--merge_unet_block_weights MERGE_UNET_BLOCK_WEIGHTS [MERGE_UNET_BLOCK_WEIGHTS ...]] [--merge_unet_alpha MERGE_UNET_ALPHA [MERGE_UNET_ALPHA ...]]
             [--merge_text_encoder_alpha MERGE_TEXT_ENCODER_ALPHA [MERGE_TEXT_ENCODER_ALPHA ...]]

Generates a grid of images by running a set of prompts through different Stable Diffusion models.

options:
  -h, --help            show this help message and exit
  --prompts PROMPTS [PROMPTS ...]
                        EITHER: a path to a JSON file containing prompt and negative prompt pairs eg [{'prompt': 'a fish', 'negative_prompt': 'distorted', 'seed':
                        123}, ...]. OR: multiple strings enclosed in "" and separated by spaces. eg --prompts "a cat" "a dog" "a fish"
  --repo_ids_or_paths REPO_IDS_OR_PATHS [REPO_IDS_OR_PATHS ...]
                        repository ids or paths to models in diffusers or ckpt format, as strings enclosed in "" and separated by spaces. eg --repo_ids_or_paths
                        "stablityai/stable-diffusion-2-1" "../models/v1-5-pruned-emaonly.ckpt"
  --output_path OUTPUT_PATH
                        Where to save the resulting image. Also saves partial images here as each row is finished rendering.
  --device DEVICE       (Optional) Device to use, eg 'cuda', 'mps', 'cpu'. if omitted, will try to pick the best device.
  --batch_size BATCH_SIZE
                        (Optional) Batch size, default 1
  --width WIDTH         (Optional) Individual image width
  --height HEIGHT       (Optional) Individual image height
  --negative_prompts NEGATIVE_PROMPTS [NEGATIVE_PROMPTS ...]
                        (Optional) Negative prompts. Must be either 1 string to use for all prompts, or one string for each prompt passed to --prompts.
  --seeds SEEDS [SEEDS ...]
                        (Optional) Seeds. Must be either 1 int to use for all prompts, or exactly 1 int per prompt.
  --cfg CFG             (Optional) CFG scale (default=7.5).
  --local_files_only    (Optional) Use only local data (do not attempt to download or update models)
  --merge_alphas MERGE_ALPHAS [MERGE_ALPHAS ...]
                        (Optional) If set, --repo_ids_or_paths must specify either 2 or 3 models, which will be merged using the given alphas and used in place of
                        multiple models.
  --merge_algorithm MERGE_ALGORITHM [MERGE_ALGORITHM ...]
                        (Optional, default is weighted_sum) If doing merges, the algorithm to use - one of 'weighted_sum', 'sigmoid', 'inv_sigmoid', or 'add_diff'. 'add_diff' only works for
                        3-way merge. Specify either 1 algorithm to use for all rows, or the same number of algorithms as there are merge_alphas.
  --merge_unet_block_weights MERGE_UNET_BLOCK_WEIGHTS [MERGE_UNET_BLOCK_WEIGHTS ...]
                        (Optional) 25 comma-separated floats specified as strings, eg "0.0, 0.0, 0.0, (... 22 more floats)", to merge each part of the UNet using a
                        different weight ('block-weighted merging'). Specify either 1 list to use for all rows, or the same number of lists as there are
                        merge_alphas.
  --merge_unet_alpha MERGE_UNET_ALPHA [MERGE_UNET_ALPHA ...]
                        (Optional) Override the merge alpha with a unet-specific alpha. Specify either 1 alpha to use for all rows, or the same number of alphas as
                        there are merge_alphas.
  --merge_text_encoder_alpha MERGE_TEXT_ENCODER_ALPHA [MERGE_TEXT_ENCODER_ALPHA ...]
                        (Optional) Override the merge alpha with a text-encoder-specific alpha. Specify either 1 alpha to use for all rows, or the same number of
                        alphas as there are merge_alphas.

```

Enjoy!

