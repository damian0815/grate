# Grate

Make a matrix of images by running the same prompt through multiple Stable Diffusion models. 

![Demo output of a grid of prompts rendered with three different stable diffusion models](grate-demo.png)

Supports huggingface repo ids, local CompVis-style `.ckpt` files, and paths to local folder hierarchies containing diffusers-format models. Uses DDPM++2 sampler with 15 steps for all generations.

The above image was rendered using the following commandline on an empty runpod instance:

```commandline
grate --prompts \
        "a cat playing with a ball in the forest" \
        "a dog chasing a postman" \
        "a fish singing the blues" \
    --repo_ids_or_paths \
        "stabilityai/stable-diffusion-2-1" \
        "runwayml/stable-diffusion-v1-5" \
        "nitrosocke/Arcane-Diffusion" \
    --output_path ./grate-demo.png
```

## Installation

```commandline
python -m pip install sdgrate
```

Once it's installed, run `grate -h` for help.

## Merging

Supports numerous merge methods.

* Basic merges between two models `a` and `b`, using `weighted_sum`, `sigmoid`, or `inv_sigmoid` weighting, with a given alpha (`alpha=0` means use only model `a`, and `alpha=1` means use only model `b`). For example: 
    ```commandline
    grate --prompts "a cat" "a dog" \
        --repo_ids_or_paths stabilityai/stable-diffusion-2-1 IlluminatiAI/Illuminati_Diffusion_v1.0 \
        --merge_alphas 0.2 0.5 0.8
    ```
  This will produce a grid with 3 rows of 2 columns showing the prompts "a cat" and "a dog" rendered using merges between `stabilityai/stable-diffusion-2-1` and `IlluminatiAI/Illuminati_Diffusion_v1.0` at alphas 0.2, 0.5, and 0.8.
  
* Three-way merge by adding the difference between models `b` and `c` to `a` (`--merge_algorithm add_diff`).
* Merges using a different weight for the unet and text_encoder modules (`--merge_unet_alpha` and `--merge_text_encoder_alpha`).
* Block-weighted merges, whereby a different weight can be used for different layers within the unet (`--merge_unet_block_weights`). Specify 12 weights for the down blocks (counting from the input layer), 1 weight for the middle block, and 12 weights for the up blocks (counting from the middle layer). An example is given below. 
  > You can find [an explanation of block-weighted merging here](https://rentry.org/Merge_Block_Weight_-china-_v1_Beta#merge-block-weight-magic-codex-10beta) (cw: waifus), and some weight presets to use [here](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui/blob/master/csv/preset.tsv) (illustrated [here](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui#presets-grids)).

For each the advanced `--merge_` arguments, you can specify either one value to apply to all rows, or one value per row, where the number of rows is determined by the number of alpha values passed for the `--merge_alphas` argument. For example, to render a grid showing the effects of using the unet down blocks from `stabilityai/stable-diffusion-2-1` and the unet up blocks from `IlluminatiAI/Illuminati_Diffusion_v1.0` on the first row, and the effect of doing the opposite weighting on the second row, use this command:

```commandline
    grate --prompts "a cat" "a dog" \
        --repo_ids_or_paths stabilityai/stable-diffusion-2-1 IlluminatiAI/Illuminati_Diffusion_v1.0 \
        --merge_alphas 0.5 0.5 \
        --merge_unet_block_weights "0,0,0,0,0,0,0,0,0,0,0,0,0.5,1,1,1,1,1,1,1,1,1,1,1,1" "1,1,1,1,1,1,1,1,1,1,1,1,0.5,0,0,0,0,0,0,0,0,0,0,0,0" \
        --output_path /tmp/unet-test-cat-dog.jpg
```

Note that the number of values passed to `--merge_alphas` matches the number of strings passed to `--merge_unet_block_weights`.

## Full arguments list

Run `python3 grate.py -h` for help:

```commandline
$ grate -h 
usage: grate [-h] --prompts PROMPTS [PROMPTS ...] --repo_ids_or_paths
             REPO_IDS_OR_PATHS [REPO_IDS_OR_PATHS ...] --output_path
             OUTPUT_PATH [--device DEVICE] [--batch_size BATCH_SIZE]
             [--width WIDTH] [--height HEIGHT]
             [--negative_prompts NEGATIVE_PROMPTS [NEGATIVE_PROMPTS ...]]
             [--seeds SEEDS [SEEDS ...]]
             [--scheduler {ddim,lms,dpm++,k_dpm++,dpm++_sde,k_dpm++_sde,euler_a,k_euler_a,pndm,ddpm,k_dpm2_a}]
             [--cfg CFG] [--steps STEPS] [--disable_nsfw_checker]
             [--local_files_only]
             [--merge_alpha MERGE_ALPHA [MERGE_ALPHA ...]]
             [--merge_algorithm MERGE_ALGORITHM [MERGE_ALGORITHM ...]]
             [--merge_unet_block_weights MERGE_UNET_BLOCK_WEIGHTS [MERGE_UNET_BLOCK_WEIGHTS ...]]
             [--merge_unet_alpha MERGE_UNET_ALPHA [MERGE_UNET_ALPHA ...]]
             [--merge_text_encoder_alpha MERGE_TEXT_ENCODER_ALPHA [MERGE_TEXT_ENCODER_ALPHA ...]]
             [--save_merge_path_prefix SAVE_MERGE_PATH_PREFIX]
             [--save_merge_float32] [--use_penultimate_clip_layer]

Generates a grid of images by running a set of prompts through different
Stable Diffusion models. Optionally, merge models together: if one or more of
the --merge_ options is passed, grate will produce multiple merged models
using all possible combinations of the passed values, and render each on its
own row in the output image. For example, grate --merge_alphas 0.333 0.667
--merge_algorithm weighted_sum sigmoid will produce an output image with 4
rows, representing a weighted_sum merge with alpha 0.333, a weighted_sum merge
with alpha 0.667, a sigmoid merge with alpha 0.333, and a sigmoid merge with
alpha 0.667, respectively. When merging, you must specify either 2 or 3 values
for --repo_ids_or_paths .

options:
  -h, --help            show this help message and exit
  --prompts PROMPTS [PROMPTS ...]
                        EITHER: a path to a JSON file containing prompts with
                        optional seeds and negative prompts, eg [{'prompt': 'a
                        fish', 'negative_prompt': 'distorted', 'seed': 123},
                        ...]. OR: multiple strings enclosed in "" and
                        separated by spaces. eg --prompts "a cat" "a dog" "a
                        fish"
  --repo_ids_or_paths REPO_IDS_OR_PATHS [REPO_IDS_OR_PATHS ...]
                        repository ids or paths to models in diffusers or ckpt
                        format, as strings enclosed in "" and separated by
                        spaces. eg --repo_ids_or_paths "stablityai/stable-
                        diffusion-2-1" "../models/v1-5-pruned-emaonly.ckpt"
  --output_path OUTPUT_PATH
                        Where to save the resulting image. Also saves
                        partially-rendered images to this location as each row
                        finishes rendering.
  --device DEVICE       (Optional) Device to use, eg 'cuda', 'mps', 'cpu'. if
                        omitted, will try to pick the best device.
  --batch_size BATCH_SIZE
                        (Optional, default=1) Batch size.
  --width WIDTH         (Optional, default=512) Individual image width
  --height HEIGHT       (Optional, defatul=512) Individual image height
  --negative_prompts NEGATIVE_PROMPTS [NEGATIVE_PROMPTS ...]
                        (Optional) Negative prompts. Specify either one string
                        to share for all `--prompts`, or as many strings as
                        there are `--prompts`.
  --seeds SEEDS [SEEDS ...]
                        (Optional) Seeds. Specify either one seed to share for
                        all `--prompts`, or as many seeds as there are
                        `--prompts`.
  --scheduler {ddim,lms,dpm++,k_dpm++,dpm++_sde,k_dpm++_sde,euler_a,k_euler_a,pndm,ddpm,k_dpm2_a}
                        (Optional, default=dpm++) Scheduler to use.
  --cfg CFG             (Optional, default=7.5) CFG scale.
  --steps STEPS         (Optional, default=15) How many inference steps to run
  --disable_nsfw_checker
                        (Optional)
  --local_files_only    (Optional) Use only local data (do not attempt to
                        download or update models)
  --merge_alpha MERGE_ALPHA [MERGE_ALPHA ...]
                        (Optional) If set, --repo_ids_or_paths must specify
                        either 2 or 3 models, which will be merged using the
                        given alpha and used in place of multiple models.
  --merge_algorithm MERGE_ALGORITHM [MERGE_ALGORITHM ...]
                        (Optional) If doing merges, the algorithm to use - one
                        of 'weighted_sum', 'sigmoid', 'inv_sigmoid', or
                        'add_diff'. 'add_diff' only works for 3-way merge.
  --merge_unet_block_weights MERGE_UNET_BLOCK_WEIGHTS [MERGE_UNET_BLOCK_WEIGHTS ...]
                        (Optional) 25 comma-separated floats specified as
                        strings, eg "0.0, 0.0, 0.0, (... 22 more floats)", to
                        merge each part of the UNet using a different weight
                        ('block-weighted merging').
  --merge_unet_alpha MERGE_UNET_ALPHA [MERGE_UNET_ALPHA ...]
                        (Optional) Override the merge alpha with a unet-
                        specific alpha.
  --merge_text_encoder_alpha MERGE_TEXT_ENCODER_ALPHA [MERGE_TEXT_ENCODER_ALPHA ...]
                        (Optional) Override the merge alpha with a text-
                        encoder-specific alpha.
  --save_merge_path_prefix SAVE_MERGE_PATH_PREFIX
                        (Optional) If doing a merge, save all merge
                        combinations using this path as a prefix.
  --save_merge_float32  (Optional) If saving merges, save with float32
                        precision (default is float16).
  --use_penultimate_clip_layer
                        (Optional) Use the outputs from penultimate (second to
                        last) CLIP hidden layer. On detected SD2.x models this
                        defaults on, otherwise it defaults off.
```

Enjoy!

## Using as a library

The main `sdgrate.grate` module includes the following functions, which may be useful: `merge_models`, `render_row`, `render_all`. 

The model merger is implemented as a custom pipeline based on a modified version of the (checkpoint_merger pipeline)

## Changelog

#### 0.2.4 - add `--scheduler` arg and fix diffusers 0.16 support 

#### 0.2.3 - fix crash when running compel, fix diffusers 0.15 support actually

#### 0.2.2 - added `--use_penultimate_clip_layer` arg ~~for improved SD2 generation quality~~ (aka "clip skip")