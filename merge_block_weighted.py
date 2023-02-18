# from https://note.com/kohya_ss/n/n9a485a066d5b
# kohya_ss
#   original code: https://github.com/eyriewow/merge-models

# use them as base of this code
# 2022/12/15
# bbc-mc

import os
import argparse
import re
from collections import defaultdict

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import BACKENDS_MAPPING
from tqdm import tqdm

from transformers import is_safetensors_available

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

LDM_TO_DIFFUSERS_UNET_KEY_PREFIX_MAP = {
    "model.diffusion_model.time_embed.": ["time_embedding."],
    "model.diffusion_model.out.": ["conv_norm_out.", "conv_out."],

    "model.diffusion_model.input_blocks.0.": ["conv_in."],
    "model.diffusion_model.input_blocks.1.": ["down_blocks.0.resnets.0", "down_blocks.0.attentions.0"],
    "model.diffusion_model.input_blocks.2.": ["down_blocks.0.resnets.1", "down_blocks.0.attentions.1"],
    "model.diffusion_model.input_blocks.3.": ["down_blocks.0.downsamplers"],
    "model.diffusion_model.input_blocks.4.": ["down_blocks.1.resnets.0", "down_blocks.1.attentions.0"],
    "model.diffusion_model.input_blocks.5.": ["down_blocks.1.resnets.1", "down_blocks.1.attentions.1"],
    "model.diffusion_model.input_blocks.6.": ["down_blocks.1.downsamplers"],
    "model.diffusion_model.input_blocks.7.": ["down_blocks.2.resnets.0", "down_blocks.2.attentions.0"],
    "model.diffusion_model.input_blocks.8.": ["down_blocks.2.resnets.1", "down_blocks.2.attentions.1"],
    "model.diffusion_model.input_blocks.9.": ["down_blocks.2.downsamplers"],
    "model.diffusion_model.input_blocks.10.": ["down_blocks.3.resnets.0"],
    "model.diffusion_model.input_blocks.11.": ["down_blocks.3.resnets.1"],

    "model.diffusion_model.middle_block": ["mid_block"],

    "model.diffusion_model.output_blocks.0.": ["up_blocks.0.resnets.0"],
    "model.diffusion_model.output_blocks.1.": ["up_blocks.0.resnets.1"],
    "model.diffusion_model.output_blocks.2.": ["up_blocks.0.resnets.2", "up_blocks.0.upsamplers.0"],
    "model.diffusion_model.output_blocks.3.": ["up_blocks.1.resnets.0", "up_blocks.1.attentions.0"],
    "model.diffusion_model.output_blocks.4.": ["up_blocks.1.resnets.1", "up_blocks.1.attentions.1"],
    "model.diffusion_model.output_blocks.5.": ["up_blocks.1.resnets.2", "up_blocks.1.upsamplers.0", "up_blocks.1.attentions.2"],
    "model.diffusion_model.output_blocks.6.": ["up_blocks.2.resnets.0", "up_blocks.2.attentions.0"],
    "model.diffusion_model.output_blocks.7.": ["up_blocks.2.resnets.1", "up_blocks.2.attentions.1"],
    "model.diffusion_model.output_blocks.8.": ["up_blocks.2.resnets.2", "up_blocks.2.upsamplers.0", "up_blocks.2.attentions.2"],
    "model.diffusion_model.output_blocks.9.": ["up_blocks.3.resnets.0", "up_blocks.3.attentions.0"],
    "model.diffusion_model.output_blocks.10.": ["up_blocks.3.resnets.1", "up_blocks.3.attentions.1"],
    "model.diffusion_model.output_blocks.11.": ["up_blocks.3.resnets.2", "up_blocks.3.attentions.2"],
}

LDM_PREFIX_TO_WEIGHT_INDEX = {
    "model.diffusion_model.time_embed.": 0,
    "model.diffusion_model.out.": NUM_TOTAL_BLOCKS - 1,

    "model.diffusion_model.input_blocks.0.": 0,
    "model.diffusion_model.input_blocks.1.": 1,
    "model.diffusion_model.input_blocks.2.": 2,
    "model.diffusion_model.input_blocks.3.": 3,
    "model.diffusion_model.input_blocks.4.": 4,
    "model.diffusion_model.input_blocks.5.": 5,
    "model.diffusion_model.input_blocks.6.": 6,
    "model.diffusion_model.input_blocks.7.": 7,
    "model.diffusion_model.input_blocks.8.": 8,
    "model.diffusion_model.input_blocks.9.": 9,
    "model.diffusion_model.input_blocks.10.": 10,
    "model.diffusion_model.input_blocks.11.": 11,

    "model.diffusion_model.middle_block": 12,

    "model.diffusion_model.output_blocks.0.": 13,
    "model.diffusion_model.output_blocks.1.": 14,
    "model.diffusion_model.output_blocks.2.": 15,
    "model.diffusion_model.output_blocks.3.": 16,
    "model.diffusion_model.output_blocks.4.": 17,
    "model.diffusion_model.output_blocks.5.": 18,
    "model.diffusion_model.output_blocks.6.": 19,
    "model.diffusion_model.output_blocks.7.": 20,
    "model.diffusion_model.output_blocks.8.": 21,
    "model.diffusion_model.output_blocks.9.": 22,
    "model.diffusion_model.output_blocks.10.": 23,
    "model.diffusion_model.output_blocks.11.": 24
}

def dprint(str, flg):
    if flg:
        print(str)

def get_weight_index_ldm(key: str) -> int:
    for k, v in LDM_PREFIX_TO_WEIGHT_INDEX:
        if key.startswith(k):
            return v
    raise ValueError(f"Unknown LDM key: {key}")


def get_weight_index_diffusers(key: str) -> int:
    for ldm_prefix, diffusers_prefixes in LDM_TO_DIFFUSERS_UNET_KEY_PREFIX_MAP:
        matching = next((p for p in diffusers_prefixes if key.startswith(p)), None)
        if matching is not None:
            return get_weight_index_ldm(ldm_prefix)
    raise ValueError(f"Unknown diffusers key: {key}")



def merge_unets_block_weighted(weights: list,
                               unet_0: UNet2DConditionModel,
                               unet_1: UNet2DConditionModel,
                               verbose=False,
                               ):
    """
    Merge unet_0 and unet_1 applying a different weight to each level in the unet.

    `weights` is a list of 25 floats:
        12 for the unet down-blocks,
        1 for the middle block,
        12 for the unet up-blocks.
    Each weight is 0..1 where 0 means use unet_0's value and 1 means use unet_1's value.
    """

    if len(weights) != NUM_TOTAL_BLOCKS:
        _err_msg = f"weights value must be {NUM_TOTAL_BLOCKS}."
        print(_err_msg)
        return False, _err_msg

    theta_0 = unet_0.state_dict()
    theta_1 = unet_1.state_dict()

    print("  merging ...")
    dprint(f"-- start Stage 1/2 --", verbose)

    for key in (tqdm(theta_0.keys(), desc="Stage 1/2")):
        weight_index = get_weight_index_diffusers(key)
        this_block_alpha = weights[weight_index]
        dprint(f"  key : {key} -> alpha {this_block_alpha}", verbose)
        theta_0[key] = (1 - this_block_alpha) * theta_0[key] + this_block_alpha * theta_1[key]

    unet_0.load_state_dict(theta_0)
    return unet_0


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_0_path', type=str)
    argparser.add_argument('model_1_path', type=str)
    argparser.add_argument('--base_alpha', type=float, required=True, help="base alpha for the merge")
    argparser.add_argument('--weights', type=str, required=True, help="weights for each unet block (12 input -> 1 mid-> 12 output) as a string containing a comma-separated list of 25 floats")
    argparser.add_argument('--output_file', type=str, required=True, help="path to output file (must be a .safetensors or .ckpt file, will be saved in LDM format)")
    argparser.add_argument('--verbose', action='store_true')
    argparser.add_argument('--skip_position_ids', type=int, required=False, default=0)

    args = argparser.parse_args()

    if os.path.splitext(args.output_file)[1].lower() not in ['.safetensors', '.ckpt']:
        raise ValueError(f"Output file {args.output_file} not of supported type (must be .safetensors or .ckpt)")

    print(f"loading {args.model_0_path}...")
    if os.path.isfile(args.model_0_path):
        model_0 = load_original_stable_diffusion_state_dict(args.model_0_path)
    else:
        model_0 = StableDiffusionPipeline.from_pretrained(args.model_0_path)

    print(f"loading {args.model_1_path}...")
    if os.path.isfile(args.model_1_path):
        model_1 = load_original_stable_diffusion_state_dict(args.model_1_path)
    else:
        model_1 = StableDiffusionPipeline.from_pretrained(args.model_1_path)

    weights = [float(x) for x in args.weights.split(',')]

    print("merging...")
    merged = merge_block_weighted_diffusers(weights, model_0, model_1,
                                   base_alpha=args.base_alpha,
                                   verbose=args.verbose,
                                   skip_position_ids=args.skip_position_ids)

    print(f"saving to {args.output_file}...")

    _, extension = os.path.splitext(args.output_file)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        safetensors.torch.save_file(merged, args.output_file, metadata={"format": "pt"})
    else:
        torch.save({"state_dict": merged}, args.output_file)

    print("Done!")
