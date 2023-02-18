# from https://note.com/kohya_ss/n/n9a485a066d5b
# kohya_ss
#   original code: https://github.com/eyriewow/merge-models

# use them as base of this code
# 2022/12/15
# bbc-mc

from diffusers import UNet2DConditionModel
from tqdm import tqdm

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

DIFFUSERS_KEY_PREFIX_TO_WEIGHT_INDEX = {
    "time_embedding.": 0,
    "conv_in.": 0,
    "down_blocks.0.resnets.0": 1,
    "down_blocks.0.attentions.0": 1,
    "down_blocks.0.resnets.1": 2,
    "down_blocks.0.attentions.1": 2,
    "down_blocks.0.downsamplers": 3,
    "down_blocks.1.resnets.0": 4,
    "down_blocks.1.attentions.0": 4,
    "down_blocks.1.resnets.1": 5,
    "down_blocks.1.attentions.1": 5,
    "down_blocks.1.downsamplers": 6,
    "down_blocks.2.resnets.0": 7,
    "down_blocks.2.attentions.0": 7,
    "down_blocks.2.resnets.1": 8,
    "down_blocks.2.attentions.1": 8,
    "down_blocks.2.downsamplers": 9,
    "down_blocks.3.resnets.0": 10,
    "down_blocks.3.resnets.1": 11,
    "mid_block": 12,
    "up_blocks.0.resnets.0": 13,
    "up_blocks.0.resnets.1": 14,
    "up_blocks.0.resnets.2": 15,
    "up_blocks.0.upsamplers.0": 15,
    "up_blocks.1.resnets.0": 16,
    "up_blocks.1.attentions.0": 16,
    "up_blocks.1.resnets.1": 17,
    "up_blocks.1.attentions.1": 17,
    "up_blocks.1.resnets.2": 18,
    "up_blocks.1.upsamplers.0": 18,
    "up_blocks.1.attentions.2": 18,
    "up_blocks.2.resnets.0": 19,
    "up_blocks.2.attentions.0": 19,
    "up_blocks.2.resnets.1": 20,
    "up_blocks.2.attentions.1": 20,
    "up_blocks.2.resnets.2": 21,
    "up_blocks.2.upsamplers.0": 21,
    "up_blocks.2.attentions.2": 21,
    "up_blocks.3.resnets.0": 22,
    "up_blocks.3.attentions.0": 22,
    "up_blocks.3.resnets.1": 23,
    "up_blocks.3.attentions.1": 23,
    "up_blocks.3.resnets.2": 24,
    "up_blocks.3.attentions.2": 24,
    "conv_norm_out.": 24,
    "conv_out.": 24,
}

def dprint(str, flg):
    if flg:
        print(str)

def get_weight_index_diffusers(key: str) -> int:
    for k, v in DIFFUSERS_KEY_PREFIX_TO_WEIGHT_INDEX:
        if key.startswith(k):
            return v
    raise ValueError(f"Unknown unet key: {key}")

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
        raise ValueError(f"weights must have {NUM_TOTAL_BLOCKS} floats.")

    theta_0 = unet_0.state_dict()
    theta_1 = unet_1.state_dict()

    dprint(f"-- start merge --", verbose)

    for key in (tqdm(theta_0.keys(), desc="Merging...")):
        weight_index = get_weight_index_diffusers(key)
        this_block_alpha = weights[weight_index]
        dprint(f"  key : {key} -> alpha {this_block_alpha}", verbose)
        theta_0[key] = (1 - this_block_alpha) * theta_0[key] + this_block_alpha * theta_1[key]

    unet_0.load_state_dict(theta_0)
    return unet_0
