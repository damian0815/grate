# from https://note.com/kohya_ss/n/n9a485a066d5b
# kohya_ss
#   original code: https://github.com/eyriewow/merge-models

# use them as base of this code
# 2022/12/15
# bbc-mc

import os
import argparse
import re
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.import_utils import BACKENDS_MAPPING
from tqdm import tqdm

from transformers import is_safetensors_available

NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS

KEY_POSITION_IDS = "cond_stage_model.transformer.text_model.embeddings.position_ids"

def dprint(str, flg):
    if flg:
        print(str)



def merge_block_weighted_diffusers(weights: list,
                         model_0: StableDiffusionPipeline | dict,
                         model_1: StableDiffusionPipeline | dict,
                         base_alpha=0.5, verbose=False,
                         skip_position_ids=0):

    if type(model_0) is StableDiffusionPipeline:
        model_0 = load_ldm_ckpt_from_diffusers_dicts(unet_state_dict=model_0.unet.state_dict(),
                                                     vae_state_dict=model_0.vae.state_dict(),
                                                     text_enc_dict=model_0.text_encoder.state_dict())
    if type(model_1) is StableDiffusionPipeline:
        model_1 = load_ldm_ckpt_from_diffusers_dicts(unet_state_dict=model_1.unet.state_dict(),
                                                     vae_state_dict=model_1.vae.state_dict(),
                                                     text_enc_dict=model_1.text_encoder.state_dict())

    if len(weights) != NUM_TOTAL_BLOCKS:
        _err_msg = f"weights value must be {NUM_TOTAL_BLOCKS}."
        print(_err_msg)
        return False, _err_msg

    theta_0 = model_0
    theta_1 = model_1

    alpha = base_alpha

    re_inp = re.compile(r'\.input_blocks\.(\d+)\.')  # 12
    re_mid = re.compile(r'\.middle_block\.(\d+)\.')  # 1
    re_out = re.compile(r'\.output_blocks\.(\d+)\.') # 12

    print("  merging ...")
    dprint(f"-- start Stage 1/2 --", verbose)
    count_target_of_basealpha = 0
    for key in (tqdm(theta_0.keys(), desc="Stage 1/2")):
        if "model" in key and key in theta_1:

            if KEY_POSITION_IDS in key:
                print(key)
                if skip_position_ids == 1:
                    print(f"  modelA: skip 'position_ids' : dtype:{theta_0[KEY_POSITION_IDS].dtype}")
                    dprint(f"{theta_0[KEY_POSITION_IDS]}", verbose)
                    continue
                elif skip_position_ids == 2:
                    theta_0[key] = torch.tensor([list(range(77))], dtype=torch.int64)
                    print(f"  modelA: reset 'position_ids': dtype:{theta_0[KEY_POSITION_IDS].dtype}")
                    dprint(f"{theta_0[KEY_POSITION_IDS]}", verbose)
                    continue
                else:
                    print(f"  modelA: 'position_ids' key found. do nothing : {skip_position_ids}: dtype:{theta_0[KEY_POSITION_IDS].dtype}")

            dprint(f"  key : {key}", verbose)
            current_alpha = alpha

            # check weighted and U-Net or not
            if weights is not None and 'model.diffusion_model.' in key:
                # check block index
                weight_index = -1

                if 'time_embed' in key:
                    weight_index = 0                # before input blocks
                elif '.out.' in key:
                    weight_index = NUM_TOTAL_BLOCKS - 1     # after output blocks
                else:
                    m = re_inp.search(key)
                    if m:
                        inp_idx = int(m.groups()[0])
                        weight_index = inp_idx
                    else:
                        m = re_mid.search(key)
                        if m:
                            weight_index = NUM_INPUT_BLOCKS
                        else:
                            m = re_out.search(key)
                            if m:
                                out_idx = int(m.groups()[0])
                                weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + out_idx

                if weight_index >= NUM_TOTAL_BLOCKS:
                    print(f"error. illegal block index: {key}")
                    return False, ""
                if weight_index >= 0:
                    current_alpha = weights[weight_index]
                    dprint(f"weighted '{key}': {current_alpha}", verbose)
            else:
                count_target_of_basealpha = count_target_of_basealpha + 1
                dprint(f"base_alpha applied: [{key}]", verbose)

            theta_0[key] = (1 - current_alpha) * theta_0[key] + current_alpha * theta_1[key]

        else:
            dprint(f"  key - {key}", verbose)

    dprint(f"-- start Stage 2/2 --", verbose)
    for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
        if "model" in key and key not in theta_0:

            if KEY_POSITION_IDS in key:
                if skip_position_ids == 1:
                    print(f"  modelB: skip 'position_ids' : {theta_0[KEY_POSITION_IDS].dtype}")
                    dprint(f"{theta_0[KEY_POSITION_IDS]}", verbose)
                    continue
                elif skip_position_ids == 2:
                    theta_0[key] = torch.tensor([list(range(77))], dtype=torch.int64)
                    print(f"  modelB: reset 'position_ids': {theta_0[KEY_POSITION_IDS].dtype}")
                    dprint(f"{theta_0[KEY_POSITION_IDS]}", verbose)
                    continue
                else:
                    print(f"  modelB: 'position_ids' key found. do nothing : {skip_position_ids}")

            dprint(f"  key : {key}", verbose)
            theta_0.update({key:theta_1[key]})

        else:
            dprint(f"  key - {key}", verbose)

    return theta_0



def load_original_stable_diffusion_state_dict(path) -> dict:
    device = "cpu"
    if os.path.splitext(path)[1].lower() == '.safetensors':
        if not is_safetensors_available():
            raise ValueError(BACKENDS_MAPPING["safetensors"][1])
        from safetensors import safe_open

        checkpoint = {}
        with safe_open(path, framework="pt", device=device) as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(path, map_location=device)

    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    return checkpoint


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
