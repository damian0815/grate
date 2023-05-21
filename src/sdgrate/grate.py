# adapted from https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb#scrollTo=36026528
import json
import os
import argparse
import pathlib
from typing import Optional

import torch
from compel import Compel
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, DPMSolverMultistepScheduler, \
    EulerAncestralDiscreteScheduler, PNDMScheduler, DDPMScheduler, KDPM2AncestralDiscreteScheduler, DDIMScheduler
from PIL import Image, ImageDraw, ImageFont
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import is_xformers_available
from tqdm import tqdm

from huggingface_hub import list_repo_refs


def disable_nsfw_safety_checker(images, **kwargs):
    return images, False


# get_wrapped_text adapted from https://stackoverflow.com/a/67203353
def get_wrapped_text(text: str, font: ImageFont,
                     line_width: int,
                     max_height: int=None, draw: ImageDraw=None,
                     max_fontsize_reduction_iterations=3) -> tuple[list[str], int|None]:
    """
    Wrap `text` so that when it is drawn using `font` it can fit horizontally within the given
    `line_width` pixels.

    If `max_height` is not None, attempt to reduce the font size at most
    `max_fontsize_reduction_iterations` times until it fits in the requested height. Requires
    that `draw` is also set.

    Returns a list of lines, and a reduced font size or None if no font size reduction
    was calculated.
    """

    remaining_fontsize_reduction_iterations = max_fontsize_reduction_iterations + 1
    reduced_font_size = None
    while True:
        lines = []
        words = text.split()
        current_line = None
        while len(words) > 0:
            next_word = words.pop(0)
            current_line_with_next_word = (next_word
                                           if current_line is None
                                           else f'{current_line} {next_word}')
            if font.getlength(current_line_with_next_word) <= line_width:
                current_line = current_line_with_next_word
            else:
                # break here
                if current_line is not None:
                    lines.append(current_line)
                    words.insert(0, next_word)
                    current_line = None
                else:
                    # single word is too long, need to truncate
                    truncated_word = next_word
                    while len(truncated_word) > 0 and font.getlength(truncated_word) > line_width:
                        truncated_word = truncated_word[:-1]
                    if len(truncated_word) > 0:
                        lines.append(truncated_word)
                        words.insert(0, next_word[len(truncated_word):])
                    else:
                        # give up
                        lines.append(next_word)
        if current_line is not None:
            lines.append(current_line)
        bbox = draw.multiline_textbbox((0,0), "\n".join(lines), font=font) # (left, top, right, bottom)
        if max_height is None or remaining_fontsize_reduction_iterations <= 0 or abs(bbox[3]-bbox[1]) <= max_height:
            # fits in box, or we shouldn't try again
            break
        # reduce font to 75% of current size and try again
        reduced_font_size = round(font.size*0.75)
        font = ImageFont.truetype(font.path, reduced_font_size)
        remaining_fontsize_reduction_iterations -= 1

    return lines, reduced_font_size


def make_label_image(label: str, font: ImageFont, width: int, height: int, margins=None):
    if margins is None:
        margins = [10, 10, 10, 10]
    line_spacing = 5  # pixels between lines
    label_image = Image.new('RGB', size=(width, height), color=(247, 247, 247))
    draw = ImageDraw.Draw(label_image)
    wrapped_lines_list, reduced_font_size_or_none = get_wrapped_text(label,
                                                                     font,
                                                                     line_width=width - (margins[0] + margins[2]),
                                                                     max_height=height - (margins[1] + margins[3]), 
                                                                     draw=draw)
    wrapped_text = '\n'.join(wrapped_lines_list)
    possibly_reduced_font = font if reduced_font_size_or_none is None else ImageFont.truetype(font.path, reduced_font_size_or_none)
    text_bbox = draw.multiline_textbbox((width / 2, 0),
                                        text=wrapped_text,
                                        font=possibly_reduced_font,
                                        anchor='ma',
                                        spacing=line_spacing,
                                        )  # anchor = horizontal middle, ascender
    text_bbox_height = text_bbox[3] - text_bbox[1]
    # print("got text_bbox ", text_bbox)
    y_offset = (height - text_bbox_height) / 2
    x_offset = (width - margins[2] - margins[0]) // 2
    draw.multiline_text((margins[0] + x_offset, margins[1] + y_offset),
                        text=wrapped_text, font=possibly_reduced_font, anchor='ma', fill=(64, 48, 32))
    return label_image


def make_image_grid(imgs, num_rows, num_cols, row_labels: list[str], col_labels: list[str]):
    assert len(imgs) == num_rows * num_cols
    assert len(row_labels) == num_rows
    assert len(col_labels) == num_cols

    print("allocating image...")

    w, h = imgs[0].size
    margin = int(0.125 * w)
    spacing = int(0.125 * w)
    grid = Image.new('RGB', size=(margin + (num_cols + 1) * (w + spacing), margin + (num_rows + 1) * (h + spacing)),
                     color=(255, 255, 255))
    # grid_w, grid_h = grid.size

    # font = ImageFont.load_default()
    font_path = os.path.join(os.path.dirname(__file__), "LibreBaskerville-DpdE.ttf")
    col_font = ImageFont.truetype(font_path, size=32)
    row_font = ImageFont.truetype(font_path, size=48)

    print(f"compositing {len(imgs)} images...")

    for i, l in enumerate(col_labels):
        label_image = make_label_image(l, col_font, w, h, margins=[10, 10, 10, 10])
        grid.paste(label_image, box=(margin + (1 + i) * (w + spacing), margin))

    for i, l in enumerate(row_labels):
        label_image = make_label_image(l, row_font, w, h, margins=[20, 10, 20, 10])
        grid.paste(label_image, box=(margin, margin + (1 + i) * (h + spacing)))

    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        grid.paste(img,
                   box=(margin + (1 + (i % num_cols)) * (w + spacing), margin + (1 + (i // num_cols)) * (h + spacing)))

    return grid


def chunk_list(list, batch_size):
    for i in range(0, len(list), batch_size):
        yield list[i:i + batch_size]


def load_model(repo_id_or_path, prefer_fp16: bool = True, local_files_only: bool=False):
    if os.path.isfile(repo_id_or_path):
        yaml_file = (os.path.splitext(repo_id_or_path)[0] + ".yaml")
        if not os.path.exists(yaml_file):
            yaml_file = None
        # this won't work on diffusers 0.15+, need to find a way to reliably support old + new
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import \
                load_pipeline_from_original_stable_diffusion_ckpt
        return load_pipeline_from_original_stable_diffusion_ckpt(repo_id_or_path, original_config_file=yaml_file)
    elif os.path.isdir(repo_id_or_path):
        return StableDiffusionPipeline.from_pretrained(repo_id_or_path)
    else:
        revision = None  # use default
        if prefer_fp16:
            refs = list_repo_refs(repo_id_or_path)
            fp16_ref = next((r for r in refs.branches if r.name == 'fp16'), None)
            if fp16_ref is not None:
                revision = 'fp16'
        return StableDiffusionPipeline.from_pretrained(repo_id_or_path, revision=revision,
                                                       local_files_only=local_files_only)

def make_scheduler(which: str, config):

    if which == 'ddim':
        return DDIMScheduler.from_config(config)
    elif which == "lms":
        return LMSDiscreteScheduler.from_config(config)
    elif which == "dpm++":
        return DPMSolverMultistepScheduler.from_config(config, algorithm_type="dpmsolver++", use_karras_sigmas=False)
    elif which == "k_dpm++":
        return DPMSolverMultistepScheduler.from_config(config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    elif which == "dpm++_sde":
        return DPMSolverMultistepScheduler.from_config(config, algorithm_type="sde-dpmsolver++",
                                                       use_karras_sigmas=False)
    elif which == "k_dpm++_sde":
        return DPMSolverMultistepScheduler.from_config(config, algorithm_type="sde-dpmsolver++",
                                                       use_karras_sigmas=True)
    elif which == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(config, use_karras_sigmas=False)
    elif which == "k_euler_a":
        return EulerAncestralDiscreteScheduler.from_config(config, use_karras_sigmas=True)
    elif which == 'pndm':
        return PNDMScheduler.from_config(config)
    elif which == 'ddpm':
        return DDPMScheduler.from_config(config)
    elif which == 'k_dpm2_a':
        return KDPM2AncestralDiscreteScheduler.from_config(config)
    else:
        raise ValueError(f"Unrecognized scheduler {which}")


def render_row(prompts: list[str],
               negative_prompts: Optional[list[str]],
               seeds: list[int],
               pipeline: StableDiffusionPipeline,
               scheduler_name: str="dpm++",
               device: str = None,
               batch_size=1,
               sample_w=512,
               sample_h=512,
               cfg=7.5,
               num_inference_steps=15, # for dpm++ solver, 15 is typically enough
               disable_nsfw_checker: bool = False,
               use_penultimate_clip_layer: bool = False
               ) -> list[Image]:
    pipeline.scheduler = make_scheduler(which=scheduler_name, config=pipeline.scheduler.config)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # noinspection PyTypeChecker
    pipeline = pipeline.to(device, torch.float16)
    if is_xformers_available():
        pipeline.enable_xformers_memory_efficient_attention()
    if disable_nsfw_checker:
        pipeline.safety_checker = disable_nsfw_safety_checker
    images = []
    negative_prompts = negative_prompts or [""] * len(prompts)
    batches = chunk_list(list(zip(prompts, negative_prompts, seeds)), batch_size)
    progress_bar = tqdm(list(batches))
    compel = Compel(tokenizer=pipeline.tokenizer,
                    text_encoder=pipeline.text_encoder,
                    use_penultimate_clip_layer=use_penultimate_clip_layer)
    for batch in progress_bar:
        batch_prompts, batch_negative_prompts, batch_seeds = zip(*batch)
        print(f" - {batch_prompts}")
        generator_device = 'cpu' if device == 'mps' else device
        manual_seed_generators = [torch.Generator(generator_device).manual_seed(seed) for seed in batch_seeds]
        pipeline_output: StableDiffusionPipelineOutput = pipeline(prompt=list(batch_prompts),
                                                                  negative_prompt=list(batch_negative_prompts),
                                                                  generator=manual_seed_generators,
                                                                  width=sample_w,
                                                                  height=sample_h,
                                                                  num_inference_steps=num_inference_steps,
                                                                  guidance_scale=cfg)
        images += pipeline_output.images

    return images


def merge_models(model_a_repo_id_or_path: str, model_b_repo_id_or_path: str, model_c_repo_id_or_path: Optional[str],
                 alpha=0.5,
                 algorithm: Optional[str] = None, unet_block_weights: Optional[list[float]] = None,
                 per_module_alphas: Optional[dict[str,float]] = None,
                 local_files_only: bool = False) \
        -> StableDiffusionPipeline:
    """
    Merge the two or three given models using the given alpha (for two models: 0.0=100% model a, 1.0=100% model b)
    """

    custom_pipeline = ("checkpoint_merger"
                       if unet_block_weights is None and per_module_alphas is None
                       else os.path.join(pathlib.Path(__file__).parent.resolve(), "checkpoint_merger_mbw.py")
                       )
    pipe = StableDiffusionPipeline.from_pretrained(model_a_repo_id_or_path, local_files_only=local_files_only,
                                                   #custom_pipeline="./checkpoint_merger.py")
                                                    custom_pipeline = custom_pipeline)

    if algorithm is None:
        algorithm = 'add_diff' if model_c_repo_id_or_path is not None else 'weighted_sum'
    if model_c_repo_id_or_path is not None:
        if algorithm != "add_diff":
            raise ValueError("Only add_diff is supported for 3-way merges")
    else:
        if algorithm == "add_diff":
            raise ValueError("add_diff is not supported for 2-way merges")

    force = True
    models = [model_a_repo_id_or_path, model_b_repo_id_or_path]
    if model_c_repo_id_or_path is not None:
        models.append(model_c_repo_id_or_path)
    merged_pipe = pipe.merge(models, local_files_only=local_files_only, interp=algorithm,
                             alpha=alpha, force=force,
                             block_weights=unet_block_weights,
                             module_override_alphas=per_module_alphas)
    del pipe

    return merged_pipe

def explode_merge_config(merge_config: dict) -> list[dict]:

    def get_merge_config_values(key, default):
        v = merge_config.get(key, None)
        if v is None:
            return default
        return v

    merge_alphas = get_merge_config_values('alphas', [0.5])
    merge_algorithms = get_merge_config_values('algorithms', ['weighted_sum'])
    unet_alphas = get_merge_config_values('unet_alphas', [None])
    text_encoder_alphas = get_merge_config_values('text_encoder_alphas', [None])
    unet_block_weights_strs = get_merge_config_values('block_weights', [])
    if len(unet_block_weights_strs) == 0:
        unet_block_weights = [None]
    else:
        unet_block_weights = [float(f)
                          for s in unet_block_weights_strs
                          for f in s.split(',')]

    merge_configs = []
    for a in merge_algorithms:
        for ma in merge_alphas:
            for ua in unet_alphas:
                for ta in text_encoder_alphas:
                    for bw in unet_block_weights:
                        merge_configs.append({
                            'alpha': ma,
                            'algorithm': a,
                            'unet_alpha': ua,
                            'text_encoder_alpha': ta,
                            'block_weights': bw,
                        })
    return merge_configs




def render_all(prompts: list[str], negative_prompts: Optional[list[str]], seeds: list[int], cfg: float,
               repo_ids_or_paths: list[str],
               scheduler: str,
               device: str,
               size: tuple[int, int],
               batch_size: int,
               inference_steps: int = 15,
               save_partial_filename: Optional[str] = None,
               local_files_only: bool = False,
               merge_config: Optional[dict] = None,
               disable_nsfw_checker: bool = False,
               use_penultimate_clip_layer: Optional[bool] = False, # autodetect if None
               ) -> Image:
    all_images = []
    print(f"{len(prompts)} prompts")

    row_labels = []

    def save_partial_if_requested():
        if save_partial_filename is not None:
            num_rows = int(len(all_images) / len(prompts))
            grid_image = make_image_grid(all_images, num_rows=num_rows, num_cols=len(prompts),
                                         row_labels=row_labels[:num_rows], col_labels=prompts)
            grid_image.save(save_partial_filename)

    if merge_config is not None:
        if len(repo_ids_or_paths) != 2 and len(repo_ids_or_paths) != 3:
            raise ValueError("Must specify either 2 or 3 models when using merge_alphas")

        merge_configs = explode_merge_config(merge_config)

        for merge_index, config in enumerate(tqdm(merge_configs)):
            unet_block_weights = config['block_weights']
            if unet_block_weights is not None and len(unet_block_weights) != 25:
                raise ValueError(f"Wrong number of unet block weights. There must be 25, eg \"0.0,0.04,0.08,0.12,0.16,"
                                 f"0.2,0.24,0.28,0.32,0.36,0.4,0.44,0.48,0.52,0.56,0.6,0.64,0.68,0.72,0.76,0.8,0.84,0.88,"
                                 f"0.92,0.96\". You have specified \"{unet_block_weights}\", which is not valid.")

            merge_algorithm = config['algorithm']
            per_module_alphas = {
                'unet': config['unet_alpha'],
                'text_encoder': config['text_encoder_alpha']
            }
            alpha = config['alpha']
            this_model_label = f"{merge_algorithm} merge of {repo_ids_or_paths} with alpha {alpha}, " \
                               f"per_module_alphas {per_module_alphas}, " \
                               f"unet_block_weights {unet_block_weights}"
            row_labels.append(this_model_label)
            print(this_model_label)
            model_c = repo_ids_or_paths[2] if len(repo_ids_or_paths) == 3 else None
            pipeline = merge_models(repo_ids_or_paths[0], repo_ids_or_paths[1], model_c, alpha=alpha,
                                    algorithm=merge_algorithm, unet_block_weights=unet_block_weights,
                                    per_module_alphas=per_module_alphas,
                                    local_files_only=local_files_only)
            row_images = render_row(prompts,
                                    negative_prompts=negative_prompts,
                                    seeds=seeds,
                                    pipeline=pipeline,
                                    scheduler_name=scheduler,
                                    device=device,
                                    batch_size=batch_size,
                                    cfg=cfg,
                                    num_inference_steps=inference_steps,
                                    sample_w=size[0], sample_h=size[1],
                                    disable_nsfw_checker=disable_nsfw_checker,
                                    use_penultimate_clip_layer=use_penultimate_clip_layer)
            all_images += row_images
            save_partial_if_requested()

            save_merge_path_prefix = merge_config.get('save_merge_path_prefix', None)
            if save_merge_path_prefix is not None:
                save_half = merge_config.get('save_merge_half', True)
                if save_half:
                    pipeline.to(pipeline.device, torch.float16)
                save_merge_path = f"{save_merge_path_prefix}_{merge_index:02d}"
                pipeline.save_pretrained(save_merge_path)

            del pipeline

        grid_image = make_image_grid(all_images, len(merge_configs), len(prompts), row_labels, prompts)
    else:
        row_labels = repo_ids_or_paths
        for repo_id_or_path in tqdm(repo_ids_or_paths):
            print(f"model {repo_id_or_path}:")
            pipeline = load_model(repo_id_or_path, local_files_only=local_files_only)
            row_images = render_row(prompts,
                                    negative_prompts=negative_prompts,
                                    seeds=seeds,
                                    pipeline=pipeline,
                                    scheduler_name=scheduler,
                                    device=device,
                                    batch_size=batch_size,
                                    cfg=cfg,
                                    num_inference_steps=inference_steps,
                                    sample_w=size[0], sample_h=size[1],
                                    disable_nsfw_checker=disable_nsfw_checker,
                                    use_penultimate_clip_layer=use_penultimate_clip_layer)
            all_images += row_images
            save_partial_if_requested()
            del pipeline
        grid_image = make_image_grid(all_images, len(repo_ids_or_paths), len(prompts), row_labels, prompts)

    return grid_image


def main():
    parser = argparse.ArgumentParser(
        prog="grate",
        description="Generates a grid of images by running a set of prompts through different Stable Diffusion models.\n"
                    "\n"
                    "Optionally, merge models together: if one or more of the --merge_ options is passed, grate will produce "
                    "multiple merged models using all possible combinations of the passed values, and render each on its "
                    "own row in the output image. For example, \n"
                    "    grate --merge_alphas 0.333 0.667 --merge_algorithm weighted_sum sigmoid\n"
                    "will produce an output image with 4 rows, representing a weighted_sum merge with alpha 0.333, a "
                    "weighted_sum merge with alpha 0.667, a sigmoid merge with alpha 0.333, and a sigmoid merge with "
                    "alpha 0.667, respectively. \n"
                    "\n"
                    "When merging, you must specify either 2 or 3 values for --repo_ids_or_paths ."
    )

    parser.add_argument("--prompts",
                        nargs='+',
                        required=True,
                        help=(
                                    "EITHER: a path to a JSON file containing prompts with optional seeds and negative prompts, eg [{'prompt': 'a fish', 'negative_prompt': 'distorted', 'seed': 123}, ...]. \n\n" +
                                    "OR: multiple strings enclosed in \"\" and separated by spaces. eg --prompts \"a cat\" \"a dog\" \"a fish\"")
                        )
    parser.add_argument("--repo_ids_or_paths",
                        nargs='+',
                        required=True,
                        help="repository ids or paths to models in diffusers or ckpt format, as strings enclosed in \"\" "
                             "and separated by spaces. eg "
                             "--repo_ids_or_paths \"stablityai/stable-diffusion-2-1\" \"../models/v1-5-pruned-emaonly.ckpt\"")
    parser.add_argument("--output_path",
                        required=True,
                        help="Where to save the resulting image. Also saves partially-rendered images to this location as each row finishes rendering.")
    parser.add_argument("--device",
                        required=False,
                        default=None,
                        help="(Optional) Device to use, eg 'cuda', 'mps', 'cpu'. if omitted, will try to pick the best device.")
    parser.add_argument("--batch_size",
                        required=False,
                        type=int,
                        default=1,
                        help="(Optional, default=1) Batch size.")
    parser.add_argument("--width",
                        required=False,
                        default=512,
                        type=int,
                        help="(Optional, default=512) Individual image width")
    parser.add_argument("--height",
                        required=False,
                        type=int,
                        default=512,
                        help="(Optional, defatul=512) Individual image height")
    parser.add_argument("--negative_prompts",
                        required=False,
                        type=str,
                        nargs="+",
                        help="(Optional) Negative prompts. Specify either one string to share for all `--prompts`, or as many strings as there are `--prompts`.")
    parser.add_argument("--seeds",
                        required=False,
                        type=int,
                        nargs="+",
                        help="(Optional) Seeds. Specify either one seed to share for all `--prompts`, or as many seeds as there are `--prompts`.")
    parser.add_argument("--scheduler",
                        choices=['ddim', "lms", "dpm++", "k_dpm++", "dpm++_sde", "k_dpm++_sde",
                                 "euler_a", "k_euler_a", 'pndm', 'ddpm', 'k_dpm2_a'],
                        default="dpm++",
                        type=str,
                        help="(Optional, default=dpm++) Scheduler to use.")
    parser.add_argument("--cfg",
                        required=False,
                        type=float,
                        default=7.5,
                        help="(Optional, default=7.5) CFG scale.")
    parser.add_argument("--steps",
                        required=False,
                        type=int,
                        default=15,
                        help="(Optional, default=15) How many inference steps to run")
    parser.add_argument("--disable_nsfw_checker",
                        required=False,
                        action="store_true",
                        help="(Optional)")
    parser.add_argument("--local_files_only",
                        required=False,
                        action='store_true',
                        help="(Optional) Use only local data (do not attempt to download or update models)")
    parser.add_argument("--merge_alpha",
                        required=False,
                        type=float,
                        nargs="+",
                        help="(Optional) If set, --repo_ids_or_paths must specify either 2 or 3 models, which will be merged using the given alpha and used in place of multiple models.")
    parser.add_argument("--merge_algorithm",
                        required=False,
                        type=str,
                        default=None,
                        nargs="+",
                        help="(Optional) If doing merges, the algorithm to use - one of 'weighted_sum', 'sigmoid', 'inv_sigmoid', or 'add_diff'. 'add_diff' only works for 3-way merge. ")
    parser.add_argument("--merge_unet_block_weights",
                        required=False,
                        type=str,
                        nargs="+",
                        help="(Optional) 25 comma-separated floats specified as strings, eg \"0.0, 0.0, 0.0, (... 22 more floats)\", to merge each part of the UNet using a different weight ('block-weighted merging').")
    parser.add_argument("--merge_unet_alpha",
                        required=False,
                        type=float,
                        nargs="+",
                        help="(Optional) Override the merge alpha with a unet-specific alpha.")
    parser.add_argument("--merge_text_encoder_alpha",
                        required=False,
                        type=float,
                        nargs="+",
                        help="(Optional) Override the merge alpha with a text-encoder-specific alpha.")
    parser.add_argument("--save_merge_path_prefix",
                        required=False,
                        type=str,
                        help="(Optional) If doing a merge, save all merge combinations using this path as a prefix.")
    parser.add_argument("--save_merge_float32",
                        action="store_true",
                        help="(Optional) If saving merges, save with float32 precision (default is float16).")
    parser.add_argument("--use_penultimate_clip_layer",
                        action="store_true",
                        help="(Optional) Use the outputs from penultimate (second to last) CLIP hidden layer. On detected SD2.x models this defaults on, otherwise it defaults off.")
    args = parser.parse_args()


    def use_arg_list_or_expand_or_default(arg: list, required_count: int, default: list|None) -> list:
        if arg is None:
            return default
        elif len(arg) == 1:
            # expand to required count
            return arg * required_count
        else:
            return arg


    if len(args.prompts) == 1 and os.path.isfile(args.prompts[0]):
        with open(args.prompts[0], 'rt') as f:
            prompts_json = json.load(f)
            prompts = [p.get('prompt', '') for p in prompts_json]
            negative_prompts = [p.get('negative_prompt', '') for p in prompts_json]
            seeds = [int(p.get('seed', 1 + i)) for i, p in enumerate(prompts_json)]
            print(f"loaded {len(prompts)} prompts from {args.prompts[0]}")
            print({'prompts': prompts, 'negative_prompts': negative_prompts, 'seeds': seeds})
    else:

        prompts = args.prompts
        negative_prompts = use_arg_list_or_expand_or_default(args.negative_prompts, len(prompts), [''] * len(prompts))
        seeds = use_arg_list_or_expand_or_default(args.seeds, len(prompts), [1 + i for i in range(len(prompts))])

    print(f"args.merge_unet_alpha: {args.merge_unet_alpha}")
    merge_config = {
        'alphas': args.merge_alpha,
        'algorithms': args.merge_algorithm,
        'unet_alphas': args.merge_unet_alpha,
        'text_encoder_alphas': args.merge_text_encoder_alpha,
        'block_weights': args.merge_unet_block_weights,
    }
    if all(v is None for v in merge_config.values()):
        merge_config = None
    else:
        merge_config['save_merge_path_prefix'] = args.save_merge_path_prefix
        merge_config['save_merge_half'] = not args.save_merge_float32


    render_all(prompts=prompts,
               negative_prompts=negative_prompts,
               scheduler=args.scheduler,
               seeds=seeds,
               repo_ids_or_paths=args.repo_ids_or_paths,
               merge_config=merge_config,
               device=args.device,
               size=(args.width, args.height),
               batch_size=args.batch_size,
               cfg=args.cfg,
               inference_steps=args.steps,
               local_files_only=args.local_files_only,
               save_partial_filename=args.output_path,
               disable_nsfw_checker=args.disable_nsfw_checker,
               use_penultimate_clip_layer=args.use_penultimate_clip_layer)
    print(f"grate saved to {args.output_path}")

if __name__ == '__main__':
    main()
