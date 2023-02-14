# adapted from https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb#scrollTo=36026528
import json
import os
import argparse
from typing import Optional

import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import load_pipeline_from_original_stable_diffusion_ckpt
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw, ImageFont
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import is_xformers_available
from tqdm import tqdm

from huggingface_hub import list_repo_refs

# get_wrapped_text adapted from https://stackoverflow.com/a/67203353
def get_wrapped_text(text: str, font: ImageFont.ImageFont,
                     line_length: int):
    lines = []
    words = text.split()
    current_line = None
    while len(words) > 0:
        next_word = words.pop(0)
        current_line_with_next_word = next_word if current_line is None else f'{current_line} {next_word}'
        if font.getlength(current_line_with_next_word) <= line_length:
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
                while len(truncated_word) > 0 and font.getlength(truncated_word) > line_length:
                    truncated_word = truncated_word[:-1]
                if len(truncated_word) > 0:
                    lines.append(truncated_word)
                    words.insert(0, next_word[len(truncated_word):])
                else:
                    # give up
                    lines.append(next_word)
    if current_line is not None:
        lines.append(current_line)
    return lines

def make_label_image(label: str, font: ImageFont, width: int, height: int, margins=None):
    if margins is None:
        margins = [10, 10, 10, 10]
    line_spacing = 5  # pixels between lines
    wrapped_text = '\n'.join(get_wrapped_text(label, font, line_length=width - (margins[0] + margins[2])))
    label_image = Image.new('RGB', size=(width, height), color=(247, 247, 247))
    draw = ImageDraw.Draw(label_image)
    text_bbox = draw.multiline_textbbox((width / 2, 0),
                                        text=wrapped_text,
                                        font=font,
                                        anchor='ma',
                                        spacing=line_spacing,
                                        )  # anchor = horizontal middle, ascender
    text_bbox_height = text_bbox[3] - text_bbox[1]
    # print("got text_bbox ", text_bbox)
    y_offset = (height - text_bbox_height) / 2
    x_offset = (width - margins[2] - margins[0]) // 2
    draw.multiline_text((margins[0] + x_offset, margins[1] + y_offset),
                        text=wrapped_text, font=font, anchor='ma', fill=(64, 48, 32))
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
        grid.paste(img, box=(margin + (1 + (i % num_cols)) * (w + spacing), margin + (1 + (i // num_cols)) * (h + spacing)))

    return grid


def chunk_list(list, batch_size):
    for i in range(0, len(list), batch_size):
        yield list[i:i+batch_size]

def load_model(repo_id_or_path, prefer_fp16: bool=True):
    if os.path.isfile(repo_id_or_path):
        return load_pipeline_from_original_stable_diffusion_ckpt(repo_id_or_path)
    elif os.path.isdir(repo_id_or_path):
        return StableDiffusionPipeline.from_pretrained(repo_id_or_path)
    else:
        revision = None  # use default
        if prefer_fp16:
            refs = list_repo_refs(repo_id_or_path)
            fp16_ref = next((r for r in refs.branches if r.name == 'fp16'), None)
            if fp16_ref is not None:
                revision = 'fp16'
        return StableDiffusionPipeline.from_pretrained(repo_id_or_path, revision=revision)


def render_row(prompts: list[str],
               negative_prompts: Optional[list[str]],
               seeds: list[int],
               pipeline: StableDiffusionPipeline,
               device: str=None,
               batch_size=1,
               sample_w = 512,
               sample_h = 512,
               cfg = 7.5,
               num_inference_steps = 15 # ddpm++ solver: 15 is typically enough
               ) -> list[Image]:

    # ddpm++
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config,
                                                                 algorithm_type="dpmsolver++")
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'cuda':
        # noinspection PyTypeChecker
        pipeline = pipeline.to(torch.float16)
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()
    pipeline = pipeline.to(device)
    images = []
    negative_prompts = negative_prompts or [""] * len(prompts)
    batches = chunk_list(list(zip(prompts, negative_prompts, seeds)), batch_size)
    progress_bar = tqdm(list(batches))
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

def merge_models(model_a_repo_id_or_path: str, model_b_repo_id_or_path: str, model_c_repo_id_or_path: Optional[str], alpha=0.5) -> StableDiffusionPipeline:
    """
    Merge the two or three given models using the given alpha (for two models: 0.0=100% model a, 1.0=100% model b)
    """
    pipe = StableDiffusionPipeline.from_pretrained(model_a_repo_id_or_path, custom_pipeline="checkpoint_merger")

    interp = "weighted_sum" if model_c_repo_id_or_path is None else "add_diff"
    force = True
    models = [model_a_repo_id_or_path, model_b_repo_id_or_path]
    if model_c_repo_id_or_path is not None:
        models.append(model_c_repo_id_or_path)
    merged_pipe = pipe.merge(models, interp=interp, alpha=alpha, force=force)
    del pipe

    return merged_pipe

def render_all(prompts: list[str], negative_prompts: Optional[list[str]], seeds: list[int],
               repo_ids_or_paths: list[str], merge_alphas: Optional[list[float]],
               device: str,
               size: tuple[int,int], batch_size: int, save_partial_prefix: str=None) -> Image:
    all_images = []
    print(f"{len(prompts)} prompts")

    def save_partial_if_requested():
        if save_partial_prefix is not None:
            num_rows = int(len(all_images) / len(prompts))
            grid_image = make_image_grid(all_images, num_rows=num_rows, num_cols=len(prompts),
                                         row_labels=repo_ids_or_paths[:num_rows], col_labels=prompts)
            grid_image.save(f"{save_partial_prefix}-row{num_rows:02}.jpg")


    if merge_alphas is not None:
        if len(repo_ids_or_paths) != 2:
            raise ValueError("Must specify exactly 2 models when using merge_alphas")
        for alpha in tqdm(merge_alphas):
            print(f"merged model from {repo_ids_or_paths} with alpha={alpha}:")
            model_c = None if len(repo_ids_or_paths) < 3 else repo_ids_or_paths[2]
            pipeline = merge_models(repo_ids_or_paths[0], repo_ids_or_paths[1], model_c, alpha=alpha)
            row_images = render_row(prompts,
                                negative_prompts=negative_prompts,
                                seeds=seeds,
                                pipeline=pipeline,
                                device=device,
                                batch_size=batch_size,
                                sample_w=size[0], sample_h=size[1])
            all_images += row_images
            save_partial_if_requested()
    else:
        for repo_id_or_path in tqdm(repo_ids_or_paths):
            print(f"model {repo_id_or_path}:")
            pipeline = load_model(repo_id_or_path)
            row_images = render_row(prompts,
                                    negative_prompts=negative_prompts,
                                    seeds=seeds,
                                    pipeline=pipeline,
                                    device=device,
                                    batch_size=batch_size,
                                    sample_w=size[0], sample_h=size[1])
            all_images += row_images
            save_partial_if_requested()

    grid_image = make_image_grid(all_images, len(repo_ids_or_paths), len(prompts), repo_ids_or_paths, prompts)
    return grid_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="grate",
        description="Generates a grid of images by running a set of prompts through different Stable Diffusion models.",
    )

    parser.add_argument("--prompts",
                        nargs='+',
                        required=True,
                        help=("EITHER: a path to a JSON file containing prompt and negative prompt pairs eg [{'prompt': 'a fish', 'negative_prompt': 'distorted', 'seed': 123}, ...]. \n\n" +
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
                        help="where to save the resulting image")
    parser.add_argument("--device",
                        required=False,
                        default=None,
                        help="device to use, eg 'cuda', 'mps', 'cpu'. if omitted, will try to pick the best device.")
    parser.add_argument("--batch_size",
                        required=False,
                        type=int,
                        default=1,
                        help="batch size, default 1")
    parser.add_argument("--width",
                        required=False,
                        default=512,
                        type=int,
                        help="individual image width")
    parser.add_argument("--height",
                        required=False,
                        type=int,
                        default=512,
                        help="individual image height")
    parser.add_argument("--merge_alphas",
                        required=False,
                        type=float,
                        nargs="+",
                        help="(Optional) If set, --repo_ids_or_paths must specify exactly 2 models, which will be merged using the given alphas and used in place of multiple models.")
    parser.add_argument("--negative_prompts",
                        required=False,
                        type=str,
                        nargs="+",
                        help="(Optional) Negative prompts. Must be either 1 string to use for all prompts, or one string for each prompt passed to --prompts.")
    parser.add_argument("--seeds",
                        required=False,
                        type=int,
                        nargs="+",
                        help="(Optional) Seeds. Must be either 1 int to use for all prompts, or specify one seed per prompts.")
    parser.add_argument("--save_partial_prefix",
                        required=False,
                        type=str,
                        default=None,
                        help="(Optional) If set, save progress images using the given prefix. eg 'tmp/grate-partial' will save images to '/tmp/grate-partial-row1.jpg' etc.")
    args = parser.parse_args()

    if len(args.prompts) == 1 and os.path.isfile(args.prompts[0]):
        with open(args.prompts[0], 'rt') as f:
            prompts_json = json.load(f)
            prompts = [p.get('prompt', '') for p in prompts_json]
            negative_prompts = [p.get('negative_prompt', '') for p in prompts_json]
            seeds = [int(p.get('seed', 1 + i)) for i, p in enumerate(prompts_json)]
            print(f"loaded {len(prompts)} prompts from {args.prompts[0]}")
            print({'prompts': prompts, 'negative_prompts': negative_prompts, 'seeds': seeds})
    else:
        def use_arg_list_or_expand_or_default(arg: list, required_count: int, default: list) -> list:
            if arg is None:
                return default
            elif len(arg) == 1:
                # expand to required count
                return arg * required_count
            else:
                return arg

        prompts = args.prompts
        negative_prompts = use_arg_list_or_expand_or_default(args.negative_prompts, len(prompts), [''] * len(prompts))
        seeds = use_arg_list_or_expand_or_default(args.seeds, len(prompts), [1 + i for i in range(len(prompts))])


    grid: Image = render_all(prompts=prompts,
                             negative_prompts=negative_prompts,
                             seeds=seeds,
                             repo_ids_or_paths=args.repo_ids_or_paths,
                             merge_alphas=args.merge_alphas,
                             device=args.device,
                             size=(args.width,args.height),
                             batch_size=args.batch_size,
                             save_partial_prefix=args.save_partial_prefix)
    print(f"saving to {args.output_path}...")
    grid.save(args.output_path)
    print("done.")
