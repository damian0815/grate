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
        "nitrosocke/Arcane-Diffusion" 
    --output_path ./grate-demo.png
```

