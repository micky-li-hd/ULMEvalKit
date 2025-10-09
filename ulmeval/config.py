from ulmeval.ulm import *
from ulmeval.api import *
from functools import partial
import os


api_models = {
    "gpt-image-1": partial(
        GPTIMAGE1,
        model="gpt-image-1",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GeminiFlashPreviewImageGeneration2-0": partial(
        GeminiFlashGeneration, model="gemini-2.0-flash-preview-image-generation", temperature=0, retry=10
    ),
    "Imagen3": partial(
        Imagen, model="imagen-3.0-generate-002", retry=10
    ),
    "Imagen4": partial(
        Imagen, model="imagen-4.0-generate-preview-06-06", retry=10
    ),
    "Imagen4-Ultra": partial(
        Imagen, model="imagen-4.0-ultra-generate-preview-06-06", retry=10
    ),
}


bagel_series = {
    'BAGEL-7B-MoT': partial(
        Bagel,
        model_path='ByteDance-Seed/BAGEL-7B-MoT'
    )
}

janus_series = {
    "Janus-1.3B": partial(JanusGeneration, model_path="deepseek-ai/Janus-1.3B"),
    "Janus-Pro-1B": partial(JanusPro, model_path="deepseek-ai/Janus-Pro-1B"),
    "Janus-Pro-7B": partial(JanusPro, model_path="deepseek-ai/Janus-Pro-7B"),
    "JanusFlow-1.3B": partial(JanusFlow, model_path="deepseek-ai/JanusFlow-1.3B"),
}

omnigen_series = {
    "OmniGen2": partial(OmniGen2, model_path="OmniGen2/OmniGen2"),
}

flux_series = {
    "FLUX.1-schnell": partial(
        Flux, model_path="black-forest-labs/FLUX.1-schnell",
    ),
    "FLUX.1-dev": partial(
        Flux, model_path="black-forest-labs/FLUX.1-dev",
    ),
    "FLUX.1-Kontext-dev": partial(
        FluxKontext, model_path="black-forest-labs/FLUX.1-Kontext-dev",
    )
}

show_o_series = {
    'Show-o': partial(
        Showo, mode='showo_demo', model_path="showlab/show-o"
    ),
    'Show-o-512x512': partial(
        Showo, mode='showo_demo_512x512', model_path="showlab/show-o-512x512"
    ),
    'Show-o-w-clip-vit-512x512': partial(
        Showo, mode='showo_demo_w_clip_vit_512x512', model_path="showlab/show-o-w-clip-vit-512x512"
    ),
    'Show-o-w-clip-vit': partial(
        Showo, mode='showo_demo_w_clip_vit', model_path="showlab/show-o-w-clip-vit"
    )
}

supported_ULM = {}

model_groups = [
    api_models,
    bagel_series,
    janus_series,
    show_o_series,
    omnigen_series,
    flux_series,
]

for grp in model_groups:
    supported_ULM.update(grp)
