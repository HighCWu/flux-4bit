import torch

from model import T5EncoderModel, FluxTransformer2DModel
from diffusers import FluxPipeline


text_encoder_2: T5EncoderModel = T5EncoderModel.from_pretrained(
    "HighCWu/FLUX.1-dev-4bit",
    subfolder="text_encoder_2",
    torch_dtype=torch.bfloat16,
    # hqq_4bit_compute_dtype=torch.float32,
)

transformer: FluxTransformer2DModel = FluxTransformer2DModel.from_pretrained(
    "HighCWu/FLUX.1-dev-4bit",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

pipe: FluxPipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder_2=text_encoder_2,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload() # with cpu offload, it cost 8.5GB vram
# pipe.remove_all_hooks()
# pipe = pipe.to('cuda') # without cpu offload, it cost 11GB vram

prompt = "realistic, best quality, extremely detailed, ray tracing, photorealistic, A blue cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    output_type="pil",
    num_inference_steps=16,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.show()
