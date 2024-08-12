# Flux.1 with 4 bit Quantization

<br>

<div align = center>

[![Badge Model]][Model]   
[![Badge Colab]][Colab]

<br>
<br>

<!---------------------------------------------------------------------------->

[Model]: https://huggingface.co/HighCWu/FLUX.1-dev-4bit
[Colab]: https://colab.research.google.com/github/HighCWu/flux-4bit/blob/main/colab_t4.ipynb


<!---------------------------------[ Badges ]---------------------------------->

[Badge Model]: https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg
[Badge Colab]: https://colab.research.google.com/assets/colab-badge.svg

<!---------------------------------------------------------------------------->


I want to train flux's LoRA using the `diffusers` library on my 16GB GPU, but it's difficult to train with flux-dev-fp8, so I want to use 4-bit weights to save VRAM.

I found that flux's text_encoder_2 (t5xxl) quantized with bnb_nf4 is not as good as hqq_4bit, and flux's transformer quantized with hqq_4bit is not as good as bnb_nf4, so I used different quantization methods for the two models.

In inference mode, using cpu_offload takes up 8.5GB of VRAM, and when it is not turned on, it takes up 11GB of VRAM.

If you want to use less VRAM during training, you can consider storing the results of text_encoder_2 as a dataset first.

***Note*** I used some patch code to make the `diffusers` model load the quantized weights properly.

# How to use

1. clone the repo:
    ```sh
    git clone https://github.com/HighCWu/flux-4bit
    cd flux-4bit
    ```

2. install requirements:
    ```sh
    pip install -r requirements.txt
    ```

3. run in python:
    ```py
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

    ```
