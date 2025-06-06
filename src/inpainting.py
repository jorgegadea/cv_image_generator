import torch
from diffusers import StableDiffusionXLInpaintPipeline, IPAdapterXL
from PIL import Image


def run_inpainting(image, mask, prompt, ip_adapter_image=None):
    """
    Run inpainting on the given image using the specified mask and prompt.

    Args:
        image (PIL.Image.Image): The input image to be inpainted.
        mask (PIL.Image.Image): The mask indicating the areas to be inpainted.
        prompt (str): The text prompt guiding the inpainting process.
        ip_adapter_image (PIL.Image.Image, optional): Reference image for IP-Adapter (face preservation).
    Returns:
        PIL.Image.Image: The inpainted image.
    """
    # Load the inpainting pipeline
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16"  # Use fp16 for better performance on GPUs
    ).to("cuda")

    # Integrate IP-Adapter if provided
    if ip_adapter_image is not None:
        ip_adapter = IPAdapterXL.from_pretrained("h94/IP-Adapter", torch_dtype=torch.float16)
        pipe.load_ip_adapter(ip_adapter)
    
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            ip_adapter_image=ip_adapter_image
        ).images[0]
    else:
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask
        ).images[0]

    return result