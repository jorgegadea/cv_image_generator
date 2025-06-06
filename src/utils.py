from PIL import Image
from transformers import DetrImageProcessor, DetrForSegmentation
import torch
import numpy as np


def load_image(image_path):
    """
    Load an image from the specified path and convert it to RGB format.
    This function is useful for ensuring that the image is in a consistent format
    for further processing, such as inpainting or applying transformations.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        PIL.Image.Image: The loaded image in RGB format.
    """
    image = Image.open(image_path).convert("RGB")
    return image

def save_image(image, output_path):
    """
    Save an image to the specified path.
    
    Args:
        image (PIL.Image.Image): The image to save.
        output_path (str): The path where the image will be saved.
    """
    image.save(output_path, format='PNG')


def generate_mask_person(image):
    """
    Genera una máscara de persona/ropa usando un modelo de segmentación de Hugging Face.
    Args:
        image (PIL.Image.Image): Imagen de entrada.
    Returns:
        PIL.Image.Image: Máscara binaria (blanco: zona a modificar, negro: conservar).
    """


    # Cargar modelo y procesador
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

    # Preprocesar imagen
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    processed_sizes = [(inputs['pixel_values'].shape[-2], inputs['pixel_values'].shape[-1])]
    result = processor.post_process_panoptic_segmentation(outputs, processed_sizes=processed_sizes, threshold=0.85)[0]

    # Extraer máscara de persona (category_id == 0 para persona en COCO)
    mask = np.zeros(result['segmentation'].shape, dtype=np.uint8)
    for seg in result['segments_info']:
        if seg['category_id'] == 0:  # 0 es 'person' en COCO
            mask[result['segmentation'] == seg['id']] = 255

    mask_img = Image.fromarray(mask).convert("L")
    return mask_img


def generate_mask_clothes(image):
    """
    Genera una máscara de solo la ropa usando un modelo de segmentación de ropa (ClothSeg).
    Args:
        image (PIL.Image.Image): Imagen de entrada.
    Returns:
        PIL.Image.Image: Máscara binaria (blanco: ropa, negro: resto).
    """
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
    import torch

    # Cargar modelo y procesador de ClothSeg
    processor = AutoImageProcessor.from_pretrained("akhaliq/cloth-segmentation")
    model = UperNetForSemanticSegmentation.from_pretrained("akhaliq/cloth-segmentation")

    # Preprocesar imagen
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # [1, num_classes, H, W]
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (alto, ancho)
        mode="bilinear",
        align_corners=False,
    )
    pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # En ClothSeg, la clase 1 suele ser "upper clothes" y la 2 "lower clothes"
    mask = np.zeros_like(pred, dtype=np.uint8)
    mask[(pred == 1) | (pred == 2)] = 255  # Puedes ajustar según el modelo

    mask_img = Image.fromarray(mask).convert("L")
    return mask_img
