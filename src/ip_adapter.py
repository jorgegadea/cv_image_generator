from PIL import Image

def apply_ip_adapter(image, reference_face_path=None):
    """
    Integra IP-Adapter para preservar el rostro usando una imagen de referencia.
    Args:
        image (PIL.Image.Image): Imagen de entrada.
        reference_face_path (str): Ruta a la imagen de referencia facial (opcional).
    Returns:
        dict: Diccionario con los argumentos necesarios para el pipeline de inpainting.
    """
    # Ejemplo de integración (requiere diffusers >=0.24 y el modelo IP-Adapter descargado)
    # Aquí solo preparamos los argumentos para el pipeline, la integración real se hace en el pipeline de inpainting.
    # Si tienes una imagen de referencia facial, la cargas aquí:
    reference_face = None
    if reference_face_path:
        reference_face = Image.open(reference_face_path).convert("RGB")
    # Devuelve la imagen original y la referencia facial para usarlas en el pipeline
    return {
        "image": image,
        "ip_adapter_image": reference_face
    }