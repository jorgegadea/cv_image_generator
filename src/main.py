import argparse
from utils import load_image, save_image, generate_mask_person, generate_mask_clothes
from inpainting import run_inpainting
from ip_adapter import apply_ip_adapter


def main():
    parser = argparse.ArgumentParser(description=" Generate a attaire image using a normal image.")
    parser.add_argument('--input', required=True, help='Path to the input image file.')
    parser.add_argument('--mask', required=True, help='Path to the mask image file.')
    parser.add_argument('--output', required=True, help='Path to save the output image file')
    parser.add_argument('--prompt', required=True, help='Prompt for the image generation')
    parser.add_argument('--mask_type', required=False, default='person', choices=['person', 'clothes'], help='Type of mask to generate (default: person)')
    parser.add_argument('--reference_face', required=False, help='Path to the reference face image for IP-Adapter (optional)')
    args = parser.parse_args()

    image = load_image(args.input)
    if args.mask:
        mask = load_image(args.mask)
    else:
        print("No mask provided, generating mask from image.")
        if args.mask_type == 'person':
            mask = generate_mask_person(image)
        else:
            mask = generate_mask_clothes(image)
        save_image(mask, "generated_mask.png")

    ip_adapter_args = apply_ip_adapter(image, args.reference_face)
    ip_adapter_image = ip_adapter_args.get("ip_adapter_image", None)

    result = run_inpainting(image, mask, args.prompt, ip_adapter_image=ip_adapter_image)
    save_image(result, args.output)

if __name__ == "__main__":
    main()