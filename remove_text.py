from openai import OpenAI
import base64
import argparse
import os
from PIL import Image, ImageDraw
import pytesseract

DEFAULT_PROMPT = "Remove all text from this image and seamlessly reconstruct the background behind the removed text; preserve every other element exactly."

def create_mask_for_text_areas(image_path):
    """Automatically detect and mask text regions using OCR."""
    # Load image in RGB for OCR
    img = Image.open(image_path)
    if img.mode != 'RGB':  parallel_svg_pipeline.py
        img = img.convert('RGB')
    # Run OCR to get bounding boxes
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    # Create a transparent mask
    mask = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)
    # Draw rectangles over each detected text region
    for i, text in enumerate(data['text']):
        if text.strip():
            x, y, w, h = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255, 255))
    return mask

def main():
    parser = argparse.ArgumentParser(description="Edit images with OpenAI's GPT-Image API")
    parser.add_argument("--api_key", required=True, help="Your OpenAI API key")
    parser.add_argument("--image_path", required=True, help="Path to the input image file")
    parser.add_argument("--mask_path", help="Path to an existing mask image file (optional)", default=None)
    parser.add_argument("--prompt", help="Image edit prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--n", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--size", default="1024x1024", help="Generated image size")
    parser.add_argument("--output", help="Output filename (defaults to edited_<image_basename>)")
    args = parser.parse_args()

    # Prepare mask: use provided or auto-generate and save permanently
    if args.mask_path:
        mask_path = args.mask_path
    else:
        mask = create_mask_for_text_areas(args.image_path)
        # Save mask beside original image
        base = os.path.basename(args.image_path)
        mask_filename = f"mask_{base}"
        mask_dir = os.path.dirname(args.image_path) or "."
        mask_path = os.path.join(mask_dir, mask_filename)
        mask.save(mask_path)
        print(f"Mask image generated at: {mask_path}")

    client = OpenAI(api_key=args.api_key)
    print("Sending request to OpenAI API...")
    result = client.images.edit(
        model="gpt-image-1",
        image=open(args.image_path, "rb"),
        mask=open(mask_path, "rb"),
        prompt=args.prompt,
        n=args.n,
        size=args.size,
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    output_file = args.output or f"edited_{os.path.basename(args.image_path)}"
    with open(output_file, "wb") as f:
        f.write(image_bytes)
    print(f"Success! Edited image saved as: {output_file}")

    # Mask file is saved permanently for inspection

if __name__ == "__main__":
    main() 