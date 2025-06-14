#!/usr/bin/env python3
import os
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from .env
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def create_mask(input_image_path):
    """Create an alpha mask from white text in the input image"""
    print(f"Creating mask from {input_image_path}...")
    
    # Load and convert to grayscale
    img = Image.open(input_image_path).convert("RGB")
    gray = img.convert("L")
    
    # Threshold to create binary mask - higher threshold for white text
    thr = 240  # Adjusted for white text
    mask_arr = np.array(gray)
    mask_binary = (mask_arr > thr).astype(np.uint8) * 255
    mask = Image.fromarray(mask_binary, mode="L")
    
    # Convert to RGBA with alpha channel
    mask_rgba = mask.convert("RGBA")
    mask_rgba.putalpha(mask)
    
    # Save mask
    timestamp = os.path.splitext(os.path.basename(input_image_path))[0]
    mask_path = f"mask_{timestamp}.png"
    mask_rgba.save(mask_path)
    print(f"✓ Saved mask to {mask_path}")
    
    return mask_path

def remove_text(input_image_path, mask_path):
    """Use OpenAI to remove text and fill background"""
    print("Removing text with OpenAI API...")
    
    try:
        response = client.images.edit(
            model="gpt-image-1",
            image=open(input_image_path, "rb"),
            mask=open(mask_path, "rb"),
            prompt=(
               "remove all text from this image and other thing remain same"
            ),  
            n=1,
            size="1024x1024"
        )
        
        # Save the result
        timestamp = os.path.splitext(os.path.basename(input_image_path))[0]
        output_path = f"edited_{timestamp}.png"
        
        img_bytes = base64.b64decode(response.data[0].b64_json)
        with open(output_path, "wb") as f:
            f.write(img_bytes)
            
        print(f"✓ Saved edited image to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {str(e)}")
        raise

def main():
    # Use the provided image path
    input_image = "gpt_image_20250608_170030_6b16f8e9.png"
    
    if not os.path.exists(input_image):
        print(f"❌ Error: Image not found at {input_image}")
        return
        
    try:
        # Create mask from input image
        mask_path = create_mask(input_image)
        
        # Remove text using the mask
        output_path = remove_text(input_image, mask_path)
        
        print("\n✨ All done! Process completed successfully:")
        print(f"Input image: {input_image}")
        print(f"Mask created: {mask_path}")
        print(f"Final output: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Process failed: {str(e)}")

if __name__ == "__main__":
    main() 