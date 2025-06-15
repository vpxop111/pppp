#!/usr/bin/env python3
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from .env
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def remove_text(input_image_path):
    """Use OpenAI to remove text directly from the image"""
    print("Processing image with OpenAI API...")
    
    try:
        # Define a detailed prompt for text removal
        prompt = (
            "Remove all visible text from this image and seamlessly inpaint the background. "
            "Do not add or modify any objects. Keep the overall layout intact."
        )
        
        response = client.images.edit(
            model="gpt-image-1",
            prompt=prompt,
            image=open(input_image_path, "rb"),
            size="1024x1024",
            quality="low"
        )
        
        # Get the base64 image from response
        image_base64 = response.data[0].b64_json
        
        # Save the result
        timestamp = os.path.splitext(os.path.basename(input_image_path))[0]
        output_path = f"edited_{timestamp}.png"
        
        # Decode and save the image
        image_bytes = base64.b64decode(image_base64)
        with open(output_path, "wb") as f:
            f.write(image_bytes)
            
        print(f"✓ Saved edited image to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {str(e)}")
        raise

def main():
    # Use the specified image path
    input_image = "gpt_image_20250608_152538_e02c5414.png"
    
    if not os.path.exists(input_image):
        print(f"❌ Error: Image not found at {input_image}")
        return
        
    try:
        # Remove text from the image
        output_path = remove_text(input_image)
        
        print("\n✨ All done! Process completed successfully:")
        print(f"Input image: {input_image}")
        print(f"Final output: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Process failed: {str(e)}")

if __name__ == "__main__":
    main() 