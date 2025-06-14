#!/usr/bin/env python3
import base64
from openai import OpenAI

# Initialize OpenAI client with API key
client = OpenAI()

def main():
    # Paths to your input image and mask
    input_path = "input_20250608_040431.png"
    mask_path = "mask_alpha_20250608_040431.png"
    
    print(f"Processing image {input_path} with mask {mask_path}...")

    # Call the edit endpoint
    response = client.images.edit(
        model="gpt-image-1",
        image=open(input_path, "rb"),
        mask=open(mask_path, "rb"),
        prompt=(
            "Seamlessly continue the blue and purple gradient background where the text was, "
            "matching the exact colors and lighting of the surrounding areas. "
            "Keep all other parts of the image unchanged."
        ),
        n=1,
        size="1024x1024"
    )

    # Decode and save the result
    output_file = "edited_result_20250608_040431.png"
    b64_data = response.data[0].b64_json
    img_bytes = base64.b64decode(b64_data)
    with open(output_file, "wb") as f:
        f.write(img_bytes)

    print(f"✅ Edited image saved as {output_file}")

if __name__ == "__main__":
    main()