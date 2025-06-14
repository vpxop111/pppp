#!/usr/bin/env python3
import os
import base64
import logging
from datetime import datetime
import uuid
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mask_to_svg.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key must be set in environment variables")

client = OpenAI(api_key=OPENAI_API_KEY)

def image_to_base64(image_path):
    """Convert image to base64 string"""
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to BytesIO
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_svg_from_mask(image_path):
    """Generate SVG from mask image using GPT-4.1 mini"""
    logger.info(f"Processing mask image: {image_path}")
    
    try:
        # Convert image to base64
        image_base64 = image_to_base64(image_path)
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]

        # Generate SVG directly from image
        logger.info("Generating SVG with GPT-4.1 mini...")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert SVG code generator specialized in converting images to complete, valid SVG code. Your task is to create a precise, clean SVG that exactly matches the provided image.

Key requirements:
1. Generate COMPLETE SVG code - never truncate or leave code incomplete
2. Use 1080x1080 dimensions with appropriate viewBox
3. Break down the image into basic SVG shapes (rect, circle, path, etc.)
4. For complex shapes, use optimized path data with absolute coordinates
5. Use proper color values (hex codes) matching the image exactly
6. Implement proper opacity and blending modes if needed
7. Group related elements with <g> tags
8. Add proper metadata and xmlns attributes
9. Ensure the SVG is self-contained (include all definitions)
10. Optimize for both rendering quality and file size

The SVG must be:
- Complete (no truncation)
- Valid (parseable XML)
- Self-contained (no external dependencies)
- Optimized (efficient path data)
- Exact match to the input image

Return ONLY the complete SVG code, starting with <?xml> and ending with </svg>"""
                },
                {
                    "role": "user",
                    "content": f"Convert this image to a complete, optimized SVG that exactly matches every visual detail. Ensure the output is complete and not truncated.\n\nImage (base64):\n{image_base64}"
                }
            ],
            max_tokens=4000,  # Increased token limit for complete SVG
            temperature=0.8,   # Lower temperature for more precise output
        )
        
        svg_code = response.choices[0].message.content
        
        # Clean up the SVG code (extract just the SVG if there's any extra text)
        if "<svg" in svg_code and "</svg>" in svg_code:
            start = svg_code.find("<svg")
            end = svg_code.find("</svg>") + 6
            svg_code = svg_code[start:end]
        
        # Save the SVG
        output_filename = f"mask_svg_{timestamp}_{unique_id}.svg"
        with open(output_filename, 'w') as f:
            f.write(svg_code)
            
        logger.info(f"✓ SVG generated and saved as: {output_filename}")
        return output_filename
        
    except Exception as e:
        logger.error(f"Error generating SVG: {str(e)}")
        raise

def main():
    # Input mask image path
    input_image = "mask_ChatGPT Image Jun 8, 2025, 04_35_06 AM.png"
    
    if not os.path.exists(input_image):
        logger.error(f"❌ Error: Image not found at {input_image}")
        return
        
    try:
        # Convert mask to SVG
        svg_path = generate_svg_from_mask(input_image)
        
        print("\n✨ All done! Process completed successfully:")
        print(f"Input mask: {input_image}")
        print(f"Output SVG: {svg_path}")
        print("\nCheck the log file for detailed process information.")
        
    except Exception as e:
        print(f"\n❌ Process failed: {str(e)}")

if __name__ == "__main__":
    main()