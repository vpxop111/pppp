import os
import logging
import uuid
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import openai
import requests
import re
import sys
import traceback
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout for Render
)
logger = logging.getLogger(__name__)

try:
    # Load environment variables
    load_dotenv()

    # Directory setup
    STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    IMAGES_DIR = os.path.join(STATIC_DIR, 'images')
    
    # Create directories if they don't exist
    os.makedirs(STATIC_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Directory for parallel pipeline outputs
    PARALLEL_OUTPUTS_DIR = os.path.join(IMAGES_DIR, 'parallel')
    os.makedirs(PARALLEL_OUTPUTS_DIR, exist_ok=True)

    logger.info("Directory setup completed successfully")
    logger.info(f"STATIC_DIR: {STATIC_DIR}")
    logger.info(f"IMAGES_DIR: {IMAGES_DIR}")
    logger.info(f"PARALLEL_OUTPUTS_DIR: {PARALLEL_OUTPUTS_DIR}")

    # API keys
    OPENAI_API_KEY_ENHANCER = os.getenv('OPENAI_API_KEY_ENHANCER')
    OPENAI_API_KEY_SVG = os.getenv('OPENAI_API_KEY_SVG')

    if not OPENAI_API_KEY_ENHANCER or not OPENAI_API_KEY_SVG:
        raise ValueError("OpenAI API keys must be set in environment variables")

    # OpenAI client setup
    openai.api_key = OPENAI_API_KEY_SVG

    # OpenAI API Endpoints
    OPENAI_API_BASE = "https://api.openai.com/v1"
    OPENAI_CHAT_ENDPOINT = f"{OPENAI_API_BASE}/chat/completions"

    # Model names
    PLANNER_MODEL = "gpt-4.1-nano"
    DESIGN_KNOWLEDGE_MODEL = "gpt-4.1-nano"
    PRE_ENHANCER_MODEL = "gpt-4.1-nano"
    PROMPT_ENHANCER_MODEL = "gpt-4.1-nano"
    GPT_IMAGE_MODEL = "gpt-image-1"
    SVG_GENERATOR_MODEL = "gpt-4.1-nano"
    CHAT_ASSISTANT_MODEL = "gpt-4.1-nano"

except Exception as e:
    logger.error(f"Failed to initialize utils: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def save_image(image_data, prefix="img", format="PNG"):
    """Save image data to file and return the filename"""
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.{format.lower()}"
        filepath = os.path.join(IMAGES_DIR, filename)

        # Convert base64 to image and save
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image.save(filepath, format=format)
        
        logger.info(f"Image saved successfully: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def save_svg(svg_code, prefix="svg"):
    """Save SVG code to file and return the filename"""
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.svg"
        filepath = os.path.join(IMAGES_DIR, filename)

        # Save SVG code to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg_code)
        
        logger.info(f"SVG saved successfully: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving SVG: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def generate_image_with_gpt(enhanced_prompt, design_context=None):
    """Generate image using GPT Image-1 model with enhanced prompting"""
    try:
        logger.info("Generating image with GPT Image-1")

        # Enhance the prompt specifically for GPT Image-1
        optimized_prompt = enhanced_prompt
        logger.info(f"Optimized prompt: {optimized_prompt[:200]}...")

        response = openai.images.generate(
            model=GPT_IMAGE_MODEL,
            prompt=optimized_prompt,
            size="1024x1024",
            quality="medium"   # Using medium quality - gpt-image-1 supports: low, medium, high, auto
            # Note: response_format parameter is not supported by gpt-image-1 (always returns b64_json)
        )

        # Get base64 image data from the response
        if response.data and len(response.data) > 0:
            image_data = response.data[0]
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                image_base64 = image_data.b64_json
            elif hasattr(image_data, 'url') and image_data.url:
                # If we got a URL instead, download the image and convert to base64
                import requests
                img_response = requests.get(image_data.url)
                if img_response.status_code == 200:
                    image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                else:
                    raise Exception(f"Failed to download image from URL: {image_data.url}")
            else:
                raise Exception("No valid image data found in response")
        else:
            raise Exception("Empty response data from OpenAI API")

        # Save the generated image
        filename = save_image(image_base64, prefix="gpt_image")

        logger.info("Image generated and saved successfully with GPT Image-1")
        return image_base64, filename
    except Exception as e:
        logger.error(f"Error generating image with GPT Image-1: {str(e)}")
        logger.error(traceback.format_exc())
        raise 