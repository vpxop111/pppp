from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import json
import logging
from flask_cors import CORS
import re
import base64
from io import BytesIO
import cairosvg
from PIL import Image
import openai
import uuid
from datetime import datetime
from dotenv import load_dotenv
import vtracer  # VTracer for image-to-SVG conversion
# Conditional import of modules that may have additional dependencies
try:
    import remove_text_simple  # Text removal for image preprocessing
    REMOVE_TEXT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"remove_text_simple not available: {e}")
    REMOVE_TEXT_AVAILABLE = False

try:
    import png_to_svg_converter  # PNG to SVG converter utilities
    PNG_CONVERTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"png_to_svg_converter not available: {e}")
    PNG_CONVERTER_AVAILABLE = False
import asyncio
import aiohttp
from functools import lru_cache
import hashlib
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import cv2
from PIL import Image, ImageOps
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS with specific settings
CORS(app, 
     origins=[
         'http://localhost:3000', 
         'http://localhost:3001',
        'http://localhost:3002',
         'http://127.0.0.1:3000', 
         'http://127.0.0.1:3001',
         'https://pppp-351z.onrender.com',
         'https://infoui.vercel.app',
         'https://infoui.vercel.app/'
     ],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom logging filter to suppress 404 errors for missing image files
class ImageNotFoundFilter(logging.Filter):
    def filter(self, record):
        # Suppress 404 logs for missing image files
        try:
            log_message = str(record.getMessage()) if hasattr(record, 'getMessage') else str(record.msg)
            
            # Check if this is a 404 request for image files
            if ('404' in log_message and 
                ('/static/images/' in log_message or 
                 'background_' in log_message or 
                 '.png HTTP/1.1' in log_message or 
                 '.jpg HTTP/1.1' in log_message or 
                 '.svg HTTP/1.1' in log_message)):
                return False
        except:
            pass
        return True

# Apply the filter to suppress image 404 logs from werkzeug (Flask's request logger)
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(ImageNotFoundFilter())
werkzeug_logger.setLevel(logging.WARNING)  # Only show warnings and errors, not INFO

# Directory setup
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Public URL configuration for deployed environment
def get_public_base_url():
    """Get the public base URL for serving images"""
    # Check if we're in production (Render deployment)
    if os.getenv('PORT'):
        # Production environment - use the deployed URL
        return 'https://pppp-351z.onrender.com'
    else:
        # Local development environment
        return 'http://localhost:5000'

def get_public_image_url(relative_path):
    """Convert relative image path to public URL"""
    base_url = get_public_base_url()
    # Ensure the path starts with /static/images/
    if not relative_path.startswith('/static/images/'):
        if relative_path.startswith('static/images/'):
            relative_path = '/' + relative_path
        elif relative_path.startswith('sessions/'):
            relative_path = '/static/images/' + relative_path
        else:
            relative_path = '/static/images/' + relative_path
    
    return f"{base_url}{relative_path}"

# API keys
OPENAI_API_KEY_ENHANCER = os.getenv('OPENAI_API_KEY_ENHANCER')
OPENAI_API_KEY_SVG = os.getenv('OPENAI_API_KEY_SVG')

if not OPENAI_API_KEY_ENHANCER or not OPENAI_API_KEY_SVG:
    raise ValueError("OpenAI API keys must be set in environment variables")

# OpenAI client setupkk
openai.api_key = OPENAI_API_KEY_SVG

# OpenAI API Endpoints
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_CHAT_ENDPOINT = f"{OPENAI_API_BASE}/chat/completions"

# Model names - updated to use GPT-4.1 mini for logic/text and gpt-image for images
PLANNER_MODEL = "gpt-4o-mini"
DESIGN_KNOWLEDGE_MODEL = "gpt-4o-mini"
PRE_ENHANCER_MODEL = "gpt-4o-mini"
PROMPT_ENHANCER_MODEL = "gpt-4o-mini"
GPT_IMAGE_MODEL = "gpt-image-1"
SVG_GENERATOR_MODEL = "gpt-4o-mini"
CHAT_ASSISTANT_MODEL = "gpt-4o-mini"

# Add parallel SVG processing imports
from concurrent.futures import ThreadPoolExecutor
import pytesseract
import numpy as np

# VTracer and related features availability check
try:
    # Test if vtracer is functional by checking its main function
    vtracer.convert_image_to_svg_py
    VTRACER_AVAILABLE = True
    logger.info("✅ VTracer is available and functional")
except (ImportError, AttributeError) as e:
    logger.warning(f"⚠️ VTracer not available: {e}")
    VTRACER_AVAILABLE = False

# Check overall parallel features availability
PARALLEL_FEATURES_AVAILABLE = VTRACER_AVAILABLE and REMOVE_TEXT_AVAILABLE
if PARALLEL_FEATURES_AVAILABLE:
    logger.info("🎉 Full parallel SVG features are available (VTracer + text removal)")
elif VTRACER_AVAILABLE:
    logger.info("⚡ Basic VTracer functionality is available (without text removal)")
else:
    logger.warning("❌ VTracer functionality is not available")

# Unified storage directory - ALL files go here in organized sessions
UNIFIED_STORAGE_DIR = os.path.join(IMAGES_DIR, 'sessions')
os.makedirs(UNIFIED_STORAGE_DIR, exist_ok=True)

# Legacy parallel directory (kept for backward compatibility)
PARALLEL_OUTPUTS_DIR = os.path.join(IMAGES_DIR, 'parallel')
os.makedirs(PARALLEL_OUTPUTS_DIR, exist_ok=True)

def enhance_prompt_for_gpt_image(user_prompt, design_context=None):
    """Single optimized prompt enhancement for GPT Image-1 based on 2024 best practices"""
    logger.info(f"Enhancing prompt for GPT Image-1: {user_prompt[:100]}...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

    # Create optimized system prompt based on GPT-4 image generation best practices
    system_prompt = """You are an expert GPT-Image-1 (DALL-E 3) prompt enhancement specialist. Your role is to transform basic user image requests into sophisticated, optimized prompts that produce professional, "next-level" image outputs.

IMPORTANT: For this task, avoid using gradient shapes or elements with gradients. Use clear, structured, and solid shapes with strong contrast. Avoid shapes or elements that have similar shades or colors as the background, as these are difficult for vectorization tools like VTracer to identify. All main elements should be visually distinct from the background and from each other, with no subtle gradients or blending. Use flat, solid colors for all shapes and ensure good separation between elements and background.

Core Enhancement Methodology
Transform user inputs using this proven structure:
[Core Subject] + [Setting/Environment] + [Style/Artistic Direction] + [Lighting/Mood] + [Technical Details] + [Composition/Perspective]
Key Principles:

Use natural, conversational language (NOT keyword lists)
Aim for 30-50 words for optimal results
Include rich contextual details and environmental descriptions
Leverage DALL-E 3's synthetic captioning strengths
Focus on complete sentences and narrative descriptions

Enhancement Framework
1. Quality Enhancement Keywords (integrate naturally):

Technical: "highly detailed," "professional photography," "studio quality," "award-winning"
Artistic: "trending on ArtStation," "masterpiece," "hyperrealistic," "cinematic"
Photographic: "shallow depth of field," "perfect composition," "professional lighting"

2. Lighting Enhancement (choose appropriate):

Dramatic: "cinematic lighting," "dramatic shadows," "rim lighting"
Professional: "studio lighting," "soft diffused light," "key lighting"
Atmospheric: "golden hour," "ethereal glow," "moody lighting"
Natural: "natural daylight," "window light," "ambient lighting"

3. Composition Control:

Camera Angles: eye level, bird's eye view, low angle, high angle
Shot Types: close-up, medium shot, wide shot, extreme close-up
Technical Specs: 85mm portrait lens, 24mm wide-angle, f/1.4 aperture

4. Style Direction (match to purpose):

Corporate/Professional: "clean modern design," "minimalist style," "professional aesthetic"
Marketing: "vibrant colors," "eye-catching design," "commercial photography style"
Artistic: "oil painting style," "digital art," "watercolor technique"

Specialized Enhancement Rules
POSTERS (Coming Soon, Events, Marketing):
Formula: "Professional [poster type] featuring [main element], [style description], [color scheme], [mood/atmosphere], [technical quality], [composition notes]"

Enhancement Focus:
- Clear hierarchy and focal points
- Bold, readable typography integration
- Brand-appropriate color schemes
- Marketing-optimized composition
- High-impact visual elements

Example Enhancement:
User: "coming soon poster"
Enhanced: "Professional 'Coming Soon' marketing poster featuring bold typography and anticipation-building imagery, modern minimalist design with vibrant brand colors, dramatic lighting creating excitement, studio-quality commercial photography style, centered composition with clear visual hierarchy."
LOGOS (Brand Identity, Business):
Formula: "Flat vector logo design for [business type], [design elements], [color scheme], [style notes], scalable vector format"

Enhancement Focus:
- Emphasize "flat vector design" and "scalable"
- Minimalist, clean aesthetics
- Professional color schemes
- Clear symbolic representation
- Timeless design principles

Example Enhancement:
User: "tech startup logo"
Enhanced: "Flat vector logo design for innovative tech startup, featuring abstract geometric symbol representing growth and connectivity, modern blue and silver color palette, minimalist professional style with clean lines, scalable vector format suitable for all applications."
TESTIMONIALS (Reviews, Social Proof):
Formula: "Professional testimonial graphic featuring [layout description], [trust elements], [design style], [color scheme], [quality specifications]"

Enhancement Focus:
- Trust-building visual elements
- Clean, readable layouts
- Professional photography style
- Space for customer photos and quotes
- Platform-optimized dimensions

Example Enhancement:
User: "customer testimonial"
Enhanced: "Professional testimonial graphic featuring clean modern layout with space for customer photo and quote text, trust-building design elements including star ratings, soft professional lighting, corporate blue and white color scheme, high-quality marketing material style."
Advanced Enhancement Techniques
Conversational Prompting:

Transform keywords into natural descriptions
Add environmental context and relationships
Include emotional and atmospheric details
Specify material textures and surface qualities

Mood-First Enhancement:

Start with emotional tone and atmosphere
Build subject details around mood
Include sensory descriptions
Add temporal and seasonal context

Technical Optimization:

Specify appropriate aspect ratios (square for social, landscape for banners, portrait for posters)
Include style settings guidance (vivid for marketing, natural for professional)
Add resolution and quality specifications
Mention lighting setup and camera techniques

Output Format
Always provide:

Enhanced Prompt: The complete, optimized prompt (30-50 words)
Key Enhancements: Brief explanation of major improvements made
Technical Notes: Any specific settings or considerations for best results
Alternative Version: A second variation with different style/mood (if helpful)

Quality Checklist
Before finalizing each enhancement, verify:

✅ Uses natural, conversational language
✅ Includes complete environmental context
✅ Incorporates appropriate quality keywords
✅ Specifies lighting and mood clearly
✅ Matches optimal word count (30-50 words)
✅ Optimized for specific image type/purpose
✅ Includes technical and composition details
✅ Maintains user's core intent while maximizing quality

Important Notes

Focus on rich contextual descriptions rather than keyword lists
Leverage DALL-E 3's strength in understanding relationships and environments
Always maintain the user's original intent while dramatically improving specificity
Consider the final use case (social media, print, web, branding) in enhancement decisions
Remember that DALL-E 3 excels with complete scene descriptions including background elements

Transform every basic request into a sophisticated prompt that unlocks DALL-E 3's full potential for professional, next-level image generation."""

    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f" {user_prompt}"
            }
        ],
        "temperature": 0.8,
        "max_tokens": 1000
    }

    try:
        logger.info("Calling OpenAI Chat API for optimized prompt enhancement")
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response_data}")
            return user_prompt

        enhanced_prompt = response_data["choices"][0]["message"]["content"]
        logger.info(f"Successfully enhanced prompt. Result: {enhanced_prompt[:100]}...")
        return enhanced_prompt

    except Exception as e:
        logger.error(f"Error enhancing prompt: {str(e)}")
        return user_prompt

def generate_image_with_gpt(enhanced_prompt, design_context=None):
    """Generate image using GPT Image-1 model with enhanced prompting"""
    try:
        logger.info("Generating image with GPT Image-1")
        logger.info(f"Using prompt: {enhanced_prompt[:200]}...")

        response = openai.images.generate(
            model=GPT_IMAGE_MODEL,
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="medium"   # Changed from "low" to "standard" for better quality
        )

        # Get base64 image data from the response
        # The response structure changed in newer versions of the API
        image_base64 = response.data[0].b64_json if hasattr(response.data[0], 'b64_json') else response.data[0].url

        # Create session ID for this generation
        session_id = f"svg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Save the generated image
        filename, relative_path, _ = save_image(image_base64, prefix="gpt_image", session_id=session_id)
        
        # Create public URL for the image
        public_url = get_public_image_url(relative_path)

        logger.info("Image generated and saved successfully with GPT Image-1")
        logger.info(f"Public URL created: {public_url}")
        return image_base64, relative_path, public_url
    except Exception as e:
        logger.error(f"Error generating image with GPT Image-1: {str(e)}")
        raise

def generate_svg_from_image(image_base64, enhanced_prompt, use_preprocessing=False):
    """Generate SVG code from image using VTracer with optimized processing"""
    logger.info(f"SVG generation from image using VTracer")
    logger.info(f"Enhanced prompt provided: {enhanced_prompt[:100]}...")
    logger.info(f"Image data size: {len(image_base64)} characters")
    logger.info(f"Preprocessing enabled: {use_preprocessing}")
    
    if not VTRACER_AVAILABLE:
        logger.error("VTracer not available - missing core dependency")
        raise NotImplementedError("VTracer not available - missing core dependency. Please install vtracer.")
    
    try:
        # Decode base64 image data
        image_data = base64.b64decode(image_base64)
        
        # Create session for this conversion
        session_id = f"vtracer_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Save the original image bytes to a temporary PNG file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_input_path = f"temp_input_{timestamp}_{uuid.uuid4()}.png"
        
        with open(temp_input_path, "wb") as f:
            f.write(image_data)
        
        logger.info(f"Saved temporary image: {temp_input_path}")
        
        # Optional preprocessing for better VTracer results
        processing_path = temp_input_path
        if use_preprocessing:
            try:
                logger.info("Applying VTracer preprocessing...")
                processing_path = preprocess_for_vtracer(temp_input_path)
                logger.info(f"Preprocessing completed: {processing_path}")
            except Exception as e:
                logger.warning(f"Preprocessing failed, using original image: {str(e)}")
                processing_path = temp_input_path
        
        # Convert the image to SVG using vtracer with improved settings
        svg_filename, svg_relative_path, _ = save_svg("", prefix="vtracer_svg", session_id=session_id)
        output_svg_path = os.path.join(IMAGES_DIR, svg_relative_path)
        
        logger.info("Converting image to SVG using VTracer with high detail settings...")
        vtracer.convert_image_to_svg_py(
            processing_path,
            output_svg_path,
            # Clustering settings
            colormode='color',         # Color mode (not B/W)
            hierarchical='stacked',    # Stacked mode for layering
            
            # Filter settings
            filter_speckle=1,         # Less filtering, keep more small elements
            
            # Color and precision settings
            color_precision=3,        # More color detail
            layer_difference=16,      # More layers
            
            # Curve fitting settings
            mode='spline',            # Smooth curves
            corner_threshold=30,      # More corners
            length_threshold=1.0,     # Keep short paths
            splice_threshold=20,      # Less aggressive merging
            
            # Additional optimization settings
            max_iterations=20,        # More refinement
            path_precision=2          # Higher precision
        )
        
        # Read the generated SVG content
        with open(output_svg_path, 'r', encoding='utf-8') as f:
            svg_code = f.read()
        
        # Clean up temporary files
        cleanup_files = [temp_input_path]
        if use_preprocessing and processing_path != temp_input_path:
            cleanup_files.append(processing_path)
            
        for file_path in cleanup_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        logger.info(f"VTracer SVG generation completed successfully")
        logger.info(f"Generated SVG size: {len(svg_code)} characters")
        logger.info(f"SVG saved to session: {session_id}")
        
        return svg_code
        
    except Exception as e:
        # Clean up temporary files on error
        cleanup_files = []
        if 'temp_input_path' in locals() and os.path.exists(temp_input_path):
            cleanup_files.append(temp_input_path)
        if 'processing_path' in locals() and processing_path != temp_input_path and os.path.exists(processing_path):
            cleanup_files.append(processing_path)
            
        for file_path in cleanup_files:
            try:
                os.remove(file_path)
            except:
                pass
        
        logger.error(f"Error in VTracer SVG generation: {str(e)}")
        raise Exception(f"VTracer SVG generation failed: {str(e)}")

def clean_svg_code_original(svg_code):
    """Original clean and validate SVG code function"""
    try:
        from xml.dom.minidom import parseString
        from xml.parsers.expat import ExpatError
        
        # Parse and clean the SVG
        try:
            doc = parseString(svg_code)
            
            # Get the SVG element
            svg_element = doc.documentElement
            
            # Ensure viewBox exists (minimal changes from original)
            if not svg_element.hasAttribute('viewBox'):
                svg_element.setAttribute('viewBox', '0 0 1080 1080')
            
            # Convert back to string with pretty printing
            cleaned_svg = doc.toxml()
            logger.info("SVG cleaned successfully")
            return cleaned_svg
            
        except ExpatError:
            logger.error("Failed to parse SVG, returning original")
            return svg_code
            
    except Exception as error:
        logger.error(f"Error cleaning SVG: {str(error)}")
        return svg_code

def save_image(image_data, prefix="img", format="PNG", session_id=None):
    """Save image data to unified storage folder and return the filename"""
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.{format.lower()}"
        
        # Use session_id if provided, otherwise create one
        if not session_id:
            session_id = f"session_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # All files go to unified storage with session organization
        session_folder = os.path.join(UNIFIED_STORAGE_DIR, session_id)
        os.makedirs(session_folder, exist_ok=True)
        filepath = os.path.join(session_folder, filename)

        # Convert base64 to image and save
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image.save(filepath, format=format)
        
        # Return relative path from IMAGES_DIR
        relative_path = os.path.relpath(filepath, IMAGES_DIR)
        logger.info(f"Image saved successfully: {filename} in session {session_id}")
        return filename, relative_path, session_id
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        raise

def save_svg(svg_code, prefix="svg", session_id=None):
    """Save SVG code to unified storage folder and return the filename"""
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.svg"
        
        # Use session_id if provided, otherwise create one
        if not session_id:
            session_id = f"session_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # All files go to unified storage with session organization
        session_folder = os.path.join(UNIFIED_STORAGE_DIR, session_id)
        os.makedirs(session_folder, exist_ok=True)
        filepath = os.path.join(session_folder, filename)

        # Save SVG code to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg_code)
        
        # Return relative path from IMAGES_DIR
        relative_path = os.path.relpath(filepath, IMAGES_DIR)
        logger.info(f"SVG saved successfully: {filename} in session {session_id}")
        return filename, relative_path, session_id
    except Exception as e:
        logger.error(f"Error saving SVG: {str(e)}")
        raise

def convert_svg_to_png(svg_code):
    """Convert SVG code to PNG and save both files"""
    try:
        # Save SVG file
        svg_filename, svg_relative_path, session_id = save_svg(svg_code)
        
        # Convert to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
        
        # Save PNG file in the same session as SVG
        png_filename, png_relative_path, _ = save_image(
            base64.b64encode(png_data).decode('utf-8'),
            prefix="converted_svg",
            format="PNG",
            session_id=session_id
        )
        
        return svg_relative_path, png_relative_path
    except Exception as e:
        logger.error(f"Error in SVG to PNG conversion: {str(e)}")
        raise

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """Serve images from the images directory with subfolder support"""
    # Handle both direct files and files in subfolders
    filepath = os.path.join(IMAGES_DIR, filename)
    
    # Security check - ensure the path is within IMAGES_DIR
    if not os.path.abspath(filepath).startswith(os.path.abspath(IMAGES_DIR)):
        return "Access denied", 403
    
    # Extract directory and filename
    directory = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    
    return send_from_directory(directory, basename)

@app.route('/static/images/parallel/<path:session_folder>/<path:filename>')
def serve_parallel_image(session_folder, filename):
    """Serve images from the parallel pipeline directory (legacy)"""
    parallel_path = os.path.join(PARALLEL_OUTPUTS_DIR, session_folder)
    return send_from_directory(parallel_path, filename)

@app.route('/static/images/sessions/<path:session_id>/<path:filename>')
def serve_session_file(session_id, filename):
    """Serve files from the unified sessions directory"""
    # Security check - ensure the path is within UNIFIED_STORAGE_DIR
    session_path = os.path.join(UNIFIED_STORAGE_DIR, session_id)
    if not os.path.abspath(session_path).startswith(os.path.abspath(UNIFIED_STORAGE_DIR)):
        return "Access denied", 403
    
    return send_from_directory(session_path, filename)

@app.route('/api/projects/templates', methods=['GET'])
def get_templates():
    """Mock templates endpoint - returns empty templates for now"""
    page = request.args.get('page', '1')
    limit = request.args.get('limit', '4')
    
    # Return empty templates response for now
    # TODO: Implement actual templates functionality
    return jsonify({
        "data": [],
        "pagination": {
            "page": int(page),
            "limit": int(limit),
            "total": 0,
            "totalPages": 0
        }
    })

@app.route('/api/generate-svg', methods=['POST'])
def generate_svg():
    """Universal SVG generator endpoint for any design request"""
    try:
        data = request.json
        user_input = data.get('prompt', '')
        skip_enhancement = data.get('skip_enhancement', False)

        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        logger.info("="*80)
        logger.info(f"Starting new design request: {user_input}")
        logger.info("="*80)

        # Single Stage: Optimize prompt for GPT Image-1 generation
        logger.info("\n[OPTIMIZED PROMPT ENHANCEMENT]")
        logger.info("-"*50)
        logger.info("Enhancing prompt for GPT Image-1 generation...")
        
        # Use the optimized prompt enhancement
        enhanced_prompt = enhance_prompt_for_gpt_image(user_input)
        logger.info("\nOptimized Prompt Generated:")
        for line in enhanced_prompt.split('\n')[:5]:  # Log first 5 lines
            logger.info(f"  {line}")
        logger.info("  ...")

        # Generate image using GPT Image-1
        logger.info("\n[IMAGE GENERATION PHASE]")
        gpt_image_base64, gpt_image_filename = generate_image_with_gpt(enhanced_prompt, None)
        logger.info("Image generated with GPT Image-1")

        # Generate SVG using vtracer
        logger.info("\n[SVG GENERATION PHASE]")
        svg_code = generate_svg_from_image(gpt_image_base64, enhanced_prompt)
        logger.info("SVG code generated from image")
        
        # Save the SVG in the same session
        svg_filename, svg_relative_path, _ = save_svg(svg_code, prefix="svg", session_id=session_id)

        return jsonify({
            "original_prompt": user_input,
            "enhanced_prompt": enhanced_prompt,
            "gpt_image_base64": gpt_image_base64,
            "gpt_image_url": f"/static/images/{gpt_image_filename}",
            "svg_code": svg_code,
            "svg_path": svg_relative_path,
            "stages": {
                "vector_suitability": {
                    "completed": True,
                    "suitable": True
                },
                "design_plan": {
                    "completed": True,
                    "content": design_plan if 'design_plan' in locals() else ""
                },
                "design_knowledge": {
                    "completed": True, 
                    "content": design_knowledge if 'design_knowledge' in locals() else ""
                },
                "pre_enhancement": {
                    "completed": True,
                    "skipped": skip_enhancement,
                    "content": pre_enhanced_prompt
                },
                "prompt_enhancement": {
                    "completed": True,
                    "skipped": skip_enhancement,
                    "content": enhanced_prompt
                },
                "image_generation": {
                    "completed": True, 
                    "image_url": f"/static/images/{gpt_image_filename}"
                },
                "svg_generation": {
                    "completed": True, 
                    "svg_path": svg_relative_path
                }
            },
            "progress": 100
        })

    except Exception as e:
        logger.error(f"Error in generate_svg: {str(e)}")
        return jsonify({"error": str(e)}), 500

def chat_with_ai_about_design(messages, current_svg=None):
    """Enhanced conversational AI that can discuss and modify designs"""
    logger.info("Starting conversational AI interaction")
    logger.info(f"Processing {len(messages)} messages with {'SVG context' if current_svg else 'no context'}")

    # Create system prompt that includes SVG knowledge
    system_prompt = """You are an expert AI design assistant with deep knowledge of SVG creation and manipulation. You can:

1. Create new designs from scratch
2. Explain existing SVG designs in detail
3. Modify existing designs based on user feedback
4. Provide design suggestions and improvements
5. Discuss design principles, colors, typography, and layout

When discussing SVGs, you understand:
- SVG elements like <rect>, <circle>, <path>, <text>, <g>
- Attributes like fill, stroke, viewBox, transform
- Design principles like color theory, typography, layout
- How to make designs accessible and responsive

Guidelines:
- Be conversational and helpful
- Explain technical concepts in simple terms
- Ask clarifying questions when needed
- Provide specific suggestions for improvements
- When modifying designs, explain what changes you're making and why

Current context: You are helping a user with their design project."""

    if current_svg:
        system_prompt += f"\n\nCurrent SVG design context:\n```svg\n{current_svg}\n```\n\nYou can reference and modify this design based on user requests."

    # Prepare messages for the AI
    ai_messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history (limit to last 10 messages to manage context)
    conversation_messages = messages[-10:] if len(messages) > 10 else messages
    
    for msg in conversation_messages:
        if msg["role"] in ["user", "assistant"]:
            # Clean SVG code blocks from previous messages to avoid clutter
            content = msg["content"]
            if "```svg" in content and msg["role"] == "assistant":
                # Keep only the explanation part, not the SVG code
                parts = content.split("```svg")
                if len(parts) > 1:
                    explanation = parts[0].strip()
                    if explanation:
                        content = explanation
                    else:
                        content = "I provided a design based on your request."
            
            ai_messages.append({
                "role": msg["role"],
                "content": content
            })

    try:
        # Use OpenAI client directly instead of raw API calls
        client = openai.OpenAI(api_key=OPENAI_API_KEY_ENHANCER)
        response = client.chat.completions.create(
            model=CHAT_ASSISTANT_MODEL,
            messages=ai_messages,
            temperature=0.7,
            max_tokens=8000
        )
        
        # Extract the response content safely
        if response and response.choices and len(response.choices) > 0:
            ai_response = response.choices[0].message.content
            logger.info(f"AI response generated: {ai_response[:100]}...")
            return ai_response
        else:
            logger.error("Empty or invalid response from OpenAI")
            return "I apologize, but I'm having trouble generating a response. Could you please rephrase your request?"
            
    except Exception as e:
        logger.error(f"Error in chat_with_ai_about_design: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

def modify_svg_with_ai(original_svg, modification_request):
    """Use AI to modify an existing SVG based on user request"""
    logger.info(f"Modifying SVG with request: {modification_request}")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }

    system_prompt = """You are an expert SVG modifier. Given an original SVG and a modification request, create a new SVG that incorporates the requested changes.

Rules:
1. Maintain the overall structure and quality of the original design
2. Make only the requested modifications
3. Ensure the SVG is valid and well-formed
4. Keep the viewBox and dimensions appropriate
5. Maintain good design principles
6. Return ONLY the modified SVG code, no explanations

The SVG should be production-ready and properly formatted."""

    payload = {
        "model": SVG_GENERATOR_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Original SVG:\n```svg\n{original_svg}\n```\n\nModification request: {modification_request}\n\nPlease provide the modified SVG:"
            }
        ],
        "temperature": 1,
        "max_tokens": 8000
    }

    logger.info("Calling AI for SVG modification")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"SVG modification error: {response_data}")
        return None

    modified_content = response_data["choices"][0]["message"]["content"]
    
    # Extract SVG code
    svg_pattern = r'<svg.*?<\/svg>'
    svg_matches = re.search(svg_pattern, modified_content, re.DOTALL)
    
    if svg_matches:
        logger.info("Successfully modified SVG")
        return svg_matches.group(0)
    
    logger.warning("Could not extract modified SVG, returning original")
    return original_svg

@app.route('/api/chat-assistant', methods=['POST'])
def chat_assistant():
    try:
        data = request.json
        messages = data.get('messages', [])
        
        logger.info("="*80)
        logger.info("CHAT ASSISTANT REQUEST")
        logger.info("="*80)
        logger.info(f"Chat history length: {len(messages)}")
        logger.info(f"Last message: {messages[-1] if messages else 'No messages'}")
        
        if not messages:
            logger.warning("No messages provided in request")
            return jsonify({"error": "No messages provided"}), 400

        # Get the latest user message
        latest_message = messages[-1]["content"].lower() if messages else ""
        
        # Analyze request type
        logger.info("\n[Request Analysis]")
        logger.info("-"*50)
        
        # Check request type
        is_create_request = any(keyword in latest_message for keyword in [
            "create", "design", "generate", "make", "draw", "poster", "build"
        ]) and not any(word in latest_message for word in ["edit", "update", "modify", "change"])

        is_modify_request = any(word in latest_message for word in ["edit", "update", "modify", "change", "adjust"]) and any(keyword in latest_message for keyword in ["design", "poster", "color", "text", "font", "size"])

        logger.info(f"Request type: {'Creation' if is_create_request else 'Modification' if is_modify_request else 'Conversation'}")
        logger.info(f"User message: {latest_message}")

        # Find existing SVG if any
        current_svg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and "```svg" in msg.get("content", ""):
                svg_start = msg["content"].find("```svg") + 6
                svg_end = msg["content"].find("```", svg_start)
                if svg_end > svg_start:
                    current_svg = msg["content"][svg_start:svg_end].strip()
                    logger.info("Found existing SVG in conversation")
                    break

        if is_create_request:
            logger.info("\n[Starting New Design Creation]")
            logger.info("-"*50)
            
            try:
                # Create session ID for this chat conversation
                chat_session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                
                # Optimized Prompt Enhancement
                logger.info("\n[OPTIMIZED PROMPT ENHANCEMENT]")
                logger.info("-"*50)
                logger.info("Enhancing prompt for GPT Image-1...")
                enhanced_prompt = enhance_prompt_for_gpt_image(latest_message)
                logger.info("\nEnhanced Prompt Generated:")
                for line in enhanced_prompt.split('\n')[:5]:
                    logger.info(f"  {line}")
                logger.info("  ...")
                
                # Stage 5: Image Generation
                logger.info("\n[STAGE 5: Image Generation]")
                logger.info("-"*50)
                logger.info("Generating initial design image...")
                logger.info(f"Using model: {GPT_IMAGE_MODEL}")
                image_base64, image_filename = generate_image_with_gpt(enhanced_prompt, design_context)
                logger.info(f"Image generated and saved as: {image_filename}")
                
                # Stage 6: SVG Generation
                logger.info("\n[STAGE 6: SVG Generation]")
                logger.info("-"*50)
                logger.info("Converting design to SVG format...")
                logger.info(f"Using model: {SVG_GENERATOR_MODEL}")
                svg_code = generate_svg_from_image(image_base64, enhanced_prompt)
                svg_filename, svg_relative_path, _ = save_svg(svg_code, prefix="assistant_svg", session_id=chat_session_id)
                logger.info(f"SVG generated and saved as: {svg_filename}")
                
                # Stage 7: Design Explanation
                logger.info("\n[STAGE 7: Design Explanation]")
                logger.info("-"*50)
                logger.info("Generating design explanation...")
                logger.info(f"Using model: {CHAT_ASSISTANT_MODEL}")
                
                explanation_prompt = f"I've created a design for the user. Here's the SVG code:\n\n```svg\n{svg_code}\n```\n\nPlease explain this design to the user in a friendly, conversational way. Describe the elements, colors, layout, and how it addresses their request."
                
                temp_messages = messages + [{"role": "user", "content": explanation_prompt}]
                ai_explanation = chat_with_ai_about_design(temp_messages, svg_code)
                
                logger.info("\nExplanation Generated:")
                for line in ai_explanation.split('\n')[:5]:
                    logger.info(f"  {line}")
                logger.info("  ...")
                
                # Create comprehensive response
                full_response = f"{ai_explanation}\n\n```svg\n{svg_code}\n```\n\nFeel free to ask me to modify any aspect of this design!"
                
                messages.append({"role": "assistant", "content": full_response})
                
                response_data = {
                    "response": full_response,
                    "svg_code": svg_code,
                    "svg_path": svg_relative_path,
                    "messages": messages
                }
                
                logger.info("\n[Design Creation Complete]")
                logger.info("="*80)
                logger.info("Summary:")
                logger.info(f"- Design plan created")
                logger.info(f"- Design knowledge gathered")
                logger.info(f"- Prompt enhanced and refined")
                logger.info(f"- Image generated: {image_filename}")
                logger.info(f"- SVG created: {svg_filename}")
                logger.info(f"- Explanation provided")
                logger.info("="*80)
                
                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"Error in design creation: {str(e)}")
                error_response = "I encountered an error while creating the design. Let me try a different approach or you can rephrase your request."
                messages.append({"role": "assistant", "content": error_response})
                return jsonify({"messages": messages})

        elif is_modify_request and current_svg:
            logger.info("Processing design modification request")
            
            try:
                # Modify the existing SVG
                modified_svg = modify_svg_with_ai(current_svg, latest_message)
                
                if modified_svg and modified_svg != current_svg:
                    # Create session ID for modification
                    mod_session_id = f"mod_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                    
                    # Save the modified SVG
                    svg_filename, svg_relative_path, _ = save_svg(modified_svg, prefix="modified_svg", session_id=mod_session_id)
                    
                    # Get AI explanation of the changes
                    change_explanation_prompt = f"I've modified the design based on the user's request: '{latest_message}'. Here's the updated SVG:\n\n```svg\n{modified_svg}\n```\n\nPlease explain what changes were made and how the design now better meets their needs."
                    
                    temp_messages = messages + [{"role": "user", "content": change_explanation_prompt}]
                    ai_explanation = chat_with_ai_about_design(temp_messages, modified_svg)
                    
                    full_response = f"{ai_explanation}\n\n```svg\n{modified_svg}\n```\n\nIs there anything else you'd like me to adjust?"
                    
                    messages.append({"role": "assistant", "content": full_response})
                    
                    response_data = {
                        "response": full_response,
                        "svg_code": modified_svg,
                        "svg_path": svg_relative_path,
                        "messages": messages
                    }
                    logger.info("Successfully modified design with explanation")
                    return jsonify(response_data)
                else:
                    # Fallback to conversational response
                    ai_response = chat_with_ai_about_design(messages, current_svg)
                    messages.append({"role": "assistant", "content": ai_response})
                    return jsonify({"messages": messages})
                    
            except Exception as e:
                logger.error(f"Error in design modification: {str(e)}")
                ai_response = "I had trouble modifying the design. Could you be more specific about what changes you'd like me to make?"
                messages.append({"role": "assistant", "content": ai_response})
                return jsonify({"messages": messages})

        else:
            # Handle general conversation
            logger.info("Processing general conversation")
            ai_response = chat_with_ai_about_design(messages, current_svg)
            messages.append({"role": "assistant", "content": ai_response})
            
            return jsonify({
                "messages": messages,
                "svg_code": current_svg,
                "svg_path": None,
                "response": ai_response
            })
            
    except Exception as e:
        error_msg = f"Error in chat_assistant: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        return jsonify({"error": error_msg}), 500


def process_ocr_svg(image_data):
    """Generate a text-only SVG using enhanced OCR and AI-powered text analysis"""
    if not PARALLEL_FEATURES_AVAILABLE:
        raise NotImplementedError("Parallel features not available - missing dependencies")
    
    # Base64-encode the PNG image
    img_b64 = base64.b64encode(image_data).decode('utf-8')
    
    # Enhanced system prompt for accurate text detection
    system_prompt = """You are an expert OCR and SVG text specialist. Your task is to create PRECISE, ACCURATE SVG code that reproduces ALL text elements from the image with PERFECT positioning, sizing, and styling.

CRITICAL REQUIREMENTS:

1. **ACCURATE TEXT DETECTION**:
   - Detect ALL text in the image, including titles, subtitles, body text, captions, labels
   - Identify exact text content character by character
   - Don't miss any text elements, no matter how small or faint

2. **PRECISE FONT ANALYSIS**:
   - Analyze font family (serif, sans-serif, decorative, script)
   - Determine exact font weight (normal, bold, extra-bold)
   - Identify font style (normal, italic, oblique)
   - Estimate accurate font size in pixels

3. **EXACT POSITIONING**:
   - Measure precise x,y coordinates for each text element
   - Use percentages or absolute pixels for accurate placement
   - Maintain proper text alignment (left, center, right)
   - Preserve line spacing and letter spacing

4. **COLOR MATCHING**:
   - Extract exact colors using hex codes (#RRGGBB)
   - Match text colors precisely from the image
   - Account for text shadows or effects if present

5. **SVG STRUCTURE**:
   - Use viewBox="0 0 1080 1080" and dimensions 1080x1080
   - Include proper text-anchor attributes (start, middle, end)
   - Use <defs> for font loading and styling
   - Group related text elements logically

6. **QUALITY STANDARDS**:
   - Text must be crisp and readable
   - Maintain proper hierarchy (titles larger than subtitles)
   - Ensure text doesn't overlap unless intended
   - Optimize for web rendering

EXAMPLE STRUCTURE:
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080" width="1080" height="1080">
  <defs>
    <style>
      .main-title { font-family: 'Arial Black', sans-serif; font-size: 120px; fill: #2E3A99; font-weight: bold; }
      .subtitle { font-family: 'Arial', sans-serif; font-size: 48px; fill: #FFFFFF; font-weight: normal; }
    </style>
  </defs>
  <text x="540" y="300" class="main-title" text-anchor="middle">EXACT TEXT</text>
  <text x="540" y="400" class="subtitle" text-anchor="middle">Subtitle Text</text>
</svg>
```

Return ONLY the complete SVG code with ALL text elements accurately positioned and styled."""

    user_content = [
        {"type": "text", "text": "Analyze this image and extract ALL text elements with PRECISE positioning, font sizing, colors, and styling. Create accurate SVG code that reproduces the text exactly as shown in the image."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
    ]
    
    # Call Chat Completions API directly to support image_url message with enhanced parameters
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 1,  # Lower temperature for more consistent text detection
        "max_tokens": 12000,  # More tokens for detailed SVG output# Avoid repetitive patterns
    }
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    if response.status_code != 200:
        logger.error(f"Error generating text SVG: {data}")
        raise Exception("Text SVG generation failed")
    
    content = data["choices"][0]["message"]["content"]
    
    # Extract the SVG
    match = re.search(r'<svg.*?</svg>', content, re.DOTALL)
    svg_code = match.group(0) if match else content.strip()
    
    # Save and return
    svg_filename, svg_relative_path, session_id = save_svg(svg_code, prefix='text_svg')
    return svg_code, svg_relative_path

def process_background_extraction(image_data):
    """Extract background using gpt-4o-mini vision to analyze and recreate background"""
    if not PARALLEL_FEATURES_AVAILABLE:
        raise NotImplementedError("Parallel features not available - missing dependencies")
    
    logger.info("Starting background extraction using gpt-4o-mini vision...")
    
    try:
        # Base64-encode the PNG image for vision analysis
        img_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # First, analyze the background with gpt-4o-mini vision
        system_prompt = """You are an expert image background extractor. Your task is to analyze the provided image and extract ONLY the core background colors and gradients, ignoring all other elements.\n\nIMPORTANT: Your goal is to generate a background image that matches the background of the provided image as closely as possible. Do not add, invent, or hallucinate any new elements, shapes, icons, or decorations. The generated background should be pixel-perfect or as visually identical as possible to the background of the input image, including all colors, gradients, and patterns, but must not include any text, icons, logos, or foreground elements.\n\nFocus exclusively on:\n- The main background color(s)\n- Any color gradients in the background\n- Basic background patterns if present\n- Overall background tone and feel\n\nDO NOT include or describe:\n- Any text elements\n- Icons or logos\n- Graphical shapes or designs\n- Borders or frames\n- Decorative elements\n\nProvide a minimal description focused purely on recreating the base background colors/gradients, and ensure the generated background is as close as possible to the original image's background."""

        user_content = [
            {"type": "text", "text": "Analyze this image and describe only the background elements in detail, ignoring all text and main graphic elements."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
        
        # Call gpt-4o-mini for background analysis
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 1,
            "max_tokens":  6000
        }
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
        }
        
        logger.info("Analyzing background with gpt-4o-mini vision...")
        response = requests.post(url, headers=headers, json=payload)
        analysis_data = response.json()
        
        if response.status_code != 200:
            logger.error(f"Error analyzing background: {analysis_data}")
            raise Exception("Background analysis failed")
        
        background_description = analysis_data["choices"][0]["message"]["content"]
        logger.info(f"Background analysis: {background_description[:200]}...")
        
        # Now use GPT Image-1 to generate background based on description
        background_generation_prompt = f"Create a clean background based on this description: {background_description}. The background should be 1024x1024 pixels, without any text, icons, or graphic elements - just the background pattern, colors, and textures described."
        
        logger.info("Generating background with GPT Image-1...")
        response = openai.images.generate(
            model=GPT_IMAGE_MODEL,
            prompt=background_generation_prompt,
            size="1024x1024",
            quality="low"
        )
        
        # Get the background-only image
        background_base64 = response.data[0].b64_json if hasattr(response.data[0], 'b64_json') else response.data[0].url
        
        # Handle URL vs base64 response
        if background_base64.startswith('http'):
            # If we got a URL, download the image
            import urllib.request
            temp_path = f"temp_bg_{uuid.uuid4().hex[:8]}.png"
            urllib.request.urlretrieve(background_base64, temp_path)
            # Read it back to get base64
            with open(temp_path, 'rb') as f:
                background_bytes = f.read()
            background_base64 = base64.b64encode(background_bytes).decode('utf-8')
            os.remove(temp_path)  # Clean up temp file
        
        # Create session and save to unified storage
        session_id = f"bg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        background_filename, background_relative_path, _ = save_image(background_base64, prefix="background", session_id=session_id)
        background_full_path = os.path.join(IMAGES_DIR, background_relative_path)
        
        # Create public URL for the background
        background_public_url = get_public_image_url(background_relative_path)
        
        logger.info(f"Background extracted and saved: {background_filename}")
        logger.info(f"Background public URL: {background_public_url}")
        return background_base64, background_filename, background_full_path, background_public_url
        
    except Exception as e:
        logger.error(f"Error in background extraction: {str(e)}")
        # Fallback: return original image as background using unified storage
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        session_id = f"bgfb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        fallback_filename, fallback_relative_path, _ = save_image(image_base64, prefix="background_fallback", session_id=session_id)
        fallback_full_path = os.path.join(IMAGES_DIR, fallback_relative_path)
        
        # Create public URL for fallback
        fallback_public_url = get_public_image_url(fallback_relative_path)
        
        logger.warning("Using original image as background fallback")
        logger.info(f"Fallback background public URL: {fallback_public_url}")
        return image_base64, fallback_filename, fallback_full_path, fallback_public_url

def process_clean_svg(image_data):
    """Process text AND background removal and convert to clean SVG (elements only)"""
    if not PARALLEL_FEATURES_AVAILABLE:
        raise NotImplementedError("Parallel features not available - missing dependencies")
    
    # Save the original image bytes to a temporary PNG file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_input_path = f"temp_input_{timestamp}_{uuid.uuid4()}.png"
    with open(temp_input_path, "wb") as f:
        f.write(image_data)

    try:
        # Remove text from the image using remove_text_simple
        # This will be our elements-only image (text removed, but background may still be present)
        # V-Tracer will handle the vectorization and naturally isolate the vector elements
        final_edited_path = remove_text_simple.remove_text(temp_input_path)
        logger.info("Text removed from image, proceeding with V-Tracer for element isolation...")

        # Preprocess for vtracer: use gradient-friendly preprocessing
        pre_vtracer_path = preprocess_for_gradients(final_edited_path)

        output_svg_path = os.path.join(IMAGES_DIR, f"elements_{timestamp}_{uuid.uuid4().hex[:8]}.svg")
        vtracer.convert_image_to_svg_py(
            pre_vtracer_path,
            output_svg_path,
            # Clustering settings
            colormode='color',         # Color mode (not B/W)
            hierarchical='stacked',    # Stacked mode for layering
            
            # Filter settings
            filter_speckle=4,         # Filter Speckle (Cleaner) = 4
            
            # Color and precision settings
            color_precision=6,        # Color Precision (More accurate) = 6
            layer_difference=65,      # Gradient Step (Less layers) = 65
            
            # Curve fitting settings
            mode='spline',            # Curve Fitting = Spline
            corner_threshold=60,      # Corner Threshold (Smoother) = 60
            length_threshold=4.0,     # Segment Length (More coarse) = 4
            splice_threshold=45,      # Splice Threshold (Less accurate) = 45
            
            # Additional optimization settings
            max_iterations=10,        # Standard iterations for quality
            path_precision=3          # Good balance of precision and file size
        )

        # Read the generated SVG
        with open(output_svg_path, 'r', encoding='utf-8') as f:
            svg_code = f.read()

        return svg_code, os.path.basename(output_svg_path), final_edited_path
    finally:
        # Clean up temporary files
        for temp_file in [temp_input_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def preprocess_for_vtracer(input_path):
    """Improved preprocessing for VTracer to enhance element separation and detail"""
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
    import numpy as np
    import cv2

    img = Image.open(input_path).convert('RGB')
    # Less aggressive posterization
    img = ImageOps.posterize(img, 4)  # 4 bits = 16 colors per channel
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    # Optional: Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    # Convert to numpy for OpenCV denoising
    img_np = np.array(img)
    img_np = cv2.fastNlMeansDenoisingColored(img_np, None, 10, 10, 7, 21)
    # Optional: Slight blur to reduce noise
    img_np = cv2.GaussianBlur(img_np, (3, 3), 0.5)
    pre_path = input_path.replace('.png', '_pre_vtracer.png')
    Image.fromarray(img_np).save(pre_path)
    return pre_path
    
def preprocess_for_gradients(input_path):
    """Minimal preprocessing specifically for gradient-rich images"""
    # Load image
    img = Image.open(input_path).convert('RGB')
    
    # Very light posterization to maintain gradient smoothness
    img = ImageOps.posterize(img, 6)  # ✅ 6 bits = 64 colors per channel
    
    # Convert to numpy array for OpenCV
    img_np = np.array(img)
    
    # Very light gaussian blur to smooth small artifacts only
    img_np = cv2.GaussianBlur(img_np, (3, 3), 0.5)
    
    # Save preprocessed image
    pre_path = input_path.replace('.png', '_gradient_pre.png')
    Image.fromarray(img_np).save(pre_path)
    return pre_path



def ai_combine_svgs(text_svg_code, elements_svg_code, background_image_url=None):
    """OPTIMIZED AI-powered combination of text, elements SVGs and background image"""
    logger.info("Using OPTIMIZED AI to combine 3-layer SVG...")
    
    # Simplified system prompt for faster processing
    system_prompt = """You are an SVG combiner. Create a single SVG with 3 layers:
1. Background: <image href="URL" x="0" y="0" width="1080" height="1080"/>
2. Elements: Graphics from elements SVG
3. Text: Text from text SVG

Structure:
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080" width="1080" height="1080">
  <g id="background-layer">[background]</g>
  <g id="elements-layer">[elements]</g>
  <g id="text-layer">[text]</g>
</svg>

Return ONLY the SVG code."""

    # Truncate SVG codes for faster processing
    max_svg_length = 15000
    if len(text_svg_code) > max_svg_length:
        text_svg_code = text_svg_code[:max_svg_length] + "</svg>"
    if len(elements_svg_code) > max_svg_length:
        elements_svg_code = elements_svg_code[:max_svg_length] + "</svg>"

    user_prompt = f"""Combine these into a 3-layer SVG:

TEXT SVG:
{text_svg_code}

ELEMENTS SVG:
{elements_svg_code}

Background URL: {background_image_url}

Return the combined SVG."""

    # Use optimized call with faster model
    ai_response = optimized_openai_call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="gpt-4.1-mini",  # Faster model
        max_tokens=32000,      # Reduced tokens
        temperature=1      # More deterministic
    )
    
    if ai_response:
        logger.info("AI successfully combined 3-layer SVG - optimized version")
        return ai_response
    else:
        logger.info("AI call failed, using fallback")
        return simple_combine_svgs_fallback(text_svg_code, elements_svg_code, background_image_url)

def simple_combine_svgs_fallback(text_svg_code, elements_svg_code, background_image_url=None):
    """Fallback simple combination method with improved error handling for 3 layers"""
    try:
        logger.info("Using fallback 3-layer SVG combination method")
        
        # Validate inputs
        if not text_svg_code or not elements_svg_code:
            logger.warning("Missing SVG input data for fallback")
            return elements_svg_code if elements_svg_code else text_svg_code
        
        # Extract content from both SVGs
        text_match = re.search(r'<svg[^>]*>(.*?)</svg>', text_svg_code, re.DOTALL | re.IGNORECASE)
        elements_match = re.search(r'<svg[^>]*>(.*?)</svg>', elements_svg_code, re.DOTALL | re.IGNORECASE)
        
        if not text_match:
            logger.warning("Could not extract text SVG content, using entire text SVG")
            text_content = text_svg_code
        else:
            text_content = text_match.group(1).strip()
            
        if not elements_match:
            logger.warning("Could not extract elements SVG content, using entire elements SVG")
            elements_content = elements_svg_code
        else:
            elements_content = elements_match.group(1).strip()
        
        # Prepare background layer using the provided URL
        background_layer = ""
        if background_image_url:
            background_layer = f'''<image href="{background_image_url}" x="0" y="0" width="1080" height="1080" preserveAspectRatio="xMidYMid slice"/>'''
        
        # Create combined SVG with 3-layer structure
        combined_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080" width="1080" height="1080">
  <defs>
    <!-- Include any definitions from original SVGs -->
  </defs>
  <g id="background-layer">
    {background_layer}
  </g>
  <g id="elements-layer" opacity="0.9">
    {elements_content}
  </g>
  <g id="text-layer">
    {text_content}
  </g>
</svg>'''
        
        logger.info("Fallback 3-layer SVG combination completed successfully")
        return combined_svg
        
    except Exception as e:
        logger.error(f"Error in fallback 3-layer SVG combination: {str(e)}")
        logger.error(f"Text SVG preview: {text_svg_code[:100] if text_svg_code else 'None'}...")
        logger.error(f"Elements SVG preview: {elements_svg_code[:100] if elements_svg_code else 'None'}...")
        logger.error(f"Background URL: {background_image_url}")
        # Return the elements SVG as the safest fallback
        return elements_svg_code if elements_svg_code else text_svg_code

@app.route('/api/generate-parallel-svg', methods=['POST'])
def generate_parallel_svg():
    """Enhanced Pipeline: Stages 1-6 image gen, then triple parallel Stage 7: Text SVG, Background Extraction, and Elements SVG generation"""
    try:
        if not PARALLEL_FEATURES_AVAILABLE:
            return jsonify({
                "error": "Parallel SVG features not available",
                "message": "Missing required dependencies (vtracer, remove_text_simple, etc.)",
                "fallback": "Please use /api/generate-svg endpoint instead"
            }), 501

        data = request.json or {}
        user_input = data.get('prompt', '')

        if not user_input:
            return jsonify({'error': 'No prompt provided'}), 400

        logger.info('=== PARALLEL SVG PIPELINE START ===')

        # Optimized Prompt Enhancement
        logger.info('Optimized Prompt Enhancement for GPT Image-1')
        enhanced_prompt = enhance_prompt_for_gpt_image(user_input)
        logger.info('Enhanced prompt generated')

        # Create unified session for parallel pipeline BEFORE image generation
        parallel_session_id = f"parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        logger.info(f'Created parallel session: {parallel_session_id}')
        
        # Stage 6: Image Generation via GPT-Image using enhanced prompt
        logger.info('Stage 6: Image Generation via GPT-Image with enhanced prompt')
        logger.debug(f'Enhanced prompt: {enhanced_prompt[:200]}...')
        image_base64, temp_image_filename, initial_public_url = generate_image_with_gpt(enhanced_prompt, None)
        image_data = base64.b64decode(image_base64)
        
        # Save the initial generated image to unified storage immediately
        logger.info(f'Saving initial generated image to unified storage session: {parallel_session_id}')
        initial_image_filename, initial_image_relative_path, _ = save_image(image_base64, prefix="initial_generated", session_id=parallel_session_id)

        # Stage 7: Triple Parallel Processing
        logger.info('Stage 7: Triple Parallel Processing - Text SVG, Background Extraction, and Elements SVG')
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all three tasks
            ocr_future = executor.submit(process_ocr_svg, image_data)
            background_future = executor.submit(process_background_extraction, image_data)
            elements_future = executor.submit(process_clean_svg, image_data)
            
            # Get results
            text_svg_code, text_svg_path = ocr_future.result()
            background_base64, background_filename, background_path, background_public_url = background_future.result()
            elements_svg_code, elements_svg_path, edited_png_path = elements_future.result()

        # Session already created above

        # Base URL for unified storage
        base_url = '/static/images/sessions'

        # Save all files directly to unified session folder using session_id
        try:
            # Initial image already saved above as 'initial_generated'
            
            # Save background image to unified storage
            _, background_relative_path, _ = save_image(background_base64, prefix="background", session_id=parallel_session_id)
            
            # Save text SVG to unified storage
            _, text_svg_relative_path, _ = save_svg(text_svg_code, prefix="text_svg", session_id=parallel_session_id)
            
            # Save elements SVG to unified storage
            _, elements_svg_relative_path, _ = save_svg(elements_svg_code, prefix="elements_svg", session_id=parallel_session_id)
            
            # Save elements PNG to unified storage
            with open(edited_png_path, 'rb') as f:
                edited_png_data = f.read()
            edited_png_base64 = base64.b64encode(edited_png_data).decode('utf-8')
            _, edited_png_relative_path, _ = save_image(edited_png_base64, prefix="elements_png", format="PNG", session_id=parallel_session_id)

        except Exception as e:
            logger.warning(f"Error saving files to unified storage: {e}")
            # Fallback to using original paths
            initial_image_relative_path = temp_image_filename
            background_relative_path = background_filename
            text_svg_relative_path = text_svg_path
            elements_svg_relative_path = elements_svg_path
            edited_png_relative_path = os.path.basename(edited_png_path)

        # Construct PUBLIC URLs for serving using unified storage paths  
        initial_image_public_url = get_public_image_url(f"sessions/{parallel_session_id}/{initial_image_filename}")
        text_svg_public_url = get_public_image_url(f"sessions/{parallel_session_id}/{os.path.basename(text_svg_relative_path)}")
        # Use the background_public_url we already have from the background extraction
        elements_svg_public_url = get_public_image_url(f"sessions/{parallel_session_id}/{os.path.basename(elements_svg_relative_path)}")
        edited_png_public_url = get_public_image_url(f"sessions/{parallel_session_id}/{os.path.basename(edited_png_relative_path)}")

        # Log all public URLs for debugging
        logger.info(f"Generated public URLs:")
        logger.info(f"  Initial image: {initial_image_public_url}")
        logger.info(f"  Background: {background_public_url}")
        logger.info(f"  Text SVG: {text_svg_public_url}")
        logger.info(f"  Elements SVG: {elements_svg_public_url}")
        logger.info(f"  Elements PNG: {edited_png_public_url}")

        # Also create relative URLs for backward compatibility in response
        initial_image_url = f"{base_url}/{parallel_session_id}/{initial_image_filename}"
        text_svg_url = f"{base_url}/{parallel_session_id}/{os.path.basename(text_svg_relative_path)}"
        background_url = f"{base_url}/{parallel_session_id}/{os.path.basename(background_relative_path)}"
        elements_svg_url = f"{base_url}/{parallel_session_id}/{os.path.basename(elements_svg_relative_path)}"
        edited_png_url = f"{base_url}/{parallel_session_id}/{os.path.basename(edited_png_relative_path)}"

        # Stage 8: AI-Powered 3-Layer SVG Combination using PUBLIC URLs for proper embedding
        logger.info('Stage 8: AI-Powered 3-Layer SVG Combination using gpt-4o-mini with PUBLIC background URL')
        logger.info(f'Using public background URL for SVG combination: {background_public_url}')
        combined_svg_code = ai_combine_svgs(text_svg_code, elements_svg_code, background_public_url)
        
        # Validate the combined SVG
        if not combined_svg_code or not combined_svg_code.strip():
            logger.error("Combined SVG is empty, using fallback with public URL")
            combined_svg_code = simple_combine_svgs_fallback(text_svg_code, elements_svg_code, background_public_url)
        
        # Ensure the SVG is well-formed
        if not combined_svg_code.strip().startswith('<svg'):
            logger.warning("Combined SVG doesn't start with <svg, using fallback with public URL")
            combined_svg_code = simple_combine_svgs_fallback(text_svg_code, elements_svg_code, background_public_url)
        
        # Stage 9: Post-process SVG with OpenAI to remove first path in elements-layer
        logger.info('Stage 9: Post-processing SVG to remove first path in elements-layer using gpt-4o-mini')
        combined_svg_code = post_process_svg_remove_first_path(combined_svg_code)
        
        # Save combined SVG to unified storage
        combined_svg_filename, combined_svg_relative_path, _ = save_svg(combined_svg_code, prefix="combined_svg", session_id=parallel_session_id)
        
        logger.info(f"Combined SVG saved successfully: {combined_svg_filename}")

        # Final combined SVG URL
        combined_svg_url = f"{base_url}/{parallel_session_id}/{combined_svg_filename}"

        # Add AI-based SVG correction step
        force_svg_fix = data.get('force_svg_fix', False)
        svg_fixed = False
        max_svg_fix_attempts = 2
        svg_fix_attempts = 0
        elements_svg_code_fixed = elements_svg_code
        while svg_fix_attempts < max_svg_fix_attempts:
            # Heuristic: If force_svg_fix is set, or SVG is too small/empty, or user wants retry
            if force_svg_fix or len(elements_svg_code_fixed) < 500 or '<path' not in elements_svg_code_fixed:
                logger.info(f"Attempting AI-based SVG correction (attempt {svg_fix_attempts+1})...")
                elements_svg_code_fixed = ai_fix_svg_with_png(elements_svg_code_fixed, edited_png_public_url, user_input)
                svg_fix_attempts += 1
                # If SVG is now much larger and contains paths, break
                if len(elements_svg_code_fixed) > 500 and '<path' in elements_svg_code_fixed:
                    svg_fixed = True
                    break
            else:
                break
        # If still not usable, try to regenerate from scratch (fallback)
        if not svg_fixed and (force_svg_fix or len(elements_svg_code_fixed) < 500 or '<path' not in elements_svg_code_fixed):
            logger.warning("SVG still not usable after AI fix, regenerating elements SVG from scratch...")
            # Regenerate using process_clean_svg again
            elements_svg_code_fixed, elements_svg_path, edited_png_path = process_clean_svg(image_data)
            # Save new SVG and PNG
            _, elements_svg_relative_path, _ = save_svg(elements_svg_code_fixed, prefix="elements_svg_regen", session_id=parallel_session_id)
            with open(edited_png_path, 'rb') as f:
                edited_png_data = f.read()
            edited_png_base64 = base64.b64encode(edited_png_data).decode('utf-8')
            _, edited_png_relative_path, _ = save_image(edited_png_base64, prefix="elements_png_regen", format="PNG", session_id=parallel_session_id)
            elements_svg_public_url = get_public_image_url(f"sessions/{parallel_session_id}/{os.path.basename(elements_svg_relative_path)}")
            edited_png_public_url = get_public_image_url(f"sessions/{parallel_session_id}/{os.path.basename(edited_png_relative_path)}")
        else:
            # Save the fixed SVG if it was changed
            if svg_fixed:
                _, elements_svg_relative_path, _ = save_svg(elements_svg_code_fixed, prefix="elements_svg_fixed", session_id=parallel_session_id)
                elements_svg_public_url = get_public_image_url(f"sessions/{parallel_session_id}/{os.path.basename(elements_svg_relative_path)}")
        # Use elements_svg_code_fixed for all downstream steps
        elements_svg_code = elements_svg_code_fixed

        return jsonify({
            'original_prompt': user_input,
            'initial_image': {
                'url': initial_image_url,
                'public_url': initial_image_public_url,
                'path': initial_image_relative_path,
                'filename': initial_image_filename
            },
            'background': {
                'base64': background_base64,
                'path': f"sessions/{parallel_session_id}/{os.path.basename(background_relative_path)}",
                'url': background_url,
                'public_url': background_public_url
            },
            'elements_png': {
                'path': f"sessions/{parallel_session_id}/{os.path.basename(edited_png_relative_path)}",
                'url': edited_png_url,
                'public_url': edited_png_public_url
            },
            'text_svg': {
                'code': text_svg_code,
                'path': f"sessions/{parallel_session_id}/{os.path.basename(text_svg_relative_path)}",
                'url': text_svg_url,
                'public_url': text_svg_public_url
            },
            'elements_svg': {
                'code': elements_svg_code,
                'path': f"sessions/{parallel_session_id}/{os.path.basename(elements_svg_relative_path)}",
                'url': elements_svg_url,
                'public_url': elements_svg_public_url
            },
            'combined_svg': {
                'code': combined_svg_code,
                'path': combined_svg_relative_path,
                'url': combined_svg_url,
                'public_url': get_public_image_url(combined_svg_relative_path)
            },
            'session_id': parallel_session_id,
            'stage': 8,
            'note': 'SVG now uses public URLs for proper image embedding'
        })

    except Exception as e:
        logger.error(f"Error in generate_parallel_svg: {str(e)}")
        return jsonify({"error": str(e)}), 500

def process_image_with_gpt_image(image_url, session_id):
    """Process main image through GPT Image-1 to remove text and background"""
    try:
        # Create prompt to remove text and background, keeping only elements
        removal_prompt = """Remove ALL text and background from this image completely. 
        Keep only the main visual elements/objects/graphics. 
        Make the background completely transparent or white. 
        Remove any text, labels, watermarks, or written content entirely. 
        The result should show only the pure visual elements without any text or background patterns."""
        
        logger.info(f"Processing image with GPT Image-1: {image_url}")
        logger.info(f"Removal prompt: {removal_prompt}")
        
        # Make request to GPT Image-1 API
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY_SVG}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GPT_IMAGE_MODEL,
            "prompt": removal_prompt,
            "image": image_url,
            "size": "1024x1024",
            "quality": "low",
            "n": 1
        }
        
        response = requests.post(
            "https://api.openai.com/v1/images/edits",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"GPT Image-1 API error: {response.status_code} - {response.text}")
            return {
                "success": False,
                "error": f"GPT Image-1 API error: {response.status_code}"
            }
        
        result = response.json()
        
        if not result.get('data') or len(result['data']) == 0:
            return {
                "success": False,
                "error": "No image generated by GPT Image-1"
            }
        
        # Download the generated clean image
        clean_image_url = result['data'][0]['url']
        
        # Save the clean image locally
        image_response = requests.get(clean_image_url, timeout=30)
        if image_response.status_code != 200:
            return {
                "success": False,
                "error": "Failed to download clean image"
            }
        
        # Save to session directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"clean_meat_ratio_{timestamp}_{uuid.uuid4().hex[:8]}.png"
        session_dir = os.path.join(UNIFIED_STORAGE_DIR, session_id)
        image_path = os.path.join(session_dir, image_filename)
        
        with open(image_path, 'wb') as f:
            f.write(image_response.content)
        
        logger.info(f"Clean image saved: {image_path}")
        
        return {
            "success": True,
            "image_path": image_path,
            "image_url": clean_image_url
        }
        
    except Exception as e:
        logger.error(f"Error processing image with GPT Image-1: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def convert_clean_image_to_svg(image_path, session_id):
    """Convert clean image to SVG using vtracer"""
    try:
        if not PARALLEL_FEATURES_AVAILABLE:
            return {
                "success": False,
                "error": "vtracer not available"
            }
        
        logger.info(f"Converting image to SVG: {image_path}")
        
        # Use vtracer to convert PNG to SVG
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        svg_filename = f"meat_ratio_svg_{timestamp}_{uuid.uuid4().hex[:8]}.svg"
        session_dir = os.path.join(UNIFIED_STORAGE_DIR, session_id)
        svg_path = os.path.join(session_dir, svg_filename)
        
        # vtracer conversion
        vtracer.convert_image_to_svg_py(
            input_path=image_path,
            output_path=svg_path,
            colormode='color',
            hierarchical='stacked',
            mode='polygon',              # Use polygon mode for smoother, simpler shapes
            filter_speckle=16,           # Remove even more small artifacts
            color_precision=1,           # Merge all similar colors
            layer_difference=64,        # Merge all similar color layers
            corner_threshold=120,        # Only keep very sharp corners
            length_threshold=16.0,       # Ignore small paths
            max_iterations=2,            # Less detail, more smoothing
            splice_threshold=90,         # Merge curves aggressively
            path_precision=0             # Maximum smoothing
        )
        
        if not os.path.exists(svg_path):
            return {
                "success": False,
                "error": "SVG file was not created"
            }
        
        logger.info(f"SVG generated: {svg_path}")
        
        return {
            "success": True,
            "svg_path": svg_path
        }
        
    except Exception as e:
        logger.error(f"Error converting image to SVG: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/api/generate-meat-ratio-svg', methods=['POST'])
def generate_meat_ratio_svg():
    """Generate SVG from meat ratio by processing main image through GPT Image-1 to remove text/background"""
    try:
        data = request.json
        main_image_url = data.get('main_image_url', '')
        
        if not main_image_url:
            return jsonify({"error": "No main image URL provided"}), 400
        
        logger.info("="*80)
        logger.info(f"Starting meat ratio SVG generation from image: {main_image_url}")
        logger.info("="*80)
        
        # Create session ID for this generation
        session_id = f"meat_ratio_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        session_dir = os.path.join(UNIFIED_STORAGE_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        logger.info(f"Created session directory: {session_dir}")
        
        # Step 1: Process main image through GPT Image-1 to remove text and background
        logger.info("\n[STEP 1: Processing main image through GPT Image-1]")
        logger.info("-"*50)
        
        clean_image_response = process_image_with_gpt_image(main_image_url, session_id)
        
        if not clean_image_response.get('success'):
            return jsonify({
                "error": "Failed to process image through GPT Image-1",
                "details": clean_image_response.get('error', 'Unknown error')
            }), 500
        
        clean_image_path = clean_image_response['image_path']
        logger.info(f"Clean image generated: {clean_image_path}")
        
        # Step 2: Convert clean image to SVG
        logger.info("\n[STEP 2: Converting clean image to SVG]")
        logger.info("-"*50)
        
        if not PARALLEL_FEATURES_AVAILABLE:
            logger.warning("vtracer not available, cannot convert to SVG")
            return jsonify({
                "warning": "SVG conversion not available",
                "clean_image_url": f"/static/images/sessions/{session_id}/{os.path.basename(clean_image_path)}"
            }), 200
        
        svg_response = convert_clean_image_to_svg(clean_image_path, session_id)
        
        if not svg_response.get('success'):
            logger.warning(f"SVG conversion failed: {svg_response.get('error')}")
            return jsonify({
                "warning": "SVG conversion failed, returning clean image only",
                "clean_image_url": f"/static/images/sessions/{session_id}/{os.path.basename(clean_image_path)}",
                "error": svg_response.get('error')
            }), 200
        
        svg_path = svg_response['svg_path']
        logger.info(f"SVG generated: {svg_path}")
        
        # Return successful response
        return jsonify({
            "success": True,
            "session_id": session_id,
            "clean_image_url": f"/static/images/sessions/{session_id}/{os.path.basename(clean_image_path)}",
            "svg_url": f"/static/images/sessions/{session_id}/{os.path.basename(svg_path)}",
            "message": "Meat ratio SVG generated successfully"
        })
        
    except Exception as e:
        logger.error(f"Meat ratio SVG generation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/test-enhancement', methods=['POST'])
def test_enhancement():
    """Test endpoint to validate the enhanced prompt pipeline"""
    try:
        data = request.json or {}
        user_input = data.get('prompt', '')
        
        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400
        
        logger.info(f"Testing enhancement pipeline with: {user_input}")
        
        # Optimized GPT Image-1 enhancement
        image_enhanced = enhance_prompt_for_gpt_image(user_input)
        
        return jsonify({
            "original": user_input,
            "image_enhanced": image_enhanced,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error in test enhancement: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/test-vtracer', methods=['POST'])
def test_vtracer():
    """Test endpoint to validate VTracer functionality with improved settings"""
    try:
        data = request.json or {}
        image_base64 = data.get('image_base64', '')
        use_gradient_preprocessing = data.get('use_gradient_preprocessing', True)
        
        if not image_base64:
            return jsonify({"error": "No image data provided"}), 400
        
        if not VTRACER_AVAILABLE:
            return jsonify({"error": "VTracer not available"}), 500
        
        logger.info(f"Testing VTracer with improved settings...")
        logger.info(f"Gradient preprocessing: {use_gradient_preprocessing}")
        
        # Test VTracer with improved settings
        try:
            svg_code = generate_svg_from_image(
                image_base64, 
                "Test VTracer conversion", 
                use_preprocessing=use_gradient_preprocessing
            )
            
            return jsonify({
                "success": True,
                "svg_code": svg_code,
                "message": "VTracer conversion completed successfully",
                "settings_used": {
                    "mode": "spline",
                    "color_precision": 6,
                    "layer_difference": 16,
                    "filter_speckle": 4,
                    "gradient_preprocessing": use_gradient_preprocessing
                }
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "message": "VTracer conversion failed"
            }), 500
        
    except Exception as e:
        logger.error(f"Error in VTracer test: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add caching for OpenAI responses
@lru_cache(maxsize=100)
def get_cached_openai_response(prompt_hash, model, system_prompt_hash):
    """Cache OpenAI responses to avoid repeated API calls for similar requests"""
    return None

def generate_prompt_hash(prompt):
    """Generate a hash for prompt caching"""
    return hashlib.md5(prompt.encode()).hexdigest()

# Create a global session with connection pooling for OpenAI API calls
def create_optimized_session():
    """Create a requests session with connection pooling and retry strategy"""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=2,  # Reduced retries for faster failure
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
        backoff_factor=0.3  # Reduced backoff for speed
    )
    
    # Configure adapter with connection pooling
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=retry_strategy
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Global session for reuse
openai_session = create_optimized_session()

# Performance monitoring for OpenAI API calls
api_performance_stats = {
    'total_calls': 0,
    'total_time': 0,
    'avg_response_time': 0,
    'fastest_call': float('inf'),
    'slowest_call': 0,
    'success_rate': 0,
    'failures': 0
}

def log_api_performance(response_time, success=True):
    """Log API performance metrics"""
    global api_performance_stats
    
    api_performance_stats['total_calls'] += 1
    if success:
        api_performance_stats['total_time'] += response_time
        api_performance_stats['fastest_call'] = min(api_performance_stats['fastest_call'], response_time)
        api_performance_stats['slowest_call'] = max(api_performance_stats['slowest_call'], response_time)
        api_performance_stats['avg_response_time'] = api_performance_stats['total_time'] / (api_performance_stats['total_calls'] - api_performance_stats['failures'])
    else:
        api_performance_stats['failures'] += 1
    
    api_performance_stats['success_rate'] = ((api_performance_stats['total_calls'] - api_performance_stats['failures']) / api_performance_stats['total_calls']) * 100
    
    if api_performance_stats['total_calls'] % 5 == 0:  # Log every 5 calls
        logger.info(f"OpenAI API Performance Stats: Avg: {api_performance_stats['avg_response_time']:.2f}s, "
                   f"Range: {api_performance_stats['fastest_call']:.2f}s-{api_performance_stats['slowest_call']:.2f}s, "
                   f"Success Rate: {api_performance_stats['success_rate']:.1f}%")

# Optimized OpenAI API call with streaming and reduced payload
def optimized_openai_call(system_prompt, user_prompt, model="gpt-4o-mini", max_tokens=8000, temperature=0.3):
    """Optimized OpenAI API call with connection pooling, caching, and reduced payload"""
    import time
    
    # Generate cache keys
    prompt_hash = generate_prompt_hash(user_prompt)
    system_hash = generate_prompt_hash(system_prompt)
    
    logger.info(f"Making OPTIMIZED OpenAI call: {model}, tokens: {max_tokens}, prompt_size: {len(user_prompt)}")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}",
        "Connection": "keep-alive"  # Enable connection reuse
    }

    # Reduce prompt sizes to speed up processing
    original_prompt_size = len(user_prompt)
    if len(user_prompt) > 30000:
        logger.info(f"Truncating user prompt from {len(user_prompt)} to 30000 chars for faster processing")
        user_prompt = user_prompt[:30000] + "...\n\nPLEASE PROCESS THE ABOVE SVG CONTENT."
    
    if len(system_prompt) > 8000:
        logger.info(f"Truncating system prompt from {len(system_prompt)} to 8000 chars for faster processing")
        system_prompt = system_prompt[:8000] + "..."

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    start_time = time.time()
    try:
        # Use the global session with connection pooling
        response = openai_session.post(url, headers=headers, json=payload, timeout=80)
        api_response_time = time.time() - start_time
        
        # Log performance
        log_api_performance(api_response_time, success=True)
        
        logger.info(f"[SUCCESS] OPTIMIZED OpenAI API response in {api_response_time:.2f}s (was {original_prompt_size} chars)")

        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            log_api_performance(api_response_time, success=False)
            return None

        response_data = response.json()
        ai_response = response_data["choices"][0]["message"]["content"]
        
        return ai_response.strip()
            
    except Exception as e:
        api_response_time = time.time() - start_time
        log_api_performance(api_response_time, success=False)
        logger.error(f"[ERROR] Error in optimized OpenAI call after {api_response_time:.2f}s: {str(e)}")
        return None

# Add endpoint to check API performance stats
@app.route('/api/performance-stats', methods=['GET'])
def get_performance_stats():
    """Get AI API performance statistics"""
    return jsonify({
        'ai_api_stats': api_performance_stats,
        'optimization_status': 'ACTIVE - Using Google Gemini-2.5-flash via OpenRouter',
        'current_model': 'google/gemini-2.5-flash',
        'api_provider': 'OpenRouter',
        'optimizations_applied': [
            'Google Gemini-2.5-flash (ultra-fast model)',
            'OpenRouter API for better performance',
            'Reduced prompt sizes (30K/8K limits)',
            'Reduced token limits (8K/6K)',
            'Lower temperature (0.1) for determinism',
            'Regex-first post-processing (instant)',
            'Aggressive content reduction',
            'Smart fallback strategies'
        ]
    })



def simple_combine_svgs_fallback(text_svg_code, elements_svg_code, background_image_url=None):
    """Fallback simple combination method with improved error handling for 3 layers"""
    try:
        logger.info("Using fallback 3-layer SVG combination method")
        
        # Validate inputs
        if not text_svg_code or not elements_svg_code:
            logger.warning("Missing SVG input data for fallback")
            return elements_svg_code if elements_svg_code else text_svg_code
        
        # Extract content from both SVGs
        text_match = re.search(r'<svg[^>]*>(.*?)</svg>', text_svg_code, re.DOTALL | re.IGNORECASE)
        elements_match = re.search(r'<svg[^>]*>(.*?)</svg>', elements_svg_code, re.DOTALL | re.IGNORECASE)
        
        if not text_match:
            logger.warning("Could not extract text SVG content, using entire text SVG")
            text_content = text_svg_code
        else:
            text_content = text_match.group(1).strip()
            
        if not elements_match:
            logger.warning("Could not extract elements SVG content, using entire elements SVG")
            elements_content = elements_svg_code
        else:
            elements_content = elements_match.group(1).strip()
        
        # Prepare background layer using the provided URL
        background_layer = ""
        if background_image_url:
            background_layer = f'''<image href="{background_image_url}" x="0" y="0" width="1080" height="1080" preserveAspectRatio="xMidYMid slice"/>'''
        
        # Create combined SVG with 3-layer structure
        combined_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080" width="1080" height="1080">
  <defs>
    <!-- Include any definitions from original SVGs -->
  </defs>
  <g id="background-layer">
    {background_layer}
  </g>
  <g id="elements-layer" opacity="0.9">
    {elements_content}
  </g>
  <g id="text-layer">
    {text_content}
  </g>
</svg>'''
        
        logger.info("Fallback 3-layer SVG combination completed successfully")
        return combined_svg
        
    except Exception as e:
        logger.error(f"Error in fallback 3-layer SVG combination: {str(e)}")
        logger.error(f"Text SVG preview: {text_svg_code[:100] if text_svg_code else 'None'}...")
        logger.error(f"Elements SVG preview: {elements_svg_code[:100] if elements_svg_code else 'None'}...")
        logger.error(f"Background URL: {background_image_url}")
        # Return the elements SVG as the safest fallback
        return elements_svg_code if elements_svg_code else text_svg_code



# Add OpenRouter configuration near the top with other API configurations
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    logger.warning("🔧 OPENROUTER_API_KEY not set. Add it to your environment variables for Google Gemini-2.5-flash support!")
    logger.warning("   Example: export OPENROUTER_API_KEY='sk-or-v1-your-key-here'")
    logger.warning("   Get your key at: https://openrouter.ai/")
    logger.warning("   Will fall back to OpenAI API for SVG operations")
else:
    logger.info("[SUCCESS] OpenRouter API key found - Google Gemini-2.5-flash enabled for ultra-fast SVG processing!")

# OpenRouter client for SVG operations
openrouter_client = None
if OPENROUTER_API_KEY:
    from openai import OpenAI
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

def optimized_openrouter_call(system_prompt, user_prompt, model="google/gemini-2.5-flash", max_tokens=8000, temperature=0.3):
    """Optimized OpenRouter API call using Google Gemini-2.5-flash"""
    import time
    
    if not openrouter_client:
        logger.error("OpenRouter client not initialized, falling back to OpenAI")
        return optimized_openai_call(system_prompt, user_prompt, "gpt-4o-mini", max_tokens, temperature)
    
    logger.info(f"Making OPTIMIZED OpenRouter call: {model}, tokens: {max_tokens}, prompt_size: {len(user_prompt)}")
    
    # Reduce prompt sizes to speed up processing
    original_prompt_size = len(user_prompt)
    if len(user_prompt) > 30000:
        logger.info(f"Truncating user prompt from {len(user_prompt)} to 30000 chars for faster processing")
        user_prompt = user_prompt[:30000] + "...\n\nPLEASE PROCESS THE ABOVE SVG CONTENT."
    
    if len(system_prompt) > 8000:
        logger.info(f"Truncating system prompt from {len(system_prompt)} to 8000 chars for faster processing")
        system_prompt = system_prompt[:8000] + "..."

    start_time = time.time()
    try:
        completion = openrouter_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://infoui.app",  # Your site URL
                "X-Title": "InfoUI SVG Generator",     # Your site name
            },
            extra_body={},
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": user_prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        api_response_time = time.time() - start_time
        
        # Log performance
        log_api_performance(api_response_time, success=True)
        
        logger.info(f"[SUCCESS] OPTIMIZED OpenRouter API response in {api_response_time:.2f}s (was {original_prompt_size} chars)")

        ai_response = completion.choices[0].message.content
        return ai_response.strip()
            
    except Exception as e:
        api_response_time = time.time() - start_time
        log_api_performance(api_response_time, success=False)
        logger.error(f"[ERROR] Error in optimized OpenRouter call after {api_response_time:.2f}s: {str(e)}")
        return None

def ai_combine_svgs(text_svg_code, elements_svg_code, background_image_url=None):
    """OPTIMIZED AI-powered combination using Google Gemini-2.5-flash via OpenRouter"""
    logger.info("Using OPTIMIZED Google Gemini-2.5-flash to combine 3-layer SVG...")
    
    # Simplified system prompt for faster processing
    system_prompt = """You are an SVG combiner. Create a single SVG with 3 layers:
1. Background: <image href="URL" x="0" y="0" width="1080" height="1080"/>
2. Elements: Graphics from elements SVG
3. Text: Text from text SVG

Structure:
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080" width="1080" height="1080">
  <g id="background-layer">[background]</g>
  <g id="elements-layer">[elements]</g>
  <g id="text-layer">[text]</g>
</svg>

Return ONLY the SVG code."""

    # Truncate SVG codes for faster processing
    max_svg_length = 15000
    if len(text_svg_code) > max_svg_length:
        text_svg_code = text_svg_code[:max_svg_length] + "</svg>"
    if len(elements_svg_code) > max_svg_length:
        elements_svg_code = elements_svg_code[:max_svg_length] + "</svg>"

    user_prompt = f"""Combine these into a 3-layer SVG:

TEXT SVG:
{text_svg_code}

ELEMENTS SVG:
{elements_svg_code}

Background URL: {background_image_url}

Return the combined SVG."""

    # Use optimized OpenRouter call with Google Gemini
    ai_response = optimized_openrouter_call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="google/gemini-2.5-flash",  # Google Gemini model
        max_tokens=32000,                  # Reduced tokens
        temperature=1                 # More deterministic
    )
    
    if ai_response:
        logger.info("Google Gemini successfully combined 3-layer SVG - optimized version")
        return ai_response
    else:
        logger.info("Gemini call failed, using fallback")
        return simple_combine_svgs_fallback(text_svg_code, elements_svg_code, background_image_url)

def simple_combine_svgs_fallback(text_svg_code, elements_svg_code, background_image_url=None):
    """Fallback simple combination method with improved error handling for 3 layers"""
    try:
        logger.info("Using fallback 3-layer SVG combination method")
        
        # Validate inputs
        if not text_svg_code or not elements_svg_code:
            logger.warning("Missing SVG input data for fallback")
            return elements_svg_code if elements_svg_code else text_svg_code
        
        # Extract content from both SVGs
        text_match = re.search(r'<svg[^>]*>(.*?)</svg>', text_svg_code, re.DOTALL | re.IGNORECASE)
        elements_match = re.search(r'<svg[^>]*>(.*?)</svg>', elements_svg_code, re.DOTALL | re.IGNORECASE)
        
        if not text_match:
            logger.warning("Could not extract text SVG content, using entire text SVG")
            text_content = text_svg_code
        else:
            text_content = text_match.group(1).strip()
            
        if not elements_match:
            logger.warning("Could not extract elements SVG content, using entire elements SVG")
            elements_content = elements_svg_code
        else:
            elements_content = elements_match.group(1).strip()
        
        # Prepare background layer using the provided URL
        background_layer = ""
        if background_image_url:
            background_layer = f'''<image href="{background_image_url}" x="0" y="0" width="1080" height="1080" preserveAspectRatio="xMidYMid slice"/>'''
        
        # Create combined SVG with 3-layer structure
        combined_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080" width="1080" height="1080">
  <defs>
    <!-- Include any definitions from original SVGs -->
  </defs>
  <g id="background-layer">
    {background_layer}
  </g>
  <g id="elements-layer" opacity="0.9">
    {elements_content}
  </g>
  <g id="text-layer">
    {text_content}
  </g>
</svg>'''
        
        logger.info("Fallback 3-layer SVG combination completed successfully")
        return combined_svg
        
    except Exception as e:
        logger.error(f"Error in fallback 3-layer SVG combination: {str(e)}")
        logger.error(f"Text SVG preview: {text_svg_code[:100] if text_svg_code else 'None'}...")
        logger.error(f"Elements SVG preview: {elements_svg_code[:100] if elements_svg_code else 'None'}...")
        logger.error(f"Background URL: {background_image_url}")
        # Return the elements SVG as the safest fallback
        return elements_svg_code if elements_svg_code else text_svg_code

def post_process_svg_remove_first_path(svg_code):
    """OPTIMIZED post-processing using Google Gemini-2.5-flash"""
    logger.info("OPTIMIZED post-processing SVG using Google Gemini-2.5-flash...")
    
    # Try regex-based approach first (much faster)
    try:
        import re
        
        # Find elements-layer group and remove first path
        pattern = r'(<g id="elements-layer"[^>]*>)(.*?)</g>'
        match = re.search(pattern, svg_code, re.DOTALL)
        
        if match:
            elements_content = match.group(2)
            # Remove first path tag
            path_pattern = r'<path[^>]*(?:/>|>.*?</path>)'
            elements_content_modified = re.sub(path_pattern, '', elements_content, count=1)
            
            # Replace in original SVG
            modified_svg = svg_code.replace(match.group(0), match.group(1) + elements_content_modified + '</g>')
            logger.info("Successfully removed first path using regex (fast method)")
            return modified_svg
        
    except Exception as e:
        logger.warning(f"Regex approach failed: {str(e)}, falling back to Gemini AI")
    
    # Fallback to Gemini AI with simplified prompt
    system_prompt = """Remove ONLY the first <path> tag inside <g id="elements-layer">. Keep everything else exactly the same. Return ONLY the SVG code."""
    
    # Truncate for faster processing
    if len(svg_code) > 25000:
        svg_code = svg_code[:25000] + "</svg>"
    
    user_prompt = f"Remove first path in elements-layer:\n{svg_code}"
    
    ai_response = optimized_openrouter_call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="google/gemini-2.5-flash",  # Google Gemini model
        max_tokens=6000,                  # Reduced tokens
        temperature=1               # More deterministic
    )
    
    if ai_response:
        logger.info("Google Gemini successfully post-processed SVG - optimized version")
        return ai_response
    else:
        logger.info("Gemini post-processing failed, returning original SVG")
        return svg_code

def ai_fix_svg_with_png(elements_svg_code, elements_png_url, prompt=None):
    """Use an AI model to fix the SVG so it matches the PNG as closely as possible, including element and text positions and sizes."""
    logger.info("Starting AI-based SVG correction using PNG reference...")
    # Use OpenAI or Gemini vision model to compare and fix SVG
    # We'll use the PNG public URL and SVG code as input
    system_prompt = (
        "You are an expert SVG repair assistant. "
        "Given an SVG code and a PNG image (URL), your job is to edit the SVG so it visually matches the PNG as closely as possible. "
        "Fix missing elements, colors, shapes, text, and details. "
        "Most importantly, ensure that the position and size of each element and all text in the SVG matches the corresponding element and text in the PNG as closely as possible. "
        "Do not just match colors and shapes, but also their exact placement, proportions, and text positioning. "
        "Return ONLY the corrected SVG code."
    )
    user_prompt = f"""
Here is the current SVG code:
```svg
{elements_svg_code}
```

Here is the PNG image (URL):
{elements_png_url}

{('Original prompt: ' + prompt) if prompt else ''}

Please return the corrected SVG code that matches the PNG as closely as possible, including element and text positions and sizes. Only output SVG code.
"""
    # Use OpenRouter Gemini if available, else fallback to OpenAI
    ai_response = optimized_openrouter_call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model="google/gemini-2.5-flash",
        max_tokens=32000,
        temperature=0.3
    )
    if ai_response and '<svg' in ai_response:
        logger.info("AI-based SVG correction successful.")
        return ai_response
    logger.warning("AI-based SVG correction failed or did not return SVG. Returning original SVG.")
    return elements_svg_code

if __name__ == '__main__':
    # Get port from environment variable (Render sets PORT=8000)
    port = int(os.getenv('PORT', 5000))
    
    # Use 0.0.0.0 for production (Render) and 127.0.0.1 for local development
    host = '0.0.0.0' if os.getenv('PORT') else '127.0.0.1'
    
    # Disable debug mode in production
    debug = not bool(os.getenv('PORT'))
    
    logger.info(f"Starting Flask application on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
