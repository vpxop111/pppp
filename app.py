from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import traceback
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
import vtracer
from concurrent.futures import ThreadPoolExecutor
import pytesseract
import numpy as np
import remove_text_simple
import png_to_svg_converter
from utils import (
    STATIC_DIR,
    IMAGES_DIR,
    save_image,
    save_svg,
    generate_image_with_gpt
)
from shared_functions import (
    check_vector_suitability,
    plan_design,
    generate_design_knowledge,
    pre_enhance_prompt,
    enhance_prompt_with_chat
)
from parallel_svg_pipeline import generate_parallel_svg_pipeline, init_parallel_pipeline

# Load environment variables
load_dotenv()

# Configure basic logging first
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Ensure logs go to stdout for Render
)
logger = logging.getLogger(__name__)

# API configuration
OPENAI_API_KEY_ENHANCER = os.getenv('OPENAI_API_KEY_ENHANCER')
OPENAI_API_KEY_SVG = os.getenv('OPENAI_API_KEY_SVG')
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_CHAT_ENDPOINT = f"{OPENAI_API_BASE}/chat/completions"

# Model names (using standard OpenAI models that are widely available)
PLANNER_MODEL = "gpt-4o-mini"
DESIGN_KNOWLEDGE_MODEL = "gpt-4o-mini"
PRE_ENHANCER_MODEL = "gpt-4o-mini"
PROMPT_ENHANCER_MODEL = "gpt-4o-mini"
GPT_IMAGE_MODEL = "dall-e-3"
SVG_GENERATOR_MODEL = "gpt-4o-mini"
CHAT_ASSISTANT_MODEL = "gpt-4o-mini"

try:
    # Initialize parallel pipeline
    vtracer_available = init_parallel_pipeline()
    if vtracer_available:
        logger.info("Parallel SVG pipeline ready with full vtracer support")
    else:
        logger.info("Parallel SVG pipeline initialized with limited functionality")
except Exception as e:
    logger.error(f"Failed to initialize parallel pipeline: {str(e)}")
    logger.error(traceback.format_exc())
    logger.warning("Continuing without parallel pipeline support")
    vtracer_available = False

app = Flask(__name__)

# Configure CORS with specific settings
CORS(app, 
     origins=[
         'http://localhost:3000', 
         'http://localhost:3001',
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

# Directory setup
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Directory for parallel pipeline outputs
PARALLEL_OUTPUTS_DIR = os.path.join(IMAGES_DIR, 'parallel')
os.makedirs(PARALLEL_OUTPUTS_DIR, exist_ok=True)

if not OPENAI_API_KEY_ENHANCER or not OPENAI_API_KEY_SVG:
    raise ValueError("OpenAI API keys must be set in environment variables")

# OpenAI client setup
openai.api_key = OPENAI_API_KEY_SVG

# Instantiate a GPT client for chat completions (parallel pipeline)
from openai import OpenAI
chat_client = OpenAI()







def enhance_prompt_for_gpt_image(user_prompt, design_context=None):
    """Enhance user prompt specifically for GPT Image-1 to create better designs"""

    # Analyze the prompt to determine design type
    prompt_lower = user_prompt.lower()

    # Design type detection
    is_poster = any(word in prompt_lower for word in ['poster', 'flyer', 'announcement', 'event', 'coming soon'])
    is_logo = any(word in prompt_lower for word in ['logo', 'brand', 'company', 'business', 'startup'])
    is_card = any(word in prompt_lower for word in ['card', 'testimonial', 'review', 'quote'])
    is_banner = any(word in prompt_lower for word in ['banner', 'header', 'cover', 'social media'])
    is_infographic = any(word in prompt_lower for word in ['infographic', 'chart', 'data', 'statistics'])

    # Base quality enhancers for GPT Image-1
    base_quality = [
        "professional design",
        "high-quality graphics",
        "clean composition",
        "modern aesthetic",
        "balanced layout",
        "crisp typography",
        "vibrant colors",
        "well-defined elements",
        "clear visual hierarchy",
        "polished finish"
    ]

    # Design-specific enhancements
    if is_poster:
        specific_enhancements = [
            "eye-catching poster design",
            "bold headline typography",
            "compelling visual focal point",
            "structured information layout",
            "attention-grabbing color scheme",
            "professional poster composition",
            "clear call-to-action placement",
            "balanced text and imagery"
        ]
    elif is_logo:
        specific_enhancements = [
            "distinctive logo design",
            "memorable brand identity",
            "scalable vector-friendly graphics",
            "simple yet impactful design",
            "professional brand aesthetics",
            "clean geometric shapes",
            "timeless design approach",
            "versatile color palette"
        ]
    elif is_card:
        specific_enhancements = [
            "elegant card design",
            "testimonial-focused layout",
            "professional presentation",
            "readable typography hierarchy",
            "trustworthy visual design",
            "clean background treatment",
            "balanced content arrangement",
            "credible aesthetic appeal"
        ]
    elif is_banner:
        specific_enhancements = [
            "dynamic banner design",
            "horizontal composition",
            "social media optimized",
            "engaging visual elements",
            "brand-consistent styling",
            "clear messaging hierarchy",
            "platform-appropriate design",
            "scroll-stopping appeal"
        ]
    elif is_infographic:
        specific_enhancements = [
            "data visualization design",
            "information hierarchy",
            "chart and graph elements",
            "educational layout",
            "statistical presentation",
            "clear data storytelling",
            "professional infographic style",
            "engaging data design"
        ]
    else:
        specific_enhancements = [
            "versatile graphic design",
            "adaptable visual style",
            "multi-purpose layout",
            "flexible design approach",
            "universal appeal",
            "broad application design"
        ]

    # Technical specifications for GPT Image-1
    technical_specs = [
        "1024x1024 resolution",
        "RGB color space",
        "high contrast elements",
        "clear edge definition",
        "optimal text readability",
        "vector-conversion friendly",
        "clean background separation",
        "distinct element boundaries"
    ]

    # Build enhanced prompt
    enhanced_parts = []

    # Add original request
    enhanced_parts.append(f"Create: {user_prompt}")

    # Add design context if provided
    if design_context:
        enhanced_parts.append(f"Context: {design_context[:200]}...")

    # Add design type specific enhancements
    enhanced_parts.append(f"Style: {', '.join(specific_enhancements[:4])}")

    # Add quality requirements
    enhanced_parts.append(f"Quality: {', '.join(base_quality[:6])}")

    # Add technical requirements
    enhanced_parts.append(f"Technical: {', '.join(technical_specs[:4])}")

    # Combine into final prompt
    final_prompt = " | ".join(enhanced_parts)

    # Ensure prompt isn't too long (GPT Image-1 has limits)
    if len(final_prompt) > 1000:
        final_prompt = final_prompt[:1000] + "..."

    return final_prompt

def generate_image_with_gpt(enhanced_prompt, design_context=None):
    """Generate image using GPT Image-1 model with enhanced prompting"""
    try:
        logger.info("Generating image with GPT Image-1")

        # Enhance the prompt specifically for GPT Image-1
        optimized_prompt = enhance_prompt_for_gpt_image(enhanced_prompt, design_context)
        logger.info(f"Optimized prompt: {optimized_prompt[:200]}...")

        response = openai.images.generate(
            model=GPT_IMAGE_MODEL,
            prompt=optimized_prompt,
            size="1024x1024",
            quality="standard",   # Using standard quality as supported by OpenAI API
            response_format="b64_json"  # Explicitly request base64 format
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
        raise

def generate_svg_from_image(image_base64, enhanced_prompt):
    """Generate SVG code from image using vtracer"""
    logger.info("SVG generation from image requested")
    
    try:
        import vtracer
        import tempfile
        import os
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_file.write(image_data)
            tmp_input_path = tmp_file.name
        
        # Generate temporary output path
        tmp_output_path = tmp_input_path.replace('.png', '.svg')
        
        # Convert image to SVG using vtracer with optimized settings
        vtracer.convert_image_to_svg_py(
            tmp_input_path,
            tmp_output_path,
            colormode='color',
            hierarchical='stacked',
            mode='spline',
            filter_speckle=4,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3
        )
        
        # Read the generated SVG
        with open(tmp_output_path, 'r') as f:
            svg_code = f.read()
        
        # Clean up temporary files
        os.unlink(tmp_input_path)
        os.unlink(tmp_output_path)
        
        logger.info("SVG generated successfully from image")
        return svg_code
        
    except ImportError:
        logger.error("vtracer not available - falling back to alternative method")
        raise NotImplementedError("vtracer not available for image-to-SVG conversion")
    except Exception as e:
        logger.error(f"Error in image-to-SVG conversion: {str(e)}")
        raise

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

def convert_svg_to_png(svg_code):
    """Convert SVG code to PNG and save both files"""
    try:
        # Save SVG file
        svg_filename = save_svg(svg_code)
        
        # Convert to PNG using cairosvg
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
        
        # Save PNG file
        png_filename = save_image(
            base64.b64encode(png_data).decode('utf-8'),
            prefix="converted_svg",
            format="PNG"
        )
        
        return svg_filename, png_filename
    except Exception as e:
        logger.error(f"Error in SVG to PNG conversion: {str(e)}")
        raise

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """Serve images from the images directory"""
    return send_from_directory(IMAGES_DIR, filename)

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

        # Stage 1: Check if prompt is suitable for SVG vector graphics
        logger.info("\n[STAGE 1: Vector Suitability Check]")
        logger.info("-"*50)
        logger.info("Checking if design is suitable for SVG format...")
        vector_suitability = check_vector_suitability(user_input)
        logger.info("Vector suitability check complete")
        logger.info(f"Result: {'Suitable' if not vector_suitability.get('not_suitable', False) else 'Not Suitable'}")
        
        if vector_suitability.get('not_suitable', False):
            logger.warning("Design request not suitable for SVG format")
            return jsonify({
                "error": "Not suitable for SVG",
                "guidance": vector_suitability.get('guidance', "Your request may not be ideal for SVG vector graphics. Please consider a simpler, more graphic design oriented request."),
                "progress_stage": "vector_suitability",
                "progress": 10
            }), 400
        
        # Stage 2: Planning Phase - Create structured design plan
        logger.info("\n[STAGE 2: Planning Phase]")
        logger.info("-"*50)
        logger.info("Creating structured design plan...")
        logger.info(f"Using model: {PLANNER_MODEL}")
        design_plan = plan_design(user_input)
        logger.info("\nDesign Plan Generated:")
        for line in design_plan.split('\n')[:10]:  # Log first 10 lines of plan
            logger.info(f"  {line}")
        logger.info("  ...")
        
        # Stage 3: Design Knowledge Generation - Gather design best practices
        logger.info("\n[STAGE 3: Design Knowledge Generation]")
        logger.info("-"*50)
        logger.info("Gathering design knowledge and best practices...")
        logger.info(f"Using model: {DESIGN_KNOWLEDGE_MODEL}")
        design_knowledge = generate_design_knowledge(design_plan, user_input)
        logger.info("\nDesign Knowledge Generated:")
        for line in design_knowledge.split('\n')[:10]:  # Log first 10 lines of knowledge
            logger.info(f"  {line}")
        logger.info("  ...")
        
        # Combine design plan and knowledge for enhanced prompts
        logger.info("\nCombining design plan and knowledge...")
        design_context = f"""Design Plan:
{design_plan}

Design Knowledge and Best Practices:
{design_knowledge}

Original Request:
{user_input}"""
        logger.info("Design context preparation complete")
        
        # Stages 4-5: Enhancement Phases Skipped
        logger.info("\n[STAGES 4-5: Enhancement Phases SKIPPED]")
        logger.info("-"*50)
        prompt_to_use = user_input
        pre_enhanced_prompt = user_input
        enhanced_prompt = user_input

        # Stage 6: Generate image using GPT Image-1
        logger.info("STAGE 6: Image Generation Phase")
        gpt_image_base64, gpt_image_filename = generate_image_with_gpt(prompt_to_use, design_context)
        logger.info("Image generated with GPT Image-1")

        # Stage 7: Generate SVG using vtracer
        logger.info("STAGE 7: SVG Generation Phase")
        svg_code = generate_svg_from_image(gpt_image_base64, prompt_to_use)
        logger.info("SVG code generated from image")
        
        # Save the SVG
        svg_filename = save_svg(svg_code, prefix="svg")

        return jsonify({
            "original_prompt": user_input,
            "pre_enhanced_prompt": pre_enhanced_prompt,
            "enhanced_prompt": enhanced_prompt,
            "gpt_image_base64": gpt_image_base64,
            "gpt_image_url": f"/static/images/{gpt_image_filename}",
            "svg_code": svg_code,
            "svg_path": svg_filename,
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
                    "svg_path": svg_filename
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
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

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
            max_tokens=1000
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
        "temperature": 0.3,
        "max_tokens": 2000
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
                # Stage 1: Planning Phase
                logger.info("\n[STAGE 1: Planning Phase]")
                logger.info("-"*50)
                logger.info("Creating structured design plan...")
                logger.info(f"Using model: {PLANNER_MODEL}")
                design_plan = plan_design(latest_message)
                logger.info("\nDesign Plan Generated:")
                for line in design_plan.split('\n')[:10]:
                    logger.info(f"  {line}")
                logger.info("  ...")

                # Stage 2: Design Knowledge Generation
                logger.info("\n[STAGE 2: Design Knowledge Generation]")
                logger.info("-"*50)
                logger.info("Gathering design knowledge and best practices...")
                logger.info(f"Using model: {DESIGN_KNOWLEDGE_MODEL}")
                design_knowledge = generate_design_knowledge(design_plan, latest_message)
                logger.info("\nDesign Knowledge Generated:")
                for line in design_knowledge.split('\n')[:10]:
                    logger.info(f"  {line}")
                logger.info("  ...")

                # Stage 3: Pre-enhancement
                logger.info("\n[STAGE 3: Pre-enhancement Phase]")
                logger.info("-"*50)
                logger.info("Pre-enhancing prompt with design context...")
                logger.info(f"Using model: {PRE_ENHANCER_MODEL}")
                design_context = f"""Design Plan:\n{design_plan}\n\nDesign Knowledge:\n{design_knowledge}\n\nOriginal Request:\n{latest_message}"""
                pre_enhanced = pre_enhance_prompt(design_context)
                logger.info("\nPre-enhanced Prompt:")
                for line in pre_enhanced.split('\n')[:10]:
                    logger.info(f"  {line}")
                logger.info("  ...")

                # Stage 4: Final Enhancement
                logger.info("\n[STAGE 4: Final Enhancement Phase]")
                logger.info("-"*50)
                logger.info("Enhancing prompt with technical specifications...")
                logger.info(f"Using model: {PROMPT_ENHANCER_MODEL}")
                enhanced_prompt = enhance_prompt_with_chat(pre_enhanced)
                logger.info("\nEnhanced Prompt Generated:")
                for line in enhanced_prompt.split('\n')[:10]:
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
                svg_filename = save_svg(svg_code, prefix="assistant_svg")
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
                    "svg_path": svg_filename,
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
                    # Save the modified SVG
                    svg_filename = save_svg(modified_svg, prefix="modified_svg")
                    
                    # Get AI explanation of the changes
                    change_explanation_prompt = f"I've modified the design based on the user's request: '{latest_message}'. Here's the updated SVG:\n\n```svg\n{modified_svg}\n```\n\nPlease explain what changes were made and how the design now better meets their needs."
                    
                    temp_messages = messages + [{"role": "user", "content": change_explanation_prompt}]
                    ai_explanation = chat_with_ai_about_design(temp_messages, modified_svg)
                    
                    full_response = f"{ai_explanation}\n\n```svg\n{modified_svg}\n```\n\nIs there anything else you'd like me to adjust?"
                    
                    messages.append({"role": "assistant", "content": full_response})
                    
                    response_data = {
                        "response": full_response,
                        "svg_code": modified_svg,
                        "svg_path": svg_filename,
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

# Parallel SVG Pipeline Functions
def process_image_parallel(image_base64, enhanced_prompt):
    """Process image in parallel using multiple techniques including text SVG"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit tasks
        vtracer_future = executor.submit(process_vtracer, image_base64)
        text_future = executor.submit(process_text_extraction, image_base64)
        simple_future = executor.submit(process_simple_conversion, image_base64)
        text_svg_future = executor.submit(process_text_svg_generation, image_base64, enhanced_prompt)
        
        # Gather results
        results = {
            'vtracer': vtracer_future.result(),
            'text': text_future.result(),
            'simple': simple_future.result(),
            'text_svg': text_svg_future.result()
        }
        
        return results

def process_ocr_svg(image_data):
    """Generate a text-only SVG using GPT-4o-mini by passing the image directly to the chat API."""
    # Base64-encode the PNG image
    img_b64 = base64.b64encode(image_data).decode('utf-8')
    # Build prompts matching parallel_svg_pipeline.py style
    system_prompt = """You are an expert SVG code generator. Your task is to create precise, clean, and optimized SVG code that exactly matches the provided image. Follow these guidelines:
1. Create SVG with dimensions 1080x1080 pixels
2. Ensure perfect positioning and alignment of all elements
3. Use appropriate viewBox and preserveAspectRatio attributes
4. Implement proper layering of elements
5. Optimize paths and shapes for better performance
6. Use semantic grouping (<g>) for related elements
7. Include necessary font definitions and styles
8. Ensure text elements are properly positioned and styled
9. Implement gradients, patterns, or filters if present in the image
10. Use precise color values matching the image exactly

Focus on producing production-ready, clean SVG code that renders identically to the input image.
Return ONLY the SVG code without any explanations or comments."""
    user_content = [
        {"type": "text", "text": "Generate an SVG that contains only text elements exactly as seen in the image."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
    ]
    # Call Chat Completions API directly to support image_url message
    payload = {
        "model": CHAT_ASSISTANT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    response = requests.post(OPENAI_CHAT_ENDPOINT, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }, json=payload)
    
    data = response.json()
    if response.status_code != 200:
        logger.error(f"Error generating text SVG: {data}")
        raise Exception("Text SVG generation failed")
    
    content = data["choices"][0]["message"]["content"]
    # Extract the SVG
    import re
    match = re.search(r'<svg.*?</svg>', content, re.DOTALL)
    svg_code = match.group(0) if match else content.strip()
    
    # Save and return
    svg_filename = save_svg(svg_code, prefix='text_svg')
    return svg_code, svg_filename

def process_clean_svg(image_data):
    """Process text removal and convert to clean SVG using vtracer"""
    try:
        # Save the original image bytes to a temporary PNG file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_input_path = os.path.join(PARALLEL_OUTPUTS_DIR, f"temp_input_{timestamp}_{uuid.uuid4().hex[:8]}.png")
        with open(temp_input_path, "wb") as f:
            f.write(image_data)

        # Remove text from the image using remove_text_simple
        edited_png_path = remove_text_simple.remove_text(temp_input_path)

        # Convert the edited PNG to SVG using vtracer with optimized settings
        output_svg_path = os.path.join(PARALLEL_OUTPUTS_DIR, f"clean_{timestamp}_{uuid.uuid4().hex[:8]}.svg")
        
        if vtracer_available:
            vtracer.convert_image_to_svg_py(
                edited_png_path,
                output_svg_path,
                colormode='color',
                hierarchical='stacked',
                mode='spline',
                filter_speckle=4,
                color_precision=6,
                layer_difference=16,
                corner_threshold=60,
                length_threshold=4.0,
                max_iterations=10,
                splice_threshold=45,
                path_precision=3
            )
        else:
            # Fallback to simple converter
            png_to_svg_converter.convert_png_to_svg(edited_png_path, output_svg_path)

        # Read the generated SVG content
        with open(output_svg_path, 'r') as f:
            clean_svg_code = f.read()

        # Cleanup the temporary input file
        os.remove(temp_input_path)

        # Return clean SVG code and paths
        return clean_svg_code, output_svg_path, edited_png_path
        
    except Exception as e:
        logger.error(f"Error in process_clean_svg: {str(e)}")
        raise

def validate_and_clean_svg(svg_code):
    """Validate and clean SVG code to ensure it's properly formatted."""
    import re

    # Ensure SVG starts and ends properly
    if not svg_code.strip().startswith('<svg'):
        # Try to find SVG content in the response
        svg_match = re.search(r'(<svg.*?</svg>)', svg_code, re.DOTALL)
        if svg_match:
            svg_code = svg_match.group(1)
        else:
            logger.warning("No valid SVG found in response")
            return svg_code

    # Ensure proper namespace
    if 'xmlns=' not in svg_code:
        svg_code = svg_code.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)

    # Ensure proper closing
    if not svg_code.strip().endswith('</svg>'):
        svg_code = svg_code.rstrip() + '</svg>'

    # Clean up whitespace
    svg_code = re.sub(r'\s+', ' ', svg_code)
    svg_code = re.sub(r'>\s+<', '><', svg_code)

    return svg_code.strip()

def reduce_svg_content(svg_code, max_chars=50000):
    """Reduce SVG content size by removing unnecessary elements and simplifying paths."""
    import re

    # Remove comments
    svg_code = re.sub(r'<!--.*?-->', '', svg_code, flags=re.DOTALL)

    # Remove unnecessary whitespace and newlines
    svg_code = re.sub(r'\s+', ' ', svg_code)
    svg_code = re.sub(r'>\s+<', '><', svg_code)

    # Simplify path data by reducing precision
    def simplify_path(match):
        path_data = match.group(1)
        # Reduce decimal precision to 2 places
        path_data = re.sub(r'(\d+\.\d{3,})', lambda m: f"{float(m.group(1)):.2f}", path_data)
        return f'd="{path_data}"'

    svg_code = re.sub(r'd="([^"]*)"', simplify_path, svg_code)

    # If still too large, truncate paths but keep structure
    if len(svg_code) > max_chars:
        logger.warning(f"SVG still too large ({len(svg_code)} chars), truncating paths")
        # Keep the SVG header and first few paths
        svg_parts = svg_code.split('<path')
        if len(svg_parts) > 10:  # Keep only first 10 paths
            truncated = svg_parts[0]
            for i in range(1, min(11, len(svg_parts))):
                truncated += '<path' + svg_parts[i]
            # Ensure proper closing
            if not truncated.endswith('</svg>'):
                truncated += '</svg>'
            svg_code = truncated

    return svg_code

def extract_svg_elements(svg_code):
    """Extract different elements from SVG code."""
    import re

    # Extract styles
    styles = re.findall(r'<style[^>]*>(.*?)</style>', svg_code, re.DOTALL)

    # Extract defs (gradients, patterns, etc.)
    defs = re.findall(r'<defs[^>]*>(.*?)</defs>', svg_code, re.DOTALL)

    # Extract paths
    paths = re.findall(r'<path[^>]*(?:/>|>.*?</path>)', svg_code, re.DOTALL)

    # Extract text elements
    texts = re.findall(r'<text[^>]*>.*?</text>', svg_code, re.DOTALL)

    # Extract other shapes (rect, circle, ellipse, etc.)
    shapes = re.findall(r'<(?:rect|circle|ellipse|polygon|polyline|line)[^>]*(?:/>|>.*?</(?:rect|circle|ellipse|polygon|polyline|line)>)', svg_code, re.DOTALL)

    # Extract groups
    groups = re.findall(r'<g[^>]*>.*?</g>', svg_code, re.DOTALL)

    return {
        'styles': styles,
        'defs': defs,
        'paths': paths,
        'texts': texts,
        'shapes': shapes,
        'groups': groups
    }

def simple_combine_svgs(text_svg_code, traced_svg_code):
    """Advanced SVG combination - properly merge all elements with correct layering."""
    import re

    logger.info('Using advanced SVG combination')

    # Extract viewBox and dimensions from traced SVG (usually more accurate)
    traced_viewbox = re.search(r'viewBox="([^"]*)"', traced_svg_code)
    traced_width = re.search(r'width="([^"]*)"', traced_svg_code)
    traced_height = re.search(r'height="([^"]*)"', traced_svg_code)

    # Set default dimensions
    viewbox = traced_viewbox.group(1) if traced_viewbox else "0 0 1080 1080"
    width = traced_width.group(1) if traced_width else "1080"
    height = traced_height.group(1) if traced_height else "1080"

    # Extract elements from both SVGs
    traced_elements = extract_svg_elements(traced_svg_code)
    text_elements = extract_svg_elements(text_svg_code)

    logger.info(f'Extracted elements - Traced: {len(traced_elements["paths"])} paths, Text: {len(text_elements["texts"])} texts')

    # Build comprehensive combined SVG
    combined_parts = []

    # Start SVG with proper namespace and attributes
    combined_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{viewbox}">')

    # Add styles from both SVGs
    if text_elements['styles'] or traced_elements['styles']:
        combined_parts.append('<style type="text/css"><![CDATA[')
        for style in text_elements['styles'] + traced_elements['styles']:
            combined_parts.append(style.strip())
        combined_parts.append(']]></style>')

    # Add defs from both SVGs
    all_defs = text_elements['defs'] + traced_elements['defs']
    if all_defs:
        combined_parts.append('<defs>')
        for def_content in all_defs:
            combined_parts.append(def_content.strip())
        combined_parts.append('</defs>')

    # Add background layer (traced paths and shapes)
    if traced_elements['paths'] or traced_elements['shapes'] or traced_elements['groups']:
        combined_parts.append('<g id="background-layer" opacity="0.9">')

        # Add traced paths (background graphics)
        for path in traced_elements['paths']:
            combined_parts.append(path)

        # Add traced shapes
        for shape in traced_elements['shapes']:
            combined_parts.append(shape)

        # Add traced groups
        for group in traced_elements['groups']:
            combined_parts.append(group)

        combined_parts.append('</g>')

    # Add foreground layer (text and text-related elements)
    if text_elements['texts'] or text_elements['paths'] or text_elements['shapes'] or text_elements['groups']:
        combined_parts.append('<g id="text-layer">')

        # Add text paths (if any)
        for path in text_elements['paths']:
            combined_parts.append(path)

        # Add text shapes
        for shape in text_elements['shapes']:
            combined_parts.append(shape)

        # Add text groups
        for group in text_elements['groups']:
            combined_parts.append(group)

        # Add text elements
        for text in text_elements['texts']:
            combined_parts.append(text)

        combined_parts.append('</g>')

    # Close SVG
    combined_parts.append('</svg>')

    # Join all parts
    combined_svg = '\n'.join(combined_parts)

    # Clean up the SVG
    combined_svg = re.sub(r'\n\s*\n', '\n', combined_svg)  # Remove empty lines
    combined_svg = re.sub(r'>\s+<', '><', combined_svg)     # Remove whitespace between tags

    logger.info(f'Advanced combination completed - Final size: {len(combined_svg)} characters')
    return combined_svg

def combine_svgs(text_svg_code, traced_svg_code):
    """Combine text and path SVGs using GPT-4o-mini to produce a unified SVG."""
    import time
    logger.info('Stage 8: Combining SVGs using HTTP API')

    # Reduce content size to fit within token limits
    original_text_size = len(text_svg_code.encode('utf-8')) if isinstance(text_svg_code, str) else len(text_svg_code)
    original_path_size = len(traced_svg_code.encode('utf-8')) if isinstance(traced_svg_code, str) else len(traced_svg_code)
    logger.info(f'Original sizes - Text SVG: {original_text_size} bytes, Traced SVG: {original_path_size} bytes')

    # Reduce traced SVG size significantly as it's usually the largest
    reduced_traced_svg = reduce_svg_content(traced_svg_code, max_chars=30000)
    reduced_text_svg = reduce_svg_content(text_svg_code, max_chars=5000)

    reduced_text_size = len(reduced_text_svg.encode('utf-8'))
    reduced_path_size = len(reduced_traced_svg.encode('utf-8'))
    logger.info(f'Reduced sizes - Text SVG: {reduced_text_size} bytes, Traced SVG: {reduced_path_size} bytes')

    # Use a simpler combination approach if content is still too large
    total_size = reduced_text_size + reduced_path_size
    if total_size > 40000:  # Still too large, use simple combination
        logger.info('Content still large, using simple combination approach')
        return simple_combine_svgs(reduced_text_svg, reduced_traced_svg)

    logger.info('Preparing HTTP API request for SVG combination')
    system_prompt = """You are an expert SVG combiner. Your task is to create a single, complete SVG that combines vector paths (background) with text elements (foreground).

REQUIREMENTS:
1. Use viewBox and dimensions from the path SVG
2. Include ALL styles, defs, and gradients from both SVGs
3. Layer paths BEHIND text elements
4. Preserve all text positioning and styling
5. Ensure proper SVG structure with xmlns namespace
6. Return ONLY the complete SVG code, no explanations

STRUCTURE:
- Start with <svg> tag with proper attributes
- Include <style> and <defs> sections if present
- Add background layer with paths/shapes
- Add foreground layer with text elements
- Close with </svg>"""

    user_msg = f"BACKGROUND SVG (paths/shapes):\n{reduced_traced_svg}\n\nFOREGROUND SVG (text/labels):\n{reduced_text_svg}\n\nCombine these into one complete SVG with proper layering."
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_msg}
    ]

    # Prepare HTTP request
    payload = {
        'model': CHAT_ASSISTANT_MODEL,
        'messages': messages,
        'temperature': 0,
        'max_tokens': 4000
    }

    start_time = time.time()
    logger.info('Sending HTTP request to OpenAI API for SVG combination')
    try:
        resp = requests.post(OPENAI_CHAT_ENDPOINT, headers={
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {OPENAI_API_KEY_SVG}"
        }, json=payload, timeout=60)
        
        api_response_time = time.time() - start_time
        logger.info(f'HTTP response received in {api_response_time:.2f} seconds')

        if resp.status_code != 200:
            logger.error(f'API Error: {resp.status_code} - {resp.text}')
            logger.info('Falling back to simple combination')
            return simple_combine_svgs(reduced_text_svg, reduced_traced_svg)

        data = resp.json()
        combined_svg = data['choices'][0]['message']['content'].strip()

        # Extract SVG from response if wrapped in markdown
        if '```' in combined_svg:
            import re
            svg_match = re.search(r'```(?:svg)?\s*(.*?)\s*```', combined_svg, re.DOTALL)
            if svg_match:
                combined_svg = svg_match.group(1).strip()

        # Validate and clean the SVG
        combined_svg = validate_and_clean_svg(combined_svg)

        combined_size = len(combined_svg.encode('utf-8'))
        logger.info(f'Combined SVG size: {combined_size} bytes')
        total_time = time.time() - start_time
        logger.info(f'SVG combination completed in {total_time:.2f} seconds total')
        return combined_svg

    except Exception as e:
        logger.error(f'API Error during SVG combination: {str(e)}')
        logger.info('Falling back to simple combination')
        return simple_combine_svgs(reduced_text_svg, reduced_traced_svg)

def process_vtracer(image_base64):
    """Process image using vtracer"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"vtracer_input_{timestamp}_{uuid.uuid4().hex[:8]}.png"
        input_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, input_filename)
        
        # Save input image
        with open(input_filepath, "wb") as f:
            f.write(image_data)
            
        # Configure vtracer
        config = vtracer.Configuration()
        config.corner_threshold = 60
        config.length_threshold = 4.0
        config.splice_threshold = 45
        config.filter_speckle = 4
        config.color_mode = "color"
        config.hierarchical = True
        config.mode = "polygon"
        config.path_precision = 8
        
        # Process with vtracer
        svg_output = vtracer.convert_image_to_svg_path(input_filepath, config)
        
        # Save SVG output
        output_filename = f"vtracer_output_{timestamp}.svg"
        output_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, output_filename)
        with open(output_filepath, "w") as f:
            f.write(svg_output)
            
        return {
            'success': True,
            'svg': svg_output,
            'path': output_filename
        }
        
    except Exception as e:
        logger.error(f"Vtracer processing error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def process_text_extraction(image_base64):
    """Extract text from image using OCR"""
    try:
        # Decode and open image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        # Get text boxes
        boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Create SVG with text elements
        width, height = image.size
        svg_elements = ['<?xml version="1.0" encoding="UTF-8"?>']
        svg_elements.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
        
        for i in range(len(boxes['text'])):
            if boxes['text'][i].strip():
                x = boxes['left'][i]
                y = boxes['top'][i] + boxes['height'][i]  # Adjust y for text baseline
                svg_elements.append(f'  <text x="{x}" y="{y}" font-family="Arial" font-size="{boxes["height"][i]}px">{boxes["text"][i]}</text>')
        
        svg_elements.append('</svg>')
        svg_output = '\n'.join(svg_elements)
        
        # Save output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"text_output_{timestamp}.svg"
        output_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, output_filename)
        with open(output_filepath, "w") as f:
            f.write(svg_output)
            
        return {
            'success': True,
            'text': text,
            'svg': svg_output,
            'path': output_filename
        }
        
    except Exception as e:
        logger.error(f"Text extraction error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def process_simple_conversion(image_base64):
    """Simple PNG to SVG conversion"""
    try:
        # Decode and save image
        image_data = base64.b64decode(image_base64)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"simple_input_{timestamp}_{uuid.uuid4().hex[:8]}.png"
        input_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, input_filename)
        
        with open(input_filepath, "wb") as f:
            f.write(image_data)
            
        # Convert using simple converter
        output_filename = f"simple_output_{timestamp}.svg"
        output_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, output_filename)
        
        png_to_svg_converter.convert_png_to_svg(input_filepath, output_filepath)
        
        with open(output_filepath, "r") as f:
            svg_output = f.read()
            
        return {
            'success': True,
            'svg': svg_output,
            'path': output_filename
        }
        
    except Exception as e:
        logger.error(f"Simple conversion error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def process_text_svg_generation(image_base64, enhanced_prompt):
    """Generate SVG from text elements using OCR and AI"""
    try:
        # Decode and save image
        image_data = base64.b64decode(image_base64)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"text_svg_input_{timestamp}_{uuid.uuid4().hex[:8]}.png"
        input_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, input_filename)
        
        with open(input_filepath, "wb") as f:
            f.write(image_data)
            
        # Determine dimensions
        img = Image.open(input_filepath)
        width, height = img.size
        view_box = f"0 0 {width} {height}"
        
        # Use OCR to extract text positions (if pytesseract is available)
        try:
            import pytesseract
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Filter out empty text and build text elements list
            text_elements = []
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if text and int(ocr_data['conf'][i]) > 30:  # Confidence threshold
                    text_elements.append({
                        'text': text,
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i] + ocr_data['height'][i],  # Baseline adjustment
                        'font_size': ocr_data['height'][i]
                    })
            
            # Generate SVG using AI with OCR context
            messages = [
                {"role": "system", "content": (
                    "You are a precise SVG code generator. "
                    "Given image dimensions and extracted text elements with positions, "
                    "create a clean SVG that reproduces the text layout accurately. "
                    "Output only valid SVG code without explanations."
                )},
                {"role": "user", "content": (
                    f"Create an SVG for an image {width}x{height}px with these text elements: {text_elements}. "
                    f"Start with: <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"{view_box}\"> "
                    "Then add <text> elements with appropriate positioning, font-family=\"sans-serif\", and colors. "
                    "Close with </svg>. Return only the SVG code."
                )}
            ]
            
        except ImportError:
            # Fallback without OCR
            messages = [
                {"role": "system", "content": (
                    "You are a precise SVG code generator. "
                    "Create a clean text-based SVG design based on the prompt. "
                    "Output only valid SVG code without explanations."
                )},
                {"role": "user", "content": (
                    f"Create a text-based SVG {width}x{height}px for: {enhanced_prompt}. "
                    f"Start with: <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"{view_box}\"> "
                    "Add appropriate text elements with good typography. Close with </svg>."
                )}
            ]
        
        # Generate SVG using OpenAI
        response = chat_client.chat.completions.create(
            model=CHAT_ASSISTANT_MODEL,
            messages=messages,
            temperature=0.7
        )
        
        svg_output = response.choices[0].message.content.strip()
        
        # Save SVG output
        output_filename = f"text_svg_output_{timestamp}.svg"
        output_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, output_filename)
        with open(output_filepath, "w") as f:
            f.write(svg_output)
            
        return {
            'success': True,
            'svg': svg_output,
            'path': output_filename,
            'method': 'text_svg_ai'
        }
        
    except Exception as e:
        logger.error(f"Text SVG generation error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/api/generate-parallel-svg', methods=['POST'])
def handle_parallel_svg():
    """Handle parallel SVG generation request"""
    try:
        result = generate_parallel_svg_pipeline(request.json)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in parallel SVG handler: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-text-svg', methods=['POST'])
def generate_image_text_svg():
    """Pipeline: Stages 1–6 image gen, then Stage 7: text OCR & SVG generation"""
    try:
        data = request.json or {}
        user_input = data.get('prompt', '')
        skip_enhancement = data.get('skip_enhancement', False)

        if not user_input:
            return jsonify({'error': 'No prompt provided'}), 400

        logger.info('=== IMAGE→TEXT→SVG PIPELINE START ===')

        # Stage 1: Vector Suitability Check
        logger.info('Stage 1: Vector Suitability Check')
        vector_suitability = check_vector_suitability(user_input)
        if vector_suitability.get('not_suitable', False):
            return jsonify({
                'error': 'Not suitable for SVG',
                'guidance': vector_suitability.get('guidance'),
                'stage': 1
            }), 400

        # Stage 2: Design Planning
        logger.info('Stage 2: Design Planning')
        design_plan = plan_design(user_input)

        # Stage 3: Design Knowledge Generation
        logger.info('Stage 3: Design Knowledge Generation')
        design_knowledge = generate_design_knowledge(design_plan, user_input)

        # Prepare context for enhancements
        design_context = f"""Design Plan:\n{design_plan}\n\nDesign Knowledge and Best Practices:\n{design_knowledge}\n\nOriginal Request:\n{user_input}"""

        # Stage 4 & 5: Prompt Enhancements
        if skip_enhancement:
            enhanced_prompt = user_input
        else:
            logger.info('Stage 4: Pre-Enhancement')
            pre = pre_enhance_prompt(design_context)
            logger.info('Stage 5: Technical Enhancement')
            enhanced_prompt = enhance_prompt_with_chat(pre)

        # Stage 6: Image Generation via GPT-Image
        logger.info('Stage 6: Image Generation via GPT-Image')
        image_base64, image_filename = generate_image_with_gpt(enhanced_prompt, design_context)

        # Stage 7: OCR and SVG generation via GPT-4o-mini
        logger.info('Stage 7: OCR and SVG Generation via GPT-4o-mini')
        
        # Decode and save the image to disk
        image_data = base64.b64decode(image_base64)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"text_input_{timestamp}_{uuid.uuid4().hex[:8]}.png"
        input_filepath = os.path.join(IMAGES_DIR, input_filename)
        with open(input_filepath, "wb") as f:
            f.write(image_data)
        
        # Determine dimensions
        img = Image.open(input_filepath)
        width, height = img.size
        view_box = f"0 0 {width} {height}"
        image_url = f"/static/images/{input_filename}"
        
        # Prepare prompt for SVG generation
        messages = [
            {"role": "system", "content": (
                "You are a precise SVG code generator. "
                "Given image dimensions and a list of text elements with their positions (x, y), font sizes, and fill colors, "
                "output only a valid SVG string with the correct <svg> wrapper and one <text> tag per element, "
                "using each 'text' value exactly as provided (no changes). Do not include explanations or markup extras."
            )},
            {"role": "user", "content": (
                f"Here is an image at URL: {image_url}. The image width is {width}px and height is {height}px. "
                "Generate a valid SVG that begins with:\n"
                f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"{view_box}\">\n"
                "Then include one <text> element per visible text string, each with appropriate x, y coordinates, "
                "font-family=\"sans-serif\", font-size matching the original, and fill color in hex. Close the SVG at the end. "
                "Return only the complete SVG code."
            )}
        ]
        
        response = chat_client.chat.completions.create(
            model=CHAT_ASSISTANT_MODEL,
            messages=messages,
            temperature=0.7
        )
        
        svg_code = response.choices[0].message.content.strip()
        
        # Save and return
        svg_filename = save_svg(svg_code, prefix='text_svg')
        
        return jsonify({
            'original_prompt': user_input,
            'enhanced_prompt': enhanced_prompt,
            'image_base64': image_base64,
            'image_url': f'/static/images/{image_filename}',
            'svg_code': svg_code,
            'svg_path': svg_filename,
            'stage': 7,
            'pipeline_type': 'text_svg',
            'stages': {
                'vector_suitability': {'completed': True, 'suitable': True},
                'design_plan': {'completed': True, 'content': design_plan},
                'design_knowledge': {'completed': True, 'content': design_knowledge},
                'pre_enhancement': {'completed': not skip_enhancement, 'skipped': skip_enhancement},
                'prompt_enhancement': {'completed': not skip_enhancement, 'skipped': skip_enhancement},
                'image_generation': {'completed': True, 'image_url': f'/static/images/{image_filename}'},
                'text_svg_generation': {'completed': True, 'svg_path': svg_filename}
            }
        })
        
    except Exception as e:
        logger.error(f"Error in text SVG pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'stage': 'error'}), 500

@app.route('/api/generate-combined-svg', methods=['POST'])
def generate_combined_svg():
    """Full parallel pipeline: Stages 1-6 + Stage 7: Text SVG + Stage 8: Traced SVG + Stage 9: Combine SVGs"""
    try:
        data = request.json or {}
        user_input = data.get('prompt', '')
        skip_enhancement = data.get('skip_enhancement', False)

        if not user_input:
            return jsonify({'error': 'No prompt provided'}), 400

        logger.info('=== COMBINED SVG PIPELINE START ===')
        logger.info(f'Processing prompt: {user_input[:100]}...')

        # Stages 1-5: Same as other pipelines
        logger.info('Stage 1: Vector Suitability Check')
        vector_suitability = check_vector_suitability(user_input)
        if vector_suitability.get('not_suitable', False):
            return jsonify({
                'error': 'Not suitable for SVG',
                'guidance': vector_suitability.get('guidance'),
                'stage': 1
            }), 400

        # Stage 2: Design Planning
        logger.info('Stage 2: Design Planning')
        design_plan = plan_design(user_input)

        # Stage 3: Design Knowledge Generation
        logger.info('Stage 3: Design Knowledge Generation')
        design_knowledge = generate_design_knowledge(design_plan, user_input)

        # Prepare context for enhancements
        design_context = f"""Design Plan:\n{design_plan}\n\nDesign Knowledge and Best Practices:\n{design_knowledge}\n\nOriginal Request:\n{user_input}"""

        # Stage 4 & 5: Prompt Enhancements
        if skip_enhancement:
            enhanced_prompt = user_input
            pre_enhanced_prompt = user_input
        else:
            logger.info('Stage 4: Pre-Enhancement')
            pre_enhanced_prompt = pre_enhance_prompt(design_context)
            logger.info('Stage 5: Technical Enhancement')
            enhanced_prompt = enhance_prompt_with_chat(pre_enhanced_prompt)

        # Stage 6: Image Generation via GPT-Image
        logger.info('Stage 6: Image Generation')
        image_base64, image_filename = generate_image_with_gpt(enhanced_prompt, design_context)
        
        # Decode image for processing
        image_data = base64.b64decode(image_base64)

        # Stage 7: Text SVG Generation using OCR + AI
        logger.info('Stage 7: Text SVG Generation (OCR + AI)')
        try:
            text_svg_code, text_svg_filename = process_ocr_svg(image_data)
            logger.info(f'Text SVG generated: {text_svg_filename}')
        except Exception as e:
            logger.error(f'Text SVG generation failed: {str(e)}')
            text_svg_code = '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1080" viewBox="0 0 1080 1080"></svg>'
            text_svg_filename = 'fallback_text.svg'

        # Stage 8: Traced SVG Generation (Clean background)
        logger.info('Stage 8: Traced SVG Generation (Clean Background)')
        try:
            if vtracer_available:
                traced_svg_code, traced_svg_path, edited_png_path = process_clean_svg(image_data)
                logger.info(f'Traced SVG generated: {traced_svg_path}')
            else:
                # Fallback: Use simple converter on original image
                logger.info('vtracer not available, using simple conversion')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_input_path = os.path.join(PARALLEL_OUTPUTS_DIR, f"simple_input_{timestamp}_{uuid.uuid4().hex[:8]}.png")
                with open(temp_input_path, "wb") as f:
                    f.write(image_data)
                
                traced_svg_path = os.path.join(PARALLEL_OUTPUTS_DIR, f"simple_traced_{timestamp}.svg")
                png_to_svg_converter.convert_png_to_svg(temp_input_path, traced_svg_path)
                
                with open(traced_svg_path, 'r') as f:
                    traced_svg_code = f.read()
                
                os.remove(temp_input_path)
                edited_png_path = None
                
        except Exception as e:
            logger.error(f'Traced SVG generation failed: {str(e)}')
            traced_svg_code = '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1080" viewBox="0 0 1080 1080"></svg>'
            traced_svg_path = 'fallback_traced.svg'
            edited_png_path = None

        # Stage 9: Combine Text SVG + Traced SVG
        logger.info('Stage 9: Combining Text SVG + Traced SVG')
        try:
            combined_svg_code = combine_svgs(text_svg_code, traced_svg_code)
            logger.info('SVG combination completed successfully')
        except Exception as e:
            logger.error(f'SVG combination failed: {str(e)}')
            logger.info('Using simple combination fallback')
            combined_svg_code = simple_combine_svgs(text_svg_code, traced_svg_code)

        # Save the final combined SVG
        combined_svg_filename = save_svg(combined_svg_code, prefix='combined_svg')
        logger.info(f'Final combined SVG saved: {combined_svg_filename}')

        # Prepare response with all outputs
        response_data = {
            'original_prompt': user_input,
            'pre_enhanced_prompt': pre_enhanced_prompt,
            'enhanced_prompt': enhanced_prompt,
            'image_base64': image_base64,
            'image_url': f'/static/images/{image_filename}',
            'text_svg_code': text_svg_code,
            'text_svg_path': text_svg_filename,
            'traced_svg_code': traced_svg_code,
            'traced_svg_path': traced_svg_path,
            'combined_svg_code': combined_svg_code,
            'combined_svg_path': combined_svg_filename,
            'edited_png_path': edited_png_path,
            'pipeline_type': 'combined_svg',
            'stage': 9,
            'stages': {
                'vector_suitability': {'completed': True, 'suitable': True},
                'design_plan': {'completed': True, 'content': design_plan},
                'design_knowledge': {'completed': True, 'content': design_knowledge},
                'pre_enhancement': {'completed': not skip_enhancement, 'skipped': skip_enhancement, 'content': pre_enhanced_prompt},
                'prompt_enhancement': {'completed': not skip_enhancement, 'skipped': skip_enhancement, 'content': enhanced_prompt},
                'image_generation': {'completed': True, 'image_url': f'/static/images/{image_filename}'},
                'text_svg_generation': {'completed': True, 'svg_path': text_svg_filename},
                'traced_svg_generation': {'completed': True, 'svg_path': traced_svg_path},
                'svg_combination': {'completed': True, 'svg_path': combined_svg_filename}
            }
        }

        logger.info('=== COMBINED SVG PIPELINE COMPLETE ===')
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in combined SVG pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'stage': 'failed'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

# Import route modules to register them (do this after app is created)
try:
    from image_to_text_svg_pipeline import create_text_svg_route
    create_text_svg_route(app)
    logger.info("Successfully registered image_to_text_svg_pipeline routes")
except Exception as e:
    logger.error(f"Failed to register image_to_text_svg_pipeline routes: {str(e)}")

if __name__ == '__main__':
    try:
        # Get port from environment variable for deployment
        port = int(os.environ.get('PORT', 5000))
        # Use 0.0.0.0 when PORT env var is set (deployment)
        host = '0.0.0.0' if 'PORT' in os.environ else '127.0.0.1'
        
        logger.info(f"Starting Flask app on {host}:{port}")
        logger.info(f"Static directory: {STATIC_DIR}")
        logger.info(f"Images directory: {IMAGES_DIR}")
        
        app.run(host=host, port=port)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)