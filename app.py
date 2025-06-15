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
    logger,
    STATIC_DIR,
    IMAGES_DIR,
    save_image,
    save_svg,
    generate_image_with_gpt,
    OPENAI_CHAT_ENDPOINT,
    OPENAI_API_KEY_ENHANCER,
    OPENAI_API_KEY_SVG,
    GPT_IMAGE_MODEL,
    SVG_GENERATOR_MODEL,
    CHAT_ASSISTANT_MODEL
)
from shared_functions import (
    check_vector_suitability,
    plan_design,
    generate_design_knowledge,
    pre_enhance_prompt,
    enhance_prompt_with_chat
)
from parallel_svg_pipeline import generate_parallel_svg_pipeline, init_parallel_pipeline
from image_to_text_svg_pipeline import generate_image_text_svg

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

try:
    # Initialize parallel pipeline
    init_parallel_pipeline()
except Exception as e:
    logger.error(f"Failed to initialize parallel pipeline: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

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

# API keys
OPENAI_API_KEY_ENHANCER = os.getenv('OPENAI_API_KEY_ENHANCER')
OPENAI_API_KEY_SVG = os.getenv('OPENAI_API_KEY_SVG')

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
            quality="low"   # Changed from "low" to "standard" for better quality
        )

        # Get base64 image data from the response
        # The response structure changed in newer versions of the API
        image_base64 = response.data[0].b64_json if hasattr(response.data[0], 'b64_json') else response.data[0].url

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
    """Process image in parallel using multiple techniques"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks
        vtracer_future = executor.submit(process_vtracer, image_base64)
        text_future = executor.submit(process_text_extraction, image_base64)
        simple_future = executor.submit(process_simple_conversion, image_base64)
        
        # Gather results
        results = {
            'vtracer': vtracer_future.result(),
            'text': text_future.result(),
            'simple': simple_future.result()
        }
        
        return results

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

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

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