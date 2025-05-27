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
import numpy as np
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS with specific settings
CORS(app, 
     origins=[
          'https://infoui-6fk6va0qk-varuns-projects-859429fc.vercel.app',
         'infoui-f2zp4fwwo-varuns-projects-859429fc.vercel.app',
         'https://infoui-git-copilot-fix-03135c45-0967ec-varuns-projects-859429fc.vercel.app',
         'http://localhost:3000', 
         'http://localhost:3001',
         'http://127.0.0.1:3000', 
         'http://127.0.0.1:3001',
         'https://pppp-351z.onrender.com',
         'https://infoui.vercel.app',
     'https://infoui-f2zp4fwwo-varuns-projects-859429fc.vercel.app',
          'https://infoui-6fk6va0qk-varuns-projects-859429fc.vercel.app',
          'infoui-lt76sqq19-varuns-projects-859429fc.vercel.app',
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
PRE_ENHANCER_MODEL = "gpt-4o-mini"
PROMPT_ENHANCER_MODEL = "gpt-4o-mini"
GPT_IMAGE_MODEL = "gpt-image-1"
SVG_GENERATOR_MODEL = "gpt-4.1-mini"
CHAT_ASSISTANT_MODEL = "gpt-4o-mini"
PLANNING_MODEL = "gpt-4o-mini"
DESIGN_KNOWLEDGE_MODEL = "gpt-4o-mini"

def create_design_plan(user_input):
    """Create a detailed plan for the design based on user input"""
    logger.info(f"Creating design plan for input: {user_input[:100]}...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }

    system_prompt = """You are a modern design planning expert. Create a minimalist, professional design plan focusing on:

1. Modern Design Approach
   - Clean, professional aesthetics
   - Minimalist composition
   - Contemporary design trends
   - Brand-focused elements only

2. Essential Components
   - Primary message/headline
   - Brand elements
   - Critical information only
   - Whitespace utilization

3. Professional Style
   - Modern typography (max 2 fonts)
   - Limited, professional color palette
   - Balanced composition
   - Clear visual hierarchy

4. Technical Considerations
   - High-quality SVG output
   - Responsive design principles
   - Professional animations (if needed)
   - Performance optimization

Keep the response in natural language, focusing on professional and modern design principles."""

    payload = {
        "model": PLANNING_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    logger.info("Calling OpenAI API for design planning")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"OpenAI API error in planning: {response_data}")
        raise Exception(f"OpenAI API error: {response_data.get('error', {}).get('message', 'Unknown error')}")

    plan = response_data["choices"][0]["message"]["content"]
    logger.info(f"Successfully created design plan: {plan[:200]}...")
    return plan

def generate_design_knowledge(design_plan):
    """Generate specific design knowledge based on the plan"""
    logger.info("Generating design knowledge based on plan")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }

    system_prompt = """You are a design knowledge expert. Based on the provided design plan, generate specific design knowledge in plain text format. Include:

1. Design principles and guidelines
2. Color theory recommendations
3. Typography suggestions
4. Layout best practices
5. Visual hierarchy tips
6. Technical requirements
7. Industry standards and trends

Keep the response in natural language, avoiding technical formats."""

    payload = {
        "model": DESIGN_KNOWLEDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": design_plan
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    logger.info("Calling OpenAI API for design knowledge generation")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"OpenAI API error in knowledge generation: {response_data}")
        raise Exception(f"OpenAI API error: {response_data.get('error', {}).get('message', 'Unknown error')}")

    knowledge = response_data["choices"][0]["message"]["content"]
    logger.info(f"Successfully generated design knowledge: {knowledge[:200]}...")
    return knowledge

def pre_enhance_prompt(user_input, design_plan, design_knowledge):
    """Initial enhancement of user query using design plan and knowledge"""
    logger.info(f"Pre-enhancing prompt with design context")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }

    system_prompt = """You are a prompt pre-enhancement expert. Using the provided design plan and knowledge:

1. Analyze the user's request in context of the design plan
2. Incorporate relevant design knowledge
3. Structure the prompt to include:
   - Design type and purpose
   - Key visual elements
   - Style and mood
   - Technical requirements
   - Specific design elements

Focus on creating a clear, detailed foundation for further enhancement."""

    payload = {
        "model": PRE_ENHANCER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": json.dumps({
                    "user_input": user_input,
                    "design_plan": design_plan,
                    "design_knowledge": design_knowledge
                })
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    logger.info("Calling OpenAI API for prompt pre-enhancement")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"OpenAI API error in pre-enhancement: {response_data}")
        raise Exception(f"OpenAI API error: {response_data.get('error', {}).get('message', 'Unknown error')}")

    pre_enhanced = response_data["choices"][0]["message"]["content"]
    logger.info(f"Successfully pre-enhanced prompt: {pre_enhanced[:200]}...")
    return pre_enhanced

def enhance_prompt_with_chat(pre_enhanced_prompt, design_plan, design_knowledge):
    """Final prompt enhancement incorporating all previous stages"""
    logger.info("Performing final prompt enhancement")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }

    system_prompt = """You are a final prompt enhancement expert. Your task is to:

1. Refine and optimize the pre-enhanced prompt
2. Ensure all design requirements are clearly specified
3. Add specific details for:
   - Layout and composition
   - Color schemes and relationships
   - Typography and text treatment
   - Visual elements and their placement
   - SVG-specific requirements

Create a comprehensive prompt that will guide the image and SVG generation."""

    payload = {
        "model": PROMPT_ENHANCER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": json.dumps({
                    "pre_enhanced_prompt": pre_enhanced_prompt,
                    "design_plan": design_plan,
                    "design_knowledge": design_knowledge
                })
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    logger.info("Calling OpenAI API for final prompt enhancement")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"OpenAI API error in final enhancement: {response_data}")
        raise Exception(f"OpenAI API error: {response_data.get('error', {}).get('message', 'Unknown error')}")

    enhanced = response_data["choices"][0]["message"]["content"]
    logger.info(f"Successfully enhanced prompt: {enhanced[:200]}...")
    return enhanced

def generate_image_with_gpt(enhanced_prompt):
    """Generate image using GPT Image-1 model"""
    try:
        logger.info("Generating image with GPT Image-1")
        response = openai.images.generate(
            model=GPT_IMAGE_MODEL,
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="low"
        )
        
        # Get base64 image data from the response
        # The response structure changed in newer versions of the API
        image_base64 = response.data[0].b64_json if hasattr(response.data[0], 'b64_json') else response.data[0].url
        
        # Save the generated image
        filename = save_image(image_base64, prefix="gpt_image")
        
        logger.info("Image generated and saved successfully with GPT Image-1")
        return filename
    except Exception as e:
        logger.error(f"Error generating image with GPT Image-1: {str(e)}")
        raise

def generate_svg_from_image(image_path, enhanced_prompt):
    """Generate SVG using Potrace based on image"""
    logger.info("Starting SVG generation from image using Potrace")
    
    try:
        # Get full path to the image
        input_image_path = os.path.join(IMAGES_DIR, image_path)
        
        # Load and preprocess the image
        image = Image.open(input_image_path)
        
        # Convert to bitmap for potrace
        bm = Bitmap(image, blacklevel=0.5)
        
        # Trace the bitmap to get curves
        plist = bm.trace(
            turdsize=2,
            turnpolicy=POTRACE_TURNPOLICY_MINORITY,
            alphamax=1,
            opticurve=True,
            opttolerance=0.2,
        )
        
        # Generate SVG content
        width, height = image.size
        svg_content = f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'''
        
        parts = []
        for curve in plist:
            fs = curve.start_point
            parts.append(f"M{fs.x},{fs.y}")
            for segment in curve.segments:
                if segment.is_corner:
                    a = segment.c
                    b = segment.end_point
                    parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                else:
                    a = segment.c1
                    b = segment.c2
                    c = segment.end_point
                    parts.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")
            parts.append("z")
        
        svg_content += f'<path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(parts)}"/>'
        svg_content += "</svg>"
        
        # Save SVG content to a temporary file
        svg_filename = save_svg(svg_content)
        
        logger.info(f"SVG generated successfully: {svg_filename}")
        return svg_filename
        
    except Exception as e:
        logger.error(f"Error in SVG generation with Potrace: {str(e)}")
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
        raise

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
    """Enhanced SVG generator endpoint with detailed workflow stages"""
    try:
        data = request.json
        user_input = data.get('prompt', '')

        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        logger.info(f"Starting design generation process for: {user_input[:50]}...")
        
        # Track progress for UI feedback
        progress = {
            "stage": "planning",
            "progress": 0,
            "message": "Creating design plan..."
        }

        # Step 1: Create Design Plan
        design_plan = create_design_plan(user_input)
        progress.update({"stage": "knowledge", "progress": 20, "message": "Generating design knowledge..."})

        # Step 2: Generate Design Knowledge
        design_knowledge = generate_design_knowledge(design_plan)
        progress.update({"stage": "pre_enhancement", "progress": 40, "message": "Pre-enhancing prompt..."})

        # Step 3: Pre-enhance Prompt
        pre_enhanced_prompt = pre_enhance_prompt(user_input, design_plan, design_knowledge)
        progress.update({"stage": "enhancement", "progress": 60, "message": "Enhancing prompt..."})

        # Step 4: Final Prompt Enhancement
        enhanced_prompt = enhance_prompt_with_chat(pre_enhanced_prompt, design_plan, design_knowledge)
        progress.update({"stage": "image_generation", "progress": 80, "message": "Generating image..."})

        # Step 5: Generate Image
        image_path = generate_image_with_gpt(enhanced_prompt)
        progress.update({"stage": "svg_generation", "progress": 90, "message": "Generating SVG..."})

        # Step 6: Generate SVG
        svg_path = generate_svg_from_image(image_path, enhanced_prompt)
        progress.update({"stage": "complete", "progress": 100, "message": "Design complete!"})

        # Save the SVG
        svg_filename = save_svg(svg_path)

        return jsonify({
            "design_plan": design_plan,
            "design_knowledge": design_knowledge,
            "original_prompt": user_input,
            "pre_enhanced_prompt": pre_enhanced_prompt,
            "enhanced_prompt": enhanced_prompt,
            "image_path": image_path,
            "svg_path": svg_path,
            "svg_url": f"/static/images/{svg_filename}",
            "progress": progress
        })

    except Exception as e:
        logger.error(f"Error in generate_svg: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({
            "error": str(e),
            "progress": {
                "stage": "error",
                "progress": 0,
                "message": f"Error: {str(e)}"
            }
        }), 500

def chat_with_ai_about_design(messages, current_svg=None):
    """Enhanced conversational AI that can discuss and modify designs"""
    logger.info("Starting conversational AI interaction")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
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

    payload = {
        "model": CHAT_ASSISTANT_MODEL,
        "messages": ai_messages,
        "temperature": 0.7,
        "max_tokens": 1000
    }

    logger.info(f"Calling conversational AI with {len(ai_messages)} messages")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"Conversational AI error: {response_data}")
        return "I'm sorry, I'm having trouble processing your request right now. Please try again."

    ai_response = response_data["choices"][0]["message"]["content"]
    logger.info(f"AI response generated: {ai_response[:100]}...")
    return ai_response

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
        
        logger.info("Received chat request")
        logger.info(f"Chat history length: {len(messages)}")
        logger.info(f"Last message: {messages[-1] if messages else 'No messages'}")
        
        if not messages:
            logger.warning("No messages provided in request")
            return jsonify({"error": "No messages provided"}), 400

        # Get the latest user message
        latest_message = messages[-1]["content"].lower() if messages else ""
        
        logger.info("Processing new design creation request")
            
        # Step 1: Create design plan
        design_plan = create_design_plan(latest_message)
        logger.info("Design Plan Generated")

        # Step 2: Generate design knowledge
        design_knowledge = generate_design_knowledge(design_plan)
        logger.info("Design Knowledge Generated")

        # Step 3: Pre-enhance prompt
        pre_enhanced_prompt = pre_enhance_prompt(latest_message, design_plan, design_knowledge)
        logger.info("Pre-enhanced Prompt Created")

        # Step 4: Final prompt enhancement
        enhanced_prompt = enhance_prompt_with_chat(pre_enhanced_prompt, design_plan, design_knowledge)
        logger.info("Enhanced Prompt Generated")

        # Step 5: Generate initial image
        image_filename = generate_image_with_gpt(enhanced_prompt)
        logger.info(f"Image generated: {image_filename}")

        # Step 6: Generate SVG from image
        svg_path = generate_svg_from_image(image_filename, enhanced_prompt)
        svg_filename = save_svg(svg_path)
        logger.info(f"SVG generated: {svg_filename}")

        # Step 7: Generate explanation
        explanation = chat_with_ai_about_design(messages + [{"role": "assistant", "content": svg_path}])
        logger.info("Explanation Generated")

        return jsonify({
            "response": explanation,
            "svg_path": svg_path,
            "image": image_filename,
            "progress": {
                "stage": "complete",
                "progress": 100,
                "message": "Design complete!"
            }
        })
            
    except Exception as e:
        logger.error(f"Error in design creation: {str(e)}")
        return jsonify({
            "error": str(e),
            "progress": {
                "stage": "error",
                "progress": 0,
                "message": f"Error: {str(e)}"
            }
        }), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets PORT=8000)
    port = int(os.getenv('PORT', 5001))
    
    # Use 0.0.0.0 for production (Render) and 127.0.0.1 for local development
    host = '0.0.0.0' if os.getenv('PORT') else '127.0.0.1'
    
    # Disable debug mode in production
    debug = not bool(os.getenv('PORT'))
    
    logger.info(f"Starting Flask application on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
