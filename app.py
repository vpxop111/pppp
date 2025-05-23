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

# Load environment variables
load_dotenv()

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

def pre_enhance_prompt(user_input):
    """Initial enhancement of user query using standard GPT-4o mini"""
    logger.info(f"Pre-enhancing prompt: {user_input[:100]}...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

    payload = {
        "model": PRE_ENHANCER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a assistant you are a prompt enhancer your task is to take input from user like "create coming soon poster for clothing company" or "create testimonial for a restaurant" you need to convert this prompt into detailed examples given below. You must modify prompt according to prompt given by user. you must make sure color and font should same as given by user if not given kindly use it on your own while keeping design principles and fundamentals in your mind.

Don't add custom elements, shapes, and random figures in prompts.
You must generate a prompt same as given below examples


Examples for Coming Soon Pages: -

- Design a clean and elegant coming soon page with a black rectangular border, centered "Coming Soon" text in a cursive font, and a white background using Water Brush font at 60px size.
- Design a modern coming soon page with a sleek black background, a prominent complex SVG graphic, centered layout, minimalist text, and seamless integration of decorative SVG elements.
- Design a stylish coming soon page with a soft pink background, golden brown border, cursive "something" in Allura font, and main text in Times New Roman font with heart symbols and website link.
- Create a coming soon page with a light beige background, dark gray content area featuring large white text for "COMING SOON," a website URL, and a - "GRAND OPENING" button styled in green with custom fonts.
- Design a natural-themed coming soon page with a dark green background, featuring Bebas Neue font in large size for "Coming" and "soon," a countdown section, and an angled exclamation mark graphic in the bottom right corner.
- Design a coming soon page with a deep blue background, featuring centrally positioned text in Tektur font (white for 'COMING' with shadow, orange for 'SOON' with shadow), and white lines for definition, all slightly rotated.
- Design a warm and welcoming coming soon page with a beige background, centered layout, modern font for the title, cursive and bold fonts for 'Coming Soon,' and simple font for a website link, including decorative SVG elements.
- Design a warm and inviting coming soon page with a light beige background, bold 'COMING SOON' text in a darker brown color, and a date below in 'Open Sans' font.
- Design a playful coming soon page with a cream background, grid pattern, bold text in warm taupe, cursive text in dark green, and sans-serif font for additional information, ensuring a centered layout for clarity and appeal.
- Design an elegant coming soon page with beige background, featuring 'LARANA STORE' in bold serif font, 'BEAUTY PRODUCT' in a decorative rectangle, and 'COMING SOON' in red and cursive font, with 'STAY TUNED' and '@REALLYGREATSITE' included, along with whimsical star shapes for a playful touch.
- Design a bold and modern coming soon page with black background, gray border, large Bebas Neue text for "COMING" and "SOON," decorative Allura font for "-Best Store-" and "Stay Tuned," and a date of 12.12.2025, with clear website link at the bottom.
- Design a clean coming soon page with a white background, a beige box with a grey border, 'LICERIA & CO.' at the top in large dark blue-grey text, 'WE ARE' below in smaller text, 'OPEN' centered, and 'OPENING HOURS' with hours displayed below.
- Design a modern and minimalistic coming soon page with a clean white background, featuring centered text in Open Sans font and time indicators styled in dark gray.
- Design a coming soon page with centered text elements in black color on a soft pink background, featuring the phrases 'NOW WE ARE', 'OPEN', and 'VISIT OUR SITE' with specific font styles and sizes. Include a website link at the bottom in a regular font.

Examples for Testimonial Designs: -


- Design a testimonial graphic with a teal background, featuring a beige square container with "TESTIMONIAL" in bold Alfarn font, three orange circles below, PT Serif font for customer experience lines, and square quotation marks for visual appeal.,
- Design a testimonial with a white background, a large pink circle in the center, testimonial text in Raleway font, customer's name at the bottom, and decorative elements like 4-spoke stars and a dotted circle.,
- Create a testimonial with a cream background, black quotation marks, centered title "Testimonial," testimonial text "We couldn't be happier with the outstanding service provided by Weblake Company...," author name "- Linda Brown -" centered below, and website URL "a.barnescopy.site.com" at the bottom in Instrument Serif font.,
- Design a testimonial with a neon green header, black background, and round corner speech box, featuring the title "CLIENT TESTIMONIAL" in Bebas Neue font at 80px, testimonial text in Neue font at 42px, and name "MIHIR GEHLOT" in Raleway font at 36px, all centered and styled accordingly.,
- Design a testimonial with a blue background and a light blue header, featuring a bold "Testimonial" title in orange Abril Fatface font, followed by a warm message in Raleway font within a white speech box with rounded corners. Include the website URL in PT Serif font at the bottom.,
- Design a testimonial with yellow background, a central white text box with dotted border, Lato font for main message, Montserrat font for name "Olivia Wilson," and a blue underline, without an image.,
- Create a testimonial with a mint green background, featuring a bold red "CLIENT FEEDBACK" title at the top, a white rounded rectangle for the testimonial text, and include the customer name "OLIVIA WILSON" with five blue stars for a 5-star rating.,
- Design a testimonial with a centered title "CLIENT REVIEWS" in bold Courier Std font, italic Coromont Garamond text in a dark gray container, and five gold stars for rating, all on a clean white background.,
- Design a testimonial with Viaboda Libre title and Playfair Display font for positive feedback by "Rakhi Sawant," with left and right quote SVGs on a pale yellow background.,
- Design a testimonial with a blue background and a light blue header, featuring a bold "Testimonial" title in orange Abril Fatface font, followed by a warm message in Raleway font within a white speech box with rounded corners. Include the website URL in PT Serif font at the bottom.,
"""
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 1,
        "max_tokens": 4000
    }

    logger.info(f"Calling OpenAI Chat API for initial prompt enhancement with model: {PRE_ENHANCER_MODEL}")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response_data}")
        logger.error(f"Response status code: {response.status_code}")
        logger.error(f"Response headers: {response.headers}")
        raise Exception(f"OpenAI API error: {response_data.get('error', {}).get('message', 'Unknown error')}")

    enhanced_prompt = response_data["choices"][0]["message"]["content"]
    logger.info(f"Successfully enhanced prompt. Result: {enhanced_prompt[:100]}...")
    return enhanced_prompt

def enhance_prompt_with_chat(user_input):
    """Enhance user prompt using Chat Completions API"""
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

    payload = {
        "model": PROMPT_ENHANCER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a prompt enhancer assistant. You transform simple, brief prompts into detailed,
                comprehensive prompts that provide specific details, requirements, and context to help generate better results.
                
                For both coming soon pages and testimonial designs, ensure you include specific details about:
                - Layout and positioning
                - Font choices, sizes, and styles
                - Color schemes and background designs
                - Decorative elements and their placement
                - Text content and hierarchy
                - Spacing and alignment

                
                Add these requirements at the end of each prompt:
                'Compulsory in you use create good svg code must meaningfull and good and also usable for user ok msut look good'
                'Compulsory in you use any color must make sense and text color and and all continer bg color must visible togther'
                'Compulsory in This must you make all svg code must be center align in good aligmnet'
                'Compulsory IN THIS FETCH FONT USING LINK AND FONT FACE BOTH
                'Compulsory IN THIS ALIGMENT MUST BE GOOD AND GOOD LOOKING"
                '"""
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 1,
        "max_tokens": 4000
    }

    logger.info(f"Calling OpenAI Chat API for prompt enhancement with model: {PROMPT_ENHANCER_MODEL}")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response_data}")
        raise Exception(f"OpenAI API error: {response_data.get('error', {}).get('message', 'Unknown error')}")

    return response_data["choices"][0]["message"]["content"]

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
        return image_base64, filename
    except Exception as e:
        logger.error(f"Error generating image with GPT Image-1: {str(e)}")
        raise

def generate_svg_from_image(image_base64, enhanced_prompt):
    """Generate SVG code using GPT-4.1 based on image and prompt"""
    logger.info("Starting SVG generation from image")
    logger.info(f"Enhanced prompt length: {len(enhanced_prompt)}")
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }

    # Restored original system prompt
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

    # Create the image content
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_base64}"
        }
    }

    message_content = [
        {
            "type": "text",
            "text": "Generate SVG code (1080x1080) that matches this image exactly."
        },
        image_content
    ]

    payload = {
        "model": SVG_GENERATOR_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": message_content
            }
        ],
        "temperature": 1,  # Restored original temperature
        "max_tokens": 4000
    }

    logger.info("Generating SVG code with GPT-4.1")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"OpenAI API error in SVG generation: {response_data}")
        logger.error(f"Response status code: {response.status_code}")
        raise Exception(f"OpenAI API error: {response_data.get('error', {}).get('message', 'Unknown error')}")

    svg_content = response_data["choices"][0]["message"]["content"]
    logger.info(f"Successfully generated SVG code. Length: {len(svg_content)}")
    
    # Extract SVG code
    svg_pattern = r'<svg.*?<\/svg>'
    svg_matches = re.search(svg_pattern, svg_content, re.DOTALL)
    
    if svg_matches:
        logger.info("Successfully extracted SVG code from response")
        raw_svg = svg_matches.group(0)
        # Use original clean function with minimal changes
        formatted_svg = clean_svg_code_original(raw_svg)
        return formatted_svg
    
    logger.warning("Could not extract SVG pattern, attempting to clean raw content")
    formatted_svg = clean_svg_code_original(svg_content)
    return formatted_svg

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
    """Original SVG generator endpoint with high quality"""
    try:
        data = request.json
        user_input = data.get('prompt', '')
        skip_enhancement = data.get('skip_enhancement', False)

        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        logger.info(f"Processing prompt: {user_input[:50]}... Skip enhancement: {skip_enhancement}")

        if skip_enhancement:
            # Skip enhancement and use the original prompt directly
            prompt_to_use = user_input
            pre_enhanced_prompt = user_input
            enhanced_prompt = user_input
            logger.info(f"Using original prompt without enhancement: {prompt_to_use[:50]}...")
        else:
            # Step 1: Pre-enhance the prompt
            pre_enhanced_prompt = pre_enhance_prompt(user_input)
            logger.info(f"Pre-enhanced prompt: {pre_enhanced_prompt[:50]}...")

            # Step 2: Further enhance the prompt
            enhanced_prompt = enhance_prompt_with_chat(pre_enhanced_prompt)
            logger.info(f"Enhanced prompt: {enhanced_prompt[:50]}...")
            
            prompt_to_use = enhanced_prompt

        # Step 3: Generate image using GPT Image-1
        gpt_image_base64, gpt_image_filename = generate_image_with_gpt(prompt_to_use)
        logger.info("Image generated with GPT Image-1")

        # Step 4: Generate SVG using GPT-4.1
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
            "svg_url": f"/static/images/{svg_filename}"
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
        
        logger.info(f"Received chat request")
        logger.info(f"Chat history length: {len(messages)}")
        logger.info(f"Last message: {messages[-1] if messages else 'No messages'}")
        
        if not messages:
            logger.warning("No messages provided in request")
            return jsonify({"error": "No messages provided"}), 400

        # Get the latest user message
        latest_message = messages[-1]["content"].lower() if messages else ""
        
        # Check if this is a design creation request (new design from scratch)
        is_create_request = any(keyword in latest_message for keyword in [
            "create", "design", "generate", "make", "draw", "poster", "build"
        ]) and not any(word in latest_message for word in ["edit", "update", "modify", "change"])

        # Check if this is a design modification request
        is_modify_request = any(word in latest_message for word in ["edit", "update", "modify", "change", "adjust"]) and any(keyword in latest_message for keyword in ["design", "poster", "color", "text", "font", "size"])

        # Find the most recent SVG in the conversation
        current_svg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and "```svg" in msg.get("content", ""):
                svg_start = msg["content"].find("```svg") + 6
                svg_end = msg["content"].find("```", svg_start)
                if svg_end > svg_start:
                    current_svg = msg["content"][svg_start:svg_end].strip()
                    break

        if is_create_request:
            logger.info("Processing new design creation request")
            
            try:
                # Use the existing SVG generation pipeline
                pre_enhanced = pre_enhance_prompt(latest_message)
                enhanced_prompt = enhance_prompt_with_chat(pre_enhanced)
                
                # Generate image and SVG
                image_base64, image_filename = generate_image_with_gpt(enhanced_prompt)
                svg_code = generate_svg_from_image(image_base64, enhanced_prompt)
                svg_filename = save_svg(svg_code, prefix="assistant_svg")
                
                # Get AI explanation of the design
                explanation_prompt = f"I've created a design for the user. Here's the SVG code:\n\n```svg\n{svg_code}\n```\n\nPlease explain this design to the user in a friendly, conversational way. Describe the elements, colors, layout, and how it addresses their request."
                
                temp_messages = messages + [{"role": "user", "content": explanation_prompt}]
                ai_explanation = chat_with_ai_about_design(temp_messages, svg_code)
                
                # Create comprehensive response
                full_response = f"{ai_explanation}\n\n```svg\n{svg_code}\n```\n\nFeel free to ask me to modify any aspect of this design!"
                
                messages.append({"role": "assistant", "content": full_response})
                
                response_data = {
                    "messages": messages,
                    "svg_code": svg_code,
                    "svg_url": f"/static/images/{svg_filename}"
                }
                logger.info("Successfully generated new design with explanation")
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
                        "messages": messages,
                        "svg_code": modified_svg,
                        "svg_url": f"/static/images/{svg_filename}"
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
                "svg_url": None
            })
            
    except Exception as e:
        error_msg = f"Error in chat_assistant: {str(e)}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets PORT=8000)
    port = int(os.getenv('PORT', 5001))
    
    # Use 0.0.0.0 for production (Render) and 127.0.0.1 for local development
    host = '0.0.0.0' if os.getenv('PORT') else '127.0.0.1'
    
    # Disable debug mode in production
    debug = not bool(os.getenv('PORT'))
    
    logger.info(f"Starting Flask application on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
