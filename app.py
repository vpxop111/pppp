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

# ----------- CORS CONFIGURATION (NEW & CORRECT) ------------
CORS(
    app,
    origins=[
        "https://info12.vercel.app",          # <-- your deployed site
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    supports_credentials=True,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

# Explicit OPTIONS handler for preflight on Render, etc.
@app.route("/api/chat-assistant", methods=["OPTIONS"])
def chat_assistant_options():
    return '', 204

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

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
IMAGES_DIR = os.path.join(STATIC_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

OPENAI_API_KEY_ENHANCER = os.getenv('OPENAI_API_KEY_ENHANCER')
OPENAI_API_KEY_SVG = os.getenv('OPENAI_API_KEY_SVG')
if not OPENAI_API_KEY_ENHANCER or not OPENAI_API_KEY_SVG:
    raise ValueError("OpenAI API keys must be set in environment variables")

openai.api_key = OPENAI_API_KEY_SVG

OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_CHAT_ENDPOINT = f"{OPENAI_API_BASE}/chat/completions"
PRE_ENHANCER_MODEL = "gpt-4o-mini"
PROMPT_ENHANCER_MODEL = "gpt-4o-mini"
GPT_IMAGE_MODEL = "gpt-image-1"
SVG_GENERATOR_MODEL = "gpt-4.1-mini"
CHAT_ASSISTANT_MODEL = "gpt-4o-mini"

def pre_enhance_prompt(user_input):
    # [Unchanged logic...]
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
- ... [and all other examples, unchanged] ...
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
    # [Unchanged logic...]
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
                "content": """You are a prompt enhancer assistant. ...[unchanged prompt, truncated for brevity]..."""
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
    try:
        logger.info("Generating image with GPT Image-1")
        response = openai.images.generate(
            model=GPT_IMAGE_MODEL,
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="low"
        )
        image_base64 = response.data[0].b64_json if hasattr(response.data[0], 'b64_json') else response.data[0].url
        filename = save_image(image_base64, prefix="gpt_image")
        logger.info("Image generated and saved successfully with GPT Image-1")
        return image_base64, filename
    except Exception as e:
        logger.error(f"Error generating image with GPT Image-1: {str(e)}")
        raise

def generate_svg_from_image(image_base64, enhanced_prompt):
    logger.info("Starting SVG generation from image")
    logger.info(f"Enhanced prompt length: {len(enhanced_prompt)}")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }
    system_prompt = """You are an expert SVG code generator. ...[unchanged prompt, truncated for brevity]..."""
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
        "temperature": 1,
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
    svg_pattern = r'<svg.*?<\/svg>'
    svg_matches = re.search(svg_pattern, svg_content, re.DOTALL)
    if svg_matches:
        logger.info("Successfully extracted SVG code from response")
        raw_svg = svg_matches.group(0)
        formatted_svg = clean_svg_code_original(raw_svg)
        return formatted_svg
    logger.warning("Could not extract SVG pattern, attempting to clean raw content")
    formatted_svg = clean_svg_code_original(svg_content)
    return formatted_svg

def clean_svg_code_original(svg_code):
    try:
        from xml.dom.minidom import parseString
        from xml.parsers.expat import ExpatError
        try:
            doc = parseString(svg_code)
            svg_element = doc.documentElement
            if not svg_element.hasAttribute('viewBox'):
                svg_element.setAttribute('viewBox', '0 0 1080 1080')
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
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.{format.lower()}"
        filepath = os.path.join(IMAGES_DIR, filename)
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image.save(filepath, format=format)
        logger.info(f"Image saved successfully: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        raise

def save_svg(svg_code, prefix="svg"):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.svg"
        filepath = os.path.join(IMAGES_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg_code)
        logger.info(f"SVG saved successfully: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving SVG: {str(e)}")
        raise

def convert_svg_to_png(svg_code):
    try:
        svg_filename = save_svg(svg_code)
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
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
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/api/generate-svg', methods=['POST'])
def generate_svg():
    try:
        data = request.json
        user_input = data.get('prompt', '')
        skip_enhancement = data.get('skip_enhancement', False)
        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400
        logger.info(f"Processing prompt: {user_input[:50]}... Skip enhancement: {skip_enhancement}")
        if skip_enhancement:
            prompt_to_use = user_input
            pre_enhanced_prompt = user_input
            enhanced_prompt = user_input
            logger.info(f"Using original prompt without enhancement: {prompt_to_use[:50]}...")
        else:
            pre_enhanced_prompt = pre_enhance_prompt(user_input)
            logger.info(f"Pre-enhanced prompt: {pre_enhanced_prompt[:50]}...")
            enhanced_prompt = enhance_prompt_with_chat(pre_enhanced_prompt)
            logger.info(f"Enhanced prompt: {enhanced_prompt[:50]}...")
            prompt_to_use = enhanced_prompt
        gpt_image_base64, gpt_image_filename = generate_image_with_gpt(prompt_to_use)
        logger.info("Image generated with GPT Image-1")
        svg_code = generate_svg_from_image(gpt_image_base64, prompt_to_use)
        logger.info("SVG code generated from image")
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
    logger.info("Starting conversational AI interaction")
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }
    system_prompt = """You are an expert AI design assistant with deep knowledge of SVG
    ... [truncated for brevity] ..."""
    if current_svg:
        system_prompt += f"\n\nCurrent SVG design context:\n```svg\n{current_svg}\n```\n\nYou can reference and modify this design based on user requests."
    ai_messages = [{"role": "system", "content": system_prompt}]
    conversation_messages = messages[-10:] if len(messages) > 10 else messages
    for msg in conversation_messages:
        if msg["role"] in ["user", "assistant"]:
            content = msg["content"]
            if "```svg" in content and msg["role"] == "assistant":
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
    logger.info(f"Modifying SVG with request: {modification_request}")
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }
    system_prompt = """You are an expert SVG modifier. ... (unchanged prompt) ..."""
    payload = {
        "model": SVG_GENERATOR_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original SVG:\n```svg\n{original_svg}\n```\n\nModification request: {modification_request}\n\nPlease provide the modified SVG:"}
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
        latest_message = messages[-1]["content"].lower() if messages else ""
        is_create_request = any(keyword in latest_message for keyword in [
            "create", "design", "generate", "make", "draw", "poster", "build"
        ]) and not any(word in latest_message for word in ["edit", "update", "modify", "change"])
        is_modify_request = any(word in latest_message for word in ["edit", "update", "modify", "change", "adjust"]) and any(keyword in latest_message for keyword in ["design", "poster", "color", "text", "font", "size"])
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
                pre_enhanced = pre_enhance_prompt(latest_message)
                enhanced_prompt = enhance_prompt_with_chat(pre_enhanced)
                image_base64, image_filename = generate_image_with_gpt(enhanced_prompt)
                svg_code = generate_svg_from_image(image_base64, enhanced_prompt)
                svg_filename = save_svg(svg_code, prefix="assistant_svg")
                explanation_prompt = f"I've created a design for the user. Here's the SVG code:\n\n```svg\n{svg_code}\n```\n\nPlease explain this design to the user in a friendly, conversational way. Describe the elements, colors, layout, and how it addresses their request."
                temp_messages = messages + [{"role": "user", "content": explanation_prompt}]
                ai_explanation = chat_with_ai_about_design(temp_messages, svg_code)
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
                modified_svg = modify_svg_with_ai(current_svg, latest_message)
                if modified_svg and modified_svg != current_svg:
                    svg_filename = save_svg(modified_svg, prefix="modified_svg")
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
                    ai_response = chat_with_ai_about_design(messages, current_svg)
                    messages.append({"role": "assistant", "content": ai_response})
                    return jsonify({"messages": messages})
            except Exception as e:
                logger.error(f"Error in design modification: {str(e)}")
                ai_response = "I had trouble modifying the design. Could you be more specific about what changes you'd like me to make?"
                messages.append({"role": "assistant", "content": ai_response})
                return jsonify({"messages": messages})
        else:
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
    port = int(os.getenv('PORT', 5001))
    host = '0.0.0.0' if os.getenv('PORT') else '127.0.0.1'
    debug = not bool(os.getenv('PORT'))
    logger.info(f"Starting Flask application on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
