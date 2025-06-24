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
# import vtracer  # Add vtracer import - temporarily disabled due to Render compilation issues

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

# OpenAI client setupkk
openai.api_key = OPENAI_API_KEY_SVG

# OpenAI API Endpoints
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_CHAT_ENDPOINT = f"{OPENAI_API_BASE}/chat/completions"

# Model names - updated to use GPT-4.1 mini for logic/text and gpt-image for images
PLANNER_MODEL = "gpt-4.1-nano"
DESIGN_KNOWLEDGE_MODEL = "gpt-4.1-nano"
PRE_ENHANCER_MODEL = "gpt-4.1-nano"
PROMPT_ENHANCER_MODEL = "gpt-4.1-nano"
GPT_IMAGE_MODEL = "gpt-image-1"
SVG_GENERATOR_MODEL = "gpt-4.1-nano"
CHAT_ASSISTANT_MODEL = "gpt-4.1-nano"

# Add parallel SVG processing imports
from concurrent.futures import ThreadPoolExecutor
import pytesseract
import numpy as np

# Add after existing imports
try:
    import vtracer
    import remove_text_simple
    import png_to_svg_converter
    PARALLEL_FEATURES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Parallel SVG features not available: {e}")
    PARALLEL_FEATURES_AVAILABLE = False

# Add after existing directory setup
PARALLEL_OUTPUTS_DIR = os.path.join(IMAGES_DIR, 'parallel')
os.makedirs(PARALLEL_OUTPUTS_DIR, exist_ok=True)

def check_vector_suitability(user_input):
    """Check if the prompt is suitable for SVG vector graphics"""
    logger.info(f"Checking vector suitability for: {user_input[:100]}...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

    payload = {
        "model": PLANNER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a vector graphics expert. Your task is to determine if a design request is suitable for SVG vector graphics.

Guidelines for SVG suitability:
1. Ideal for logos, icons, illustrations, typography, and geometric designs
2. Good for flat or minimalist designs
3. Suitable for designs with clear shapes and paths
4. Works well with text and typography
5. Perfect for scalable graphics without loss of quality

Not suitable for:
1. Photorealistic images
2. Complex textures and gradients
3. Designs requiring many minute details
4. Photographs or photo manipulations
5. Complex 3D renderings

Provide guidance if the request isn't suitable."""
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"Vector suitability check error: {response_data}")
        return {"not_suitable": False}  # Default to allowing if check fails

    analysis = response_data["choices"][0]["message"]["content"].lower()
    not_suitable = "not suitable" in analysis or "unsuitable" in analysis
    
    return {
        "not_suitable": not_suitable,
        "guidance": response_data["choices"][0]["message"]["content"] if not_suitable else None
    }

def plan_design(user_input):
    """Plan the design approach based on user input with enhanced focus on image generation quality"""
    logger.info(f"Planning design for: {user_input[:100]}...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

    # Use simplified, practical planning approach focused on PERFECT CENTRALIZATION
    system_content = """You are a practical design planner specializing in PERFECTLY CENTERED layouts. Create a clear, actionable plan for the design request that ensures ALL elements are perfectly centered.

CRITICAL REQUIREMENT: ALL DESIGNS MUST BE PERFECTLY CENTERED - Every element must be centrally aligned both horizontally and vertically.

Your plan should include:
1. CENTERED Layout Structure
   - Overall composition: PERFECTLY CENTERED approach (all elements centered)
   - Key elements placement: ALL CENTERED and symmetrically positioned
   - Visual hierarchy: CENTERED focal points with balanced distribution
   - Symmetrical design with perfect center alignment

2. CENTERED Typography Specifications
   - Specific font recommendations (Google Fonts preferred) - ALL TEXT CENTERED
   - Font sizes and weights for different text elements - ALL CENTERED
   - Text alignment: PERFECT CENTER ALIGNMENT for all text
   - Spacing: Symmetrical spacing around centered elements

3. Color Scheme for Centered Design
   - Primary background color that enhances centered content
   - Text colors for readability with centered alignment
   - Accent colors for highlights that support center focus
   - Specific hex codes when possible

4. CENTERED Content Elements
   - Main heading/title: PERFECTLY CENTERED treatment
   - Secondary text: CENTERED placement below main title
   - Decorative elements: SYMMETRICALLY PLACED around center
   - Brand elements: CENTERED if applicable
   - All content arranged in perfect center alignment

5. Technical Requirements for Centered Design
   - Dimensions and aspect ratio optimized for center alignment
   - SVG viewBox centered at 0 0 1080 1080
   - Quality standards ensuring crisp centered output
   - Perfect center positioning coordinates

Focus on creating a practical, implementable plan where EVERYTHING is perfectly centered - no element should be off-center."""

    payload = {
        "model": PLANNER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.8,
        "max_tokens": 1200
    }

    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"Design planning error: {response_data}")
        return "Error in design planning"

    return response_data["choices"][0]["message"]["content"]

def generate_design_knowledge(design_plan, user_input):
    """Generate specific design knowledge based on the plan and user input with focus on stunning visuals"""
    logger.info("Generating design knowledge...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

    # Use practical, actionable design knowledge approach focused on PERFECT CENTRALIZATION
    system_content = """You are a practical design knowledge expert specializing in PERFECTLY CENTERED designs. Provide specific, actionable design insights and best practices for creating perfectly centered layouts.

CRITICAL FOCUS: ALL advice must emphasize PERFECT CENTER ALIGNMENT - every element must be perfectly centered.

Provide practical knowledge for:

1. CENTERED Typography Best Practices
   - Recommended font combinations that work well together when PERFECTLY CENTERED
   - Optimal font sizes for different screen sizes with CENTER ALIGNMENT
   - Proper line spacing and letter spacing for CENTERED text
   - Font loading and fallback strategies for CENTERED layouts
   - Center text alignment techniques and positioning

2. Color Theory for CENTERED Designs
   - Proven color combinations with hex codes that enhance CENTERED content
   - Contrast ratios for accessibility compliance in CENTERED layouts
   - Background and text color pairings that support CENTER FOCUS
   - Brand-appropriate color choices for PERFECTLY CENTERED designs

3. CENTERED Layout and Composition
   - CENTER ALIGNMENT principles - everything must be centered
   - Symmetrical grid systems for PERFECT CENTER ALIGNMENT
   - White space utilization around CENTERED content for clarity
   - Visual hierarchy techniques with CENTERED focal points
   - CENTERED responsive design considerations

4. Technical Implementation for CENTERED SVGs
   - SVG center positioning techniques (x="50%", text-anchor="middle")
   - Perfect center coordinates calculation (540, 540 for 1080x1080)
   - CENTER ALIGNMENT optimization techniques
   - Cross-browser compatibility for CENTERED elements
   - Performance optimization for CENTERED layouts

5. CENTERED Design Quality Assurance
   - CENTER ALIGNMENT consistency checkpoints
   - Verification that ALL elements are perfectly centered
   - Accessibility standards for CENTERED content
   - User experience considerations for CENTERED designs
   - Testing and validation for PERFECT CENTER ALIGNMENT

Focus on providing concrete, implementable advice that ensures PERFECT CENTER ALIGNMENT and directly improves centered design quality."""

    payload = {
        "model": DESIGN_KNOWLEDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": f"Design Plan:\n{design_plan}\n\nUser Request:\n{user_input}"
            }
        ],
        "temperature": 0.8,
        "max_tokens": 1800
    }

    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"Design knowledge generation error: {response_data}")
        return "Error in generating design knowledge"

    return response_data["choices"][0]["message"]["content"]

def pre_enhance_prompt(user_input):
    """Initial enhancement of user query using proven examples and detailed specifications"""
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

CRITICAL REQUIREMENT: ALL DESIGNS MUST BE PERFECTLY CENTERED - Every element, text, image, and component must be centrally aligned both horizontally and vertically. The entire composition must follow a centered approach.

Don't add custom elements, shapes, and random figures in prompts.
You must generate a prompt same as given below examples


Examples for Coming Soon Pages (ALL CENTERED): -

- Design a clean and elegant coming soon page with a black rectangular border, PERFECTLY CENTERED "Coming Soon" text in a cursive font, and a white background using Water Brush font at 60px size, with all elements centrally aligned.
- Design a modern coming soon page with a sleek black background, a prominent complex SVG graphic CENTERED, centralized layout, minimalist text CENTERED, and seamless integration of decorative SVG elements all perfectly centered.
- Design a stylish coming soon page with a soft pink background, golden brown border, cursive "something" in Allura font CENTERED, and main text in Times New Roman font CENTERED with heart symbols and website link all centrally positioned.
- Create a coming soon page with a light beige background, dark gray content area featuring large white text for "COMING SOON" PERFECTLY CENTERED, a website URL CENTERED, and a "GRAND OPENING" button styled in green with custom fonts CENTRALLY PLACED.
- Design a natural-themed coming soon page with a dark green background, featuring Bebas Neue font in large size for "Coming" and "soon" PERFECTLY CENTERED, a countdown section CENTERED, and decorative elements positioned symmetrically around the center.
- Design a coming soon page with a deep blue background, featuring PERFECTLY CENTERED text in Tektur font (white for 'COMING' with shadow, orange for 'SOON' with shadow), and white lines for definition, all elements perfectly aligned to center.
- Design a warm and welcoming coming soon page with a beige background, PERFECTLY CENTERED layout, modern font for the title CENTERED, cursive and bold fonts for 'Coming Soon' CENTERED, and simple font for a website link CENTERED, including decorative SVG elements symmetrically placed around center.
- Design a warm and inviting coming soon page with a light beige background, bold 'COMING SOON' text PERFECTLY CENTERED in a darker brown color, and a date below CENTERED in 'Open Sans' font.
- Design a playful coming soon page with a cream background, grid pattern, bold text in warm taupe CENTERED, cursive text in dark green CENTERED, and sans-serif font for additional information CENTERED, ensuring a PERFECTLY CENTERED layout for clarity and appeal.
- Design an elegant coming soon page with beige background, featuring 'LARANA STORE' PERFECTLY CENTERED in bold serif font, 'BEAUTY PRODUCT' in a decorative rectangle CENTERED, and 'COMING SOON' in red and cursive font CENTERED, with 'STAY TUNED' and '@REALLYGREATSITE' CENTERED, along with whimsical star shapes symmetrically placed around the center.
- Design a bold and modern coming soon page with black background, gray border, large Bebas Neue text for "COMING" and "SOON" PERFECTLY CENTERED, decorative Allura font for "-Best Store-" and "Stay Tuned" CENTERED, and a date CENTERED, with clear website link CENTERED at the bottom.
- Design a clean coming soon page with a white background, a beige box with a grey border CENTERED, 'LICERIA & CO.' at the top CENTERED in large dark blue-grey text, 'WE ARE' below CENTERED in smaller text, 'OPEN' PERFECTLY CENTERED, and 'OPENING HOURS' with hours displayed CENTERED below.
- Design a modern and minimalistic coming soon page with a clean white background, featuring PERFECTLY CENTERED text in Open Sans font and time indicators styled in dark gray, all elements centrally aligned.
- Design a coming soon page with PERFECTLY CENTERED text elements in black color on a soft pink background, featuring the phrases 'NOW WE ARE', 'OPEN', and 'VISIT OUR SITE' with specific font styles and sizes ALL CENTERED. Include a website link at the bottom CENTERED in a regular font.

Examples for Testimonial Designs (ALL CENTERED): -

- Design a testimonial graphic with a teal background, featuring a beige square container PERFECTLY CENTERED with "TESTIMONIAL" in bold Alfarn font CENTERED, three orange circles below CENTERED, PT Serif font for customer experience lines CENTERED, and square quotation marks CENTERED for visual appeal.
- Design a testimonial with a white background, a large pink circle PERFECTLY CENTERED, testimonial text in Raleway font CENTERED, customer's name at the bottom CENTERED, and decorative elements like 4-spoke stars and a dotted circle symmetrically placed around center.
- Create a testimonial with a cream background, black quotation marks CENTERED, PERFECTLY CENTERED title "Testimonial," testimonial text "We couldn't be happier with the outstanding service provided by Weblake Company..." CENTERED, author name "- Linda Brown -" PERFECTLY CENTERED below, and website URL CENTERED at the bottom in Instrument Serif font.
- Design a testimonial with a neon green header CENTERED, black background, and round corner speech box CENTERED, featuring the title "CLIENT TESTIMONIAL" in Bebas Neue font at 80px PERFECTLY CENTERED, testimonial text in Neue font at 42px CENTERED, and name "MIHIR GEHLOT" in Raleway font at 36px CENTERED, all elements perfectly aligned to center.
- Design a testimonial with a blue background and a light blue header CENTERED, featuring a bold "Testimonial" title in orange Abril Fatface font PERFECTLY CENTERED, followed by a warm message in Raleway font within a white speech box with rounded corners CENTERED. Include the website URL in PT Serif font CENTERED at the bottom.
- Design a testimonial with yellow background, a PERFECTLY CENTERED white text box with dotted border, Lato font for main message CENTERED, Montserrat font for name "Olivia Wilson" CENTERED, and a blue underline CENTERED, without an image.
- Create a testimonial with a mint green background, featuring a bold red "CLIENT FEEDBACK" title PERFECTLY CENTERED at the top, a white rounded rectangle CENTERED for the testimonial text, and include the customer name "OLIVIA WILSON" CENTERED with five blue stars CENTERED for a 5-star rating.
- Design a testimonial with a PERFECTLY CENTERED title "CLIENT REVIEWS" in bold Courier Std font, italic Coromont Garamond text in a dark gray container CENTERED, and five gold stars for rating CENTERED, all on a clean white background.
- Design a testimonial with Viaboda Libre title PERFECTLY CENTERED and Playfair Display font for positive feedback by "Rakhi Sawant" CENTERED, with left and right quote SVGs symmetrically placed around the centered content on a pale yellow background.
- Design a testimonial with a blue background and a light blue header CENTERED, featuring a bold "Testimonial" title in orange Abril Fatface font PERFECTLY CENTERED, followed by a warm message in Raleway font within a white speech box with rounded corners CENTERED. Include the website URL in PT Serif font CENTERED at the bottom.
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
    """Enhance user prompt using Chat Completions API with proven quality requirements"""
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
                
                CRITICAL CENTRALIZATION REQUIREMENT: ALL DESIGNS MUST BE PERFECTLY CENTERED. Every single element must be centrally aligned both horizontally and vertically.
                
                For both coming soon pages and testimonial designs, ensure you include specific details about:
                - Layout and positioning (ALL CENTERED)
                - Font choices, sizes, and styles (ALL CENTERED)
                - Color schemes and background designs
                - Decorative elements and their placement (SYMMETRICALLY CENTERED)
                - Text content and hierarchy (ALL CENTERED)
                - Spacing and alignment (PERFECT CENTER ALIGNMENT)

                
                Add these requirements at the end of each prompt:
                'Compulsory: ALL content must be PERFECTLY CENTERED - every text, image, shape, and element must be centrally aligned both horizontally and vertically'
                'Compulsory: Create good svg code that is meaningful, good, and also usable for user - must look good with PERFECT CENTER ALIGNMENT'
                'Compulsory: Use colors that make sense and text color and all container bg colors must be visible together'
                'Compulsory: ALL SVG code must be PERFECTLY CENTER ALIGNED with excellent alignment - no element should be off-center'
                'Compulsory: Fetch fonts using both LINK and FONT FACE methods'
                'Compulsory: Alignment must be PERFECT CENTER ALIGNMENT and good looking - everything centered'
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

def enhance_prompt_for_gpt_image(user_prompt, design_context=None):
    """Enhance user prompt using OpenAI specifically for GPT Image-1 to create mind-blowing designs"""
    logger.info(f"Enhancing prompt for GPT Image-1: {user_prompt[:100]}...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

    # Analyze prompt to determine design type for specialized enhancement
    prompt_lower = user_prompt.lower()
    is_coming_soon = any(word in prompt_lower for word in ['coming soon', 'coming', 'soon', 'announcement', 'launch', 'reveal'])
    is_testimonial = any(word in prompt_lower for word in ['testimonial', 'review', 'quote', 'feedback', 'recommendation'])
    is_poster = any(word in prompt_lower for word in ['poster', 'flyer', 'announcement', 'event'])

    # Create specialized system prompt based on design type
    if is_coming_soon:
        system_prompt = """You are an expert prompt enhancer for GPT Image-1, specializing in creating MIND-BLOWING "Coming Soon" posters that stop viewers in their tracks. Transform user requests into detailed, specific prompts that will generate visually stunning coming soon designs.

Your enhanced prompts should focus on:
1. VISUAL IMPACT: Eye-catching elements, dramatic lighting, bold compositions
2. TYPOGRAPHY: Massive, attention-grabbing "COMING SOON" text with premium fonts
3. COLOR SCHEMES: Vibrant gradients, neon accents, or sophisticated palettes
4. BACKGROUND ELEMENTS: Dynamic textures, abstract patterns, or cinematic backdrops
5. MOOD & ATMOSPHERE: Excitement, anticipation, premium quality, exclusivity
6. TECHNICAL SPECS: 1024x1024, high contrast, GPT Image-1 optimized

Enhancement Guidelines:
- Use words like "explosive", "dramatic", "premium", "cutting-edge", "revolutionary"
- Specify exact color palettes with hex codes when possible
- Include lighting effects (neon glow, dramatic shadows, spotlights)
- Add texture details (metallic, glass, holographic, matte)
- Specify typography styles (futuristic, bold sans-serif, modern)
- Include composition details (centered, asymmetrical, dynamic)

Return ONLY the enhanced prompt optimized for GPT Image-1, no explanations."""

    elif is_testimonial:
        system_prompt = """You are an expert prompt enhancer for GPT Image-1, specializing in creating MIND-BLOWING testimonial posters that build trust and credibility while being visually stunning. Transform user requests into detailed prompts for generating powerful testimonial designs.

Your enhanced prompts should focus on:
1. TRUST ELEMENTS: Professional layouts, clean typography, credible visual design
2. VISUAL HIERARCHY: Clear quote presentation, prominent attribution, star ratings
3. COLOR PSYCHOLOGY: Trust-building colors (blues, whites, golds), sophisticated palettes
4. BACKGROUND DESIGN: Clean, professional, or subtly branded backgrounds
5. TYPOGRAPHY: Readable quote fonts, professional attribution text, emphasis on key phrases
6. CREDIBILITY INDICATORS: Stars, badges, professional photos, company logos
7. TECHNICAL SPECS: 1024x1024, high readability, clean composition

EXAMPLE TESTIMONIAL PROMPTS FROM TRAINING DATA:

Example 1: "Create a testimonial with a teal background featuring a beige square container at the center. The layout includes a large bold 'TESTIMONIAL' text at the top, styled with the Alfarn font, placed in the middle of the container. Below, there are three decorative circles in orange arranged horizontally, adding a playful touch. The body text, written in PT Serif font, is centered and displays a series of lines about customer experience, ensuring a clear and engaging read. Finally, include square quotation marks on either side of the testimonial text to enhance the visual appeal and authenticity. **Image:** No image **Fonts:** Alfarn, PT Serif, Aileron **Custom Elements:** Square quotation marks *2"

Example 2: "Create a testimonial with a white background featuring a large pink circle at the center. The layout includes testimonial text placed centrally with multiple lines, and the customer's name at the bottom. Decoratively, there are four 4-spoke stars located in each corner of the design and a dotted circle element subtly integrated around the testimonial. The text is styled using the Raleway font, with the testimonial text in a size of 42, and the customer's name in a larger size of 48, both in black and dark blue respectively. The design maintains a clean, professional appearance suitable for showcasing customer feedback. **Image:** No image **Fonts:** Raleway **Custom Elements:** 4 spoke star *4, dotted circle"

Example 3: "Create a testimonial with a vibrant green background, featuring a large dashed border square that is softly rounded at the corners. Inside the square, use bold Arvo font in a rich reddish-brown color for the title 'What They Say!' positioned at the top center. Below the title, add a testimonial text in Raleway font, styled in dark gray, conveying a positive review about customer service and quality furnishings. At the bottom of the square, include the name 'Cezzane Ali' in a slightly larger size of Raleway font, colored in the same reddish-brown as the title. Decorate the layout with six small circular dots in a warm peach color, positioned above and below the testimonial text, and include a subtle grid pattern in the background for added depth. **Custom Elements:** Dotted rounded corner square, Circular dots (6), Grid pattern made up of line squares on the right **Fonts:** Arvo, Raleway **Image:** No image"

Example 4: "Create a testimonial with a deep navy blue background (#2C3E50) adorned with vibrant coral accents (#FF7F50) for the text and decorative elements, featuring a centered layout that includes a prominent title 'PROFESSIONAL BUSINESS' in bold white (#F5F5DC) and a subtitle 'Testimonials' in coral. The design includes two testimonials, each with a circular user icon and decorative quotation marks. The first testimonial showcases Victoria Wotton, an Entrepreneur, with an italicized quote stating, 'takes care of everything that you need to build an unshakable and brand in this new world.' The second testimonial features Matthew Smith, also an Entrepreneur, sharing his experience with the quote, 'Innovative and unique work. me continuous learning.' **Custom Elements:** Circular quotation marks (4), Rectangle speech boxes (2) **Fonts:** Montserrat (used for titles, names, and testimonial text) **Image:** No image"

Example 5: "Create a testimonial with a light beige background (#E5E0DB) adorned with playful sparkles in orange (#FFA726), featuring a prominent title reading 'CUSTOMER TESTIMONIAL' in Bebas Neue font at the top. Use Aileron font for the customer's name 'Nelly Dean' in a bold style, followed by a star rating represented by five golden stars (#827717) beneath the name. The review text should be styled in Aileron font, conveying a positive message with phrases like 'Awesome! This's very good product with affordable price. Good Job! Thank You.' Below, incorporate a wavy green shape (#556B2F) at the bottom of the image, and include a phone mockup with makeup tools and small stars' icons displayed on the screen. Finally, add the email address 'WWW.ORGANICSERUM@GMAIL.COM' in Amatic SC font, centered at the bottom in white. **Custom Elements:** ['4 spoke star * 1', '4 spoke small star * 2', 'Green color wavy icon on the bottom', 'makeup tools with small stars' icon'] **Fonts:** ['Bebas Neue', 'Aileron', 'Amatic SC'] **Image:** ['phone in hand']"

Example 6: "Create a testimonial with a white background featuring a subtle diamond pattern in light grey, a rounded rectangular header box in deep navy blue (#152238) with bold white text stating 'CLIENT TESTIMONIAL', and a vibrant orange main content box (#F39C12) below. The testimonial should include a star rating displayed with five white stars, four fully filled and one partially filled, along with the testimonial text saying 'Stylish and well-crafted furniture adding elegance to any living space.' in elegant white font. Below the text, include the client name 'Robert William' along with a decorative line underneath. At the bottom, feature a rounded rectangular bar in orange with the text 'AMAZON' in bold white and a smaller text for the website 'Visit our website www.flyfurniture@gmail.com'. **Custom Elements:** 5 spoke stars (5), small lines (1), line (1) **Fonts:** Poppins (used for headers, testimonials, and client names) **Image:** No image"

Example 7: "Create a testimonial with a soft pink background (#D8A1C4) and a central cream area (#FFF1E6), featuring a bold title in Raleway font in coral (#FF7F50) reading 'CLIENT' and 'TESTIMONIALS'. The layout includes decorative half circles in coral at the sides, and a circular user photo area at the center. The testimonial text, styled in Poppins font, is presented within a rectangular frame with rounded corners, filled with no color and outlined in coral, highlighting a heartfelt review: 'Thanks to this product, I feel more confident and radiant than ever, and I am excited to continue using this amazing product in my facial care routine. Thank you for creating something so wonderful and transformative!' The review is concluded with a star rating in a deep blue (#4169E1) and signed by 'Jones Williams' in an italicized Raleway font. **Custom Elements:** ['5 spoke star *5', 'Hemisphere *2', 'Rectangular frame around body text'] **Fonts:** ['Raleway', 'Poppins'] **Image:** ['custome image']"

Enhancement Guidelines:
- Use descriptive, specific language like the examples above
- Include exact color codes (#RRGGBB format)
- Specify font names and sizes precisely
- Detail custom decorative elements (stars, shapes, icons)
- Describe layout structure and positioning
- Include background treatments and visual effects
- Follow the format: detailed description + **Custom Elements:** + **Fonts:** + **Image:**

Return ONLY the enhanced prompt optimized for GPT Image-1, no explanations."""

    elif is_poster:
        system_prompt = """You are an expert prompt enhancer for GPT Image-1, specializing in creating MIND-BLOWING posters that capture attention and communicate effectively. Transform user requests into detailed prompts for generating visually stunning poster designs.

Your enhanced prompts should focus on:
1. VISUAL IMPACT: Bold compositions, striking imagery, attention-grabbing elements
2. TYPOGRAPHY: Powerful headlines, clear hierarchy, readable text
3. COLOR SCHEMES: Vibrant, purposeful color palettes that match the message
4. LAYOUT: Balanced composition, clear focal points, effective use of space
5. MOOD & MESSAGE: Appropriate atmosphere that supports the poster's purpose
6. TECHNICAL SPECS: 1024x1024, high contrast, print-ready quality

Enhancement Guidelines:
- Use descriptive words for visual impact and mood
- Specify exact color palettes and typography styles
- Include composition and layout details
- Add texture and effect specifications
- Focus on the poster's purpose and target audience

Return ONLY the enhanced prompt optimized for GPT Image-1, no explanations."""

    else:
        system_prompt = """You are an expert prompt enhancer for GPT Image-1, specializing in creating MIND-BLOWING graphic designs that are visually stunning and professionally crafted. Transform user requests into detailed prompts optimized for GPT Image-1.

Your enhanced prompts should focus on:
1. VISUAL EXCELLENCE: Premium quality, professional finish, striking aesthetics
2. COMPOSITION: Balanced layouts, clear hierarchy, effective use of space
3. COLOR & STYLE: Sophisticated palettes, modern aesthetics, brand-appropriate
4. TYPOGRAPHY: Professional fonts, readable text, proper hierarchy
5. TECHNICAL QUALITY: 1024x1024, high resolution, crisp details
6. PURPOSE: Design that serves its intended function effectively

EXAMPLE ENHANCED PROMPTS FROM TRAINING DATA:

Example 1: "Create a vibrant testimonial with a dark teal background and golden sun rays on the sides, featuring elegant white text in Josefin Sans font and bold Bebas Neue font at the bottom. The rich dark teal background (#004B49) is adorned with decorative sun rays in golden yellow (#DAA520) on both left and right sides, creating a vibrant and uplifting atmosphere."

Example 2: "Create a modern coming soon poster with a sleek black background (#000000) and a soft light gray content area (#F5F5F5), featuring a bold title in Alegreya font that reads 'COMING SOON' prominently at the top. Include dramatic lighting effects with neon glow accents and metallic texture details."

Example 3: "Create a testimonial with a soft beige background (#F5E6E0) adorned with decorative concentric circles in the top right and bottom left corners, a centered title reading 'CLIENT REVIEW' in bold Philosopher font, featuring a detailed review styled with Lato font for the review text, and a star rating illustrated with 5 brown stars (#8B4513)."

Enhancement Guidelines:
- Use descriptive, specific language like the examples above
- Include exact color codes (#RRGGBB format) when possible
- Specify font names, sizes, and visual effects precisely
- Detail layout structure, positioning, and decorative elements
- Include technical specifications for optimal GPT Image-1 output
- Describe background treatments, lighting effects, and textures
- Focus on professional, high-quality results with specific visual details

Return ONLY the enhanced prompt optimized for GPT Image-1, no explanations."""

    # Prepare the user content
    user_content = f"Original request: {user_prompt}"
    if design_context:
        user_content += f"\n\nDesign context: {design_context[:300]}..."

    payload = {
        "model": PROMPT_ENHANCER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "temperature": 0.8,
        "max_tokens": 800
    }

    try:
        logger.info(f"Calling OpenAI for GPT Image-1 prompt enhancement with model: {PROMPT_ENHANCER_MODEL}")
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"OpenAI API error for GPT Image enhancement: {response_data}")
            # Fallback to original prompt if API fails
            return user_prompt

        enhanced_prompt = response_data["choices"][0]["message"]["content"].strip()
        
        # Ensure prompt isn't too long for GPT Image-1
        if len(enhanced_prompt) > 1000:
            enhanced_prompt = enhanced_prompt[:1000] + "..."
        
        logger.info(f"Successfully enhanced prompt for GPT Image-1: {enhanced_prompt[:100]}...")
        return enhanced_prompt

    except Exception as e:
        logger.error(f"Error enhancing prompt for GPT Image-1: {str(e)}")
        # Return original prompt as fallback
        return user_prompt

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
    """Generate SVG code from image - vtracer temporarily disabled for deployment"""
    logger.info("SVG generation from image requested - vtracer temporarily disabled")
    
    # For now, return a message indicating vtracer is disabled
    # In the future, you can implement an alternative approach or re-enable vtracer
    raise NotImplementedError("Image-to-SVG conversion temporarily disabled due to deployment constraints. Please use text-based SVG generation instead.")

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

@app.route('/static/images/parallel/<path:session_folder>/<path:filename>')
def serve_parallel_image(session_folder, filename):
    """Serve images from the parallel pipeline directory"""
    parallel_path = os.path.join(PARALLEL_OUTPUTS_DIR, session_folder)
    return send_from_directory(parallel_path, filename)

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

def build_advanced_image_prompt(user_input, design_context):
    """Build an advanced image prompt optimized for creating PERFECTLY CENTERED stunning visuals"""
    logger.info(f"Building advanced CENTERED image prompt: {user_input[:100]}...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

    # Use unified centered approach for all designs
    system_content = """You are an expert at creating FINAL image generation prompts that produce PERFECTLY CENTERED visuals. Take the enhanced prompt and design context to create the ultimate prompt for GPT Image-1 that will generate stunning designs with PERFECT CENTER ALIGNMENT.

CRITICAL REQUIREMENT: ALL ELEMENTS MUST BE PERFECTLY CENTERED - Every text, image, shape, and component must be centrally aligned both horizontally and vertically.

Your final prompt must be optimized for:

1. PERFECT CENTER ALIGNMENT
   - ALL elements PERFECTLY CENTERED both horizontally and vertically
   - Symmetrical composition with center focal point
   - CENTERED typography that commands attention
   - Balanced layout with everything aligned to center
   - NO off-center elements whatsoever

2. GPT IMAGE-1 OPTIMIZATION FOR CENTERED DESIGNS
   - Clear specifications for CENTERED element generation
   - Technical specifications for crisp 1024x1024 output with CENTER alignment
   - CENTERED element separation for clean generation
   - Prompt structure that maximizes CENTERED generation quality
   - CENTER positioning instructions for all components

3. CENTERED VISUAL EXCELLENCE
   - Professional CENTERED composition that captures attention
   - CENTERED typography and layout that serves the design purpose
   - Color schemes that enhance CENTERED content
   - Visual hierarchy with CENTERED focal points
   - Symmetrical design elements around the center

4. CENTERED TECHNICAL REQUIREMENTS
   - Specify CENTER ALIGNMENT for all text elements
   - CENTERED positioning for all visual components
   - Symmetrical spacing around CENTERED content
   - Professional aesthetics with PERFECT CENTER ALIGNMENT
   - Balanced composition that works with CENTERED approach

MANDATORY: Every element in the final prompt must emphasize PERFECT CENTER ALIGNMENT. Create a final prompt that will generate visuals where EVERYTHING is perfectly centered."""

    payload = {
        "model": PROMPT_ENHANCER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": f"Enhanced Prompt: {user_input}\n\nDesign Context: {design_context[:500]}..."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        logger.info("Calling OpenAI for advanced image prompt building")
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"OpenAI API error for advanced prompt building: {response_data}")
            # Fallback to simplified prompt
            return f"Create a stunning visual design: {user_input}. Professional quality, 1024x1024 resolution, high contrast, vibrant colors, clear typography, balanced composition."

        final_prompt = response_data["choices"][0]["message"]["content"].strip()
        
        # Ensure prompt isn't too long for GPT Image-1
        if len(final_prompt) > 1000:
            final_prompt = final_prompt[:1000] + "..."
        
        logger.info(f"Successfully built advanced image prompt: {final_prompt[:100]}...")
        return final_prompt

    except Exception as e:
        logger.error(f"Error building advanced image prompt: {str(e)}")
        # Return simplified fallback
        return f"Create a stunning visual design: {user_input}. Professional quality, 1024x1024 resolution, high contrast, vibrant colors, clear typography, balanced composition."

def process_ocr_svg(image_data):
    """Generate a text-only SVG using GPT-4.1-mini by passing the image directly to the chat API."""
    if not PARALLEL_FEATURES_AVAILABLE:
        raise NotImplementedError("Parallel features not available - missing dependencies")
    
    # Base64-encode the PNG image
    img_b64 = base64.b64encode(image_data).decode('utf-8')
    
    # Build prompts matching generate_svg_from_image style
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
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 1,
        "max_tokens": 2000
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
    svg_filename = save_svg(svg_code, prefix='text_svg')
    return svg_code, svg_filename

def process_clean_svg(image_data):
    """Process text removal and convert to clean SVG"""
    if not PARALLEL_FEATURES_AVAILABLE:
        raise NotImplementedError("Parallel features not available - missing dependencies")
    
    # Save the original image bytes to a temporary PNG file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_input_path = f"temp_input_{timestamp}_{uuid.uuid4()}.png"
    with open(temp_input_path, "wb") as f:
        f.write(image_data)

    try:
        # Remove text from the image using remove_text_simple
        edited_png_path = remove_text_simple.remove_text(temp_input_path)

        # Convert the edited PNG to SVG using vtracer with optimized settings
        output_svg_path = os.path.join(IMAGES_DIR, f"clean_{timestamp}_{uuid.uuid4().hex[:8]}.svg")
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

        # Read the generated SVG
        with open(output_svg_path, 'r', encoding='utf-8') as f:
            svg_code = f.read()

        return svg_code, os.path.basename(output_svg_path), edited_png_path
    finally:
        # Clean up temporary file
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

def ai_combine_svgs(text_svg_code, traced_svg_code):
    """AI-powered combination of text and traced SVGs using OpenAI GPT-4.1-nano"""
    logger.info("Using AI to intelligently combine SVGs...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_SVG}"
    }

    system_prompt = """You are an expert SVG designer and combiner. Your task is to intelligently combine two SVG files:
1. A text-focused SVG (contains text elements extracted from an image)
2. A traced/background SVG (contains the visual graphics without text)

Your goal is to create a single, perfectly combined SVG that:
- Maintains proper layering (background graphics behind, text on top)
- Ensures optimal positioning and alignment
- Preserves all visual elements from both SVGs
- Uses appropriate opacity and blending for visual harmony
- Creates a cohesive, professional design
- Maintains proper SVG structure and dimensions (1080x1080)
- Uses semantic grouping with descriptive IDs
- Ensures text readability over background elements

Guidelines:
- Analyze both SVGs carefully before combining
- Preserve the integrity of text elements (fonts, sizes, positions)
- Maintain the visual appeal of background graphics
- Use proper layering with <g> groups
- Apply subtle opacity adjustments if needed for text readability
- Ensure the combined result looks professional and cohesive
- Return ONLY the final combined SVG code, no explanations"""

    user_prompt = f"""Please combine these two SVGs intelligently:

TEXT SVG (contains text elements):
```svg
{text_svg_code}
```

BACKGROUND/TRACED SVG (contains graphics/shapes):
```svg
{traced_svg_code}
```

Create a single, perfectly combined SVG that merges both elements beautifully."""

    payload = {
        "model": "gpt-4.1-nano",
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
        "temperature": 0.3,
        "max_tokens": 3000
    }

    try:
        logger.info("Calling OpenAI for intelligent SVG combination...")
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code != 200:
            logger.error(f"OpenAI API error for SVG combination: {response_data}")
            # Fallback to simple combination
            return simple_combine_svgs_fallback(text_svg_code, traced_svg_code)

        ai_response = response_data["choices"][0]["message"]["content"]
        
        # Extract SVG code from the response
        svg_pattern = r'<svg.*?</svg>'
        svg_match = re.search(svg_pattern, ai_response, re.DOTALL)
        
        if svg_match:
            combined_svg = svg_match.group(0)
            logger.info("AI successfully combined SVGs")
            return combined_svg
        else:
            logger.warning("Could not extract SVG from AI response, using fallback")
            return simple_combine_svgs_fallback(text_svg_code, traced_svg_code)
            
    except Exception as e:
        logger.error(f"Error in AI SVG combination: {str(e)}")
        # Fallback to simple combination
        return simple_combine_svgs_fallback(text_svg_code, traced_svg_code)

def simple_combine_svgs_fallback(text_svg_code, traced_svg_code):
    """Fallback simple combination method"""
    try:
        # Extract content from both SVGs
        text_match = re.search(r'<svg[^>]*>(.*?)</svg>', text_svg_code, re.DOTALL)
        traced_match = re.search(r'<svg[^>]*>(.*?)</svg>', traced_svg_code, re.DOTALL)
        
        if not text_match or not traced_match:
            logger.warning("Could not extract SVG content, returning traced SVG")
            return traced_svg_code
        
        text_content = text_match.group(1).strip()
        traced_content = traced_match.group(1).strip()
        
        # Create combined SVG
        combined_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080" width="1080" height="1080">
  <g id="background-layer" opacity="0.9">
    {traced_content}
  </g>
  <g id="text-layer">
    {text_content}
  </g>
</svg>'''
        
        return combined_svg
    except Exception as e:
        logger.error(f"Error in fallback SVG combination: {str(e)}")
        return traced_svg_code

@app.route('/api/generate-parallel-svg', methods=['POST'])
def generate_parallel_svg():
    """Pipeline: Stages 1-6 image gen, then parallel Stage 7: OCR+SVG and Clean SVG generation"""
    try:
        if not PARALLEL_FEATURES_AVAILABLE:
            return jsonify({
                "error": "Parallel SVG features not available",
                "message": "Missing required dependencies (vtracer, remove_text_simple, etc.)",
                "fallback": "Please use /api/generate-svg endpoint instead"
            }), 501

        data = request.json or {}
        user_input = data.get('prompt', '')
        skip_enhancement = data.get('skip_enhancement', False)

        if not user_input:
            return jsonify({'error': 'No prompt provided'}), 400

        logger.info('=== PARALLEL SVG PIPELINE START ===')

        # Stage 2: Design Planning
        logger.info('Stage 2: Design Planning')
        design_plan = plan_design(user_input)

        # Stage 3: Design Knowledge Generation
        logger.info('Stage 3: Design Knowledge Generation')
        design_knowledge = generate_design_knowledge(design_plan, user_input)

        # Prepare context for enhancements
        design_context = f"""Design Plan:\n{design_plan}\n\nDesign Knowledge and Best Practices:\n{design_knowledge}\n\nOriginal Request:\n{user_input}"""

        # Stage 4: Pre-Enhancement Phase
        logger.info('Stage 4: Pre-Enhancement Phase')
        logger.info('Pre-enhancing prompt with design context...')
        logger.info(f'Using model: {PRE_ENHANCER_MODEL}')
        pre_enhanced_prompt = pre_enhance_prompt(design_context)
        logger.info('Pre-enhanced prompt generated')

        # Stage 5: Final Enhancement Phase
        logger.info('Stage 5: Final Enhancement Phase')
        logger.info('Enhancing prompt with technical specifications...')
        logger.info(f'Using model: {PROMPT_ENHANCER_MODEL}')
        enhanced_prompt = enhance_prompt_with_chat(pre_enhanced_prompt)
        logger.info('Enhanced prompt generated')

        # Build advanced image prompt optimized for parallel SVG processing
        image_prompt = build_advanced_image_prompt(enhanced_prompt, design_context)

        # Stage 6: Image Generation via GPT-Image using enhanced prompt
        logger.info('Stage 6: Image Generation via GPT-Image with enhanced prompt')
        logger.debug(f'Image prompt: {image_prompt[:200]}...')
        image_base64, image_filename = generate_image_with_gpt(image_prompt, design_context)
        image_data = base64.b64decode(image_base64)

        # Stage 7: Parallel Processing
        logger.info('Stage 7: Parallel Processing - OCR+SVG and Clean SVG')
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            ocr_future = executor.submit(process_ocr_svg, image_data)
            clean_future = executor.submit(process_clean_svg, image_data)
            
            # Get results
            text_svg_code, text_svg_path = ocr_future.result()
            clean_svg_code, clean_svg_path, edited_png_path = clean_future.result()

        # Stage 8: AI-Powered SVG Combination
        logger.info('Stage 8: AI-Powered SVG Combination using GPT-4.1-nano')
        combined_svg_code = ai_combine_svgs(text_svg_code, clean_svg_code)
        combined_svg_filename = f"combined_svg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.svg"
        combined_svg_path = os.path.join(IMAGES_DIR, combined_svg_filename)
        with open(combined_svg_path, 'w') as f:
            f.write(combined_svg_code)

        # Create a session subfolder and move outputs there
        session_folder = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        output_folder = os.path.join(PARALLEL_OUTPUTS_DIR, session_folder)
        os.makedirs(output_folder, exist_ok=True)

        # Base URL for parallel outputs
        base_url = '/static/images/parallel'

        # Move files to session folder
        try:
            # Move generated image
            src_image = os.path.join(IMAGES_DIR, image_filename)
            dst_image = os.path.join(output_folder, image_filename)
            if os.path.exists(src_image):
                os.rename(src_image, dst_image)

            # Move text SVG
            src_text_svg = os.path.join(IMAGES_DIR, text_svg_path)
            dst_text_svg = os.path.join(output_folder, text_svg_path)
            if os.path.exists(src_text_svg):
                os.rename(src_text_svg, dst_text_svg)

            # Move cleaned SVG
            src_clean_svg = os.path.join(IMAGES_DIR, clean_svg_path) if not os.path.isabs(clean_svg_path) else clean_svg_path
            dst_clean_svg = os.path.join(output_folder, os.path.basename(clean_svg_path))
            if os.path.exists(src_clean_svg):
                os.rename(src_clean_svg, dst_clean_svg)

            # Move combined SVG
            dst_combined_svg = os.path.join(output_folder, combined_svg_filename)
            if os.path.exists(combined_svg_path):
                os.rename(combined_svg_path, dst_combined_svg)

            # Move cleaned PNG
            dst_edited_png = os.path.join(output_folder, os.path.basename(edited_png_path))
            if os.path.exists(edited_png_path):
                os.rename(edited_png_path, dst_edited_png)

        except Exception as e:
            logger.warning(f"Error moving files to session folder: {e}")

        # Construct URLs for client access
        image_url = f"{base_url}/{session_folder}/{image_filename}"
        text_svg_url = f"{base_url}/{session_folder}/{text_svg_path}"
        clean_svg_url = f"{base_url}/{session_folder}/{os.path.basename(clean_svg_path)}"
        combined_svg_url = f"{base_url}/{session_folder}/{combined_svg_filename}"
        edited_png_url = f"{base_url}/{session_folder}/{os.path.basename(edited_png_path)}"

        return jsonify({
            'original_prompt': user_input,
            'image_url': image_url,
            'edited_png': {
                'path': f"parallel/{session_folder}/{os.path.basename(edited_png_path)}",
                'url': edited_png_url
            },
            'text_svg': {
                'code': text_svg_code,
                'path': f"parallel/{session_folder}/{text_svg_path}"
            },
            'clean_svg': {
                'code': clean_svg_code,
                'path': f"parallel/{session_folder}/{os.path.basename(clean_svg_path)}"
            },
            'combined_svg': {
                'code': combined_svg_code,
                'path': f"parallel/{session_folder}/{combined_svg_filename}",
                'url': combined_svg_url
            },
            'stage': 8
        })

    except Exception as e:
        logger.error(f"Error in generate_parallel_svg: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets PORT=8000)
    port = int(os.getenv('PORT', 5000))
    
    # Use 0.0.0.0 for production (Render) and 127.0.0.1 for local development
    host = '0.0.0.0' if os.getenv('PORT') else '127.0.0.1'
    
    # Disable debug mode in production
    debug = not bool(os.getenv('PORT'))
    
    logger.info(f"Starting Flask application on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
