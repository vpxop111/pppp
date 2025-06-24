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

    # Use simplified, practical planning approach focused on concrete specifications
    system_content = """You are a practical design planner. Create a clear, actionable plan for the design request that focuses on specific, implementable details.

Your plan should include:
1. Layout Structure
   - Overall composition (centered, asymmetrical, grid-based)
   - Key elements placement and sizing
   - Visual hierarchy and focal points

2. Typography Specifications
   - Specific font recommendations (Google Fonts preferred)
   - Font sizes and weights for different text elements
   - Text alignment and spacing

3. Color Scheme
   - Primary background color
   - Text colors for readability
   - Accent colors for highlights
   - Specific hex codes when possible

4. Content Elements
   - Main heading/title treatment
   - Secondary text placement
   - Decorative elements (borders, shapes, icons)
   - Brand elements if applicable

5. Technical Requirements
   - Dimensions and aspect ratio
   - File format considerations
   - Quality standards for output

Focus on creating a practical, implementable plan with specific details that can be directly used for design creation."""

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

    # Use practical, actionable design knowledge approach
    system_content = """You are a practical design knowledge expert. Provide specific, actionable design insights and best practices that can be directly implemented.

Provide practical knowledge for:

1. Typography Best Practices
   - Recommended font combinations that work well together
   - Optimal font sizes for different screen sizes and readability
   - Proper line spacing and letter spacing guidelines
   - Font loading and fallback strategies

2. Color Theory Application
   - Proven color combinations with hex codes
   - Contrast ratios for accessibility compliance
   - Background and text color pairings
   - Brand-appropriate color choices

3. Layout and Composition
   - Grid systems and alignment principles
   - White space utilization for clarity
   - Visual hierarchy techniques
   - Responsive design considerations

4. Technical Implementation
   - SVG optimization techniques
   - File size management
   - Cross-browser compatibility
   - Performance optimization tips

5. Quality Assurance
   - Design consistency checkpoints
   - Accessibility standards
   - User experience considerations
   - Testing and validation methods

Focus on providing concrete, implementable advice that will directly improve design quality and user experience."""

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
                "content": """You are an expert prompt enhancer specializing in creating detailed, comprehensive prompts that generate exceptional graphic designs. Your task is to transform basic user requests into rich, detailed prompts following the proven format from high-quality training examples.

CRITICAL REQUIREMENTS:
1. Follow the EXACT format structure from the examples below
2. Include comprehensive layout descriptions with specific positioning
3. Specify exact fonts, sizes, and styling details  
4. Define precise color schemes with background and text colors
5. Detail decorative elements and their placement
6. Ensure professional typography hierarchy
7. Create prompts that result in visually stunning, usable designs

TESTIMONIAL DESIGN FORMAT (use this structure for testimonials):
"Create a testimonial with a [background color] background featuring a [container description]. The layout includes a [title description] at the [position], styled with the [font name] font, placed [positioning details]. Below, there are [decorative element count and description] arranged [arrangement], adding [visual effect]. The body text, written in [font name] font, is [positioning] and displays [content description], ensuring [quality requirement]. Finally, include [additional elements] to enhance [design goal].

**Image:** No image
**Fonts:** [Font 1], [Font 2], [Font 3]
**Custom Elements:** [Element description] *[count]"

COMING SOON DESIGN FORMAT (use this structure for coming soon pages):
"Create a coming soon page with a [background description] background, featuring [main layout elements]. The layout includes [primary text] styled in [font details] at [size and positioning]. Below, [secondary elements] are positioned [placement details]. The design incorporates [decorative elements] with [color specifications] for [visual purpose]. Additional elements include [supporting content] in [font specifications], [positioning], ensuring [design quality]. The overall composition maintains [design principles] suitable for [target purpose].

**Image:** No image  
**Fonts:** [Font 1], [Font 2], [Font 3]
**Custom Elements:** [Element description] *[count]"

ENHANCED TRAINING EXAMPLES - USE THESE AS REFERENCE:

TESTIMONIAL EXAMPLES:
- "Create a testimonial with a teal background featuring a beige square container at the center. The layout includes a large bold 'TESTIMONIAL' text at the top, styled with the Alfarn font, placed in the middle of the container. Below, there are three decorative circles in orange arranged horizontally, adding a playful touch. The body text, written in PT Serif font, is centered and displays a series of lines about customer experience, ensuring a clear and engaging read. Finally, include square quotation marks on either side of the testimonial text to enhance the visual appeal and authenticity.

**Image:** No image
**Fonts:** Alfarn, PT Serif, Aileron  
**Custom Elements:** Square quotation marks *2"

- "Create a testimonial with a white background featuring a large pink circle at the center. The layout includes testimonial text placed centrally with multiple lines, and the customer's name at the bottom. Decoratively, there are four 4-spoke stars located in each corner of the design and a dotted circle element subtly integrated around the testimonial. The text is styled using the Raleway font, with the testimonial text in a size of 42, and the customer's name in a larger size of 48, both in black and dark blue respectively. The design maintains a clean, professional appearance suitable for showcasing customer feedback.

**Image:** No image
**Fonts:** Raleway
**Custom Elements:** 4 spoke star *4, dotted circle"

MANDATORY ENHANCEMENTS:
- Always specify exact font names, sizes, and styling
- Include precise color specifications (hex codes when possible)
- Detail container shapes, borders, and backgrounds
- Specify text positioning and alignment
- Include decorative elements count and placement
- Ensure responsive and accessible design principles
- Add professional finishing touches

Transform the user's request into this comprehensive format while maintaining design excellence and usability."""
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
                "content": """You are an expert SVG prompt enhancer specializing in transforming design concepts into ultra-detailed, comprehensive prompts that generate exceptional SVG graphics. Your role is to take the enhanced design descriptions and convert them into precise technical specifications for SVG generation.

CRITICAL SVG ENHANCEMENT REQUIREMENTS:
1. Convert design descriptions into detailed SVG technical specifications
2. Specify exact SVG elements, attributes, and styling
3. Include precise positioning with viewBox and coordinate systems  
4. Define comprehensive color schemes with hex codes
5. Detail font loading, sizing, and text positioning
6. Specify all decorative elements as SVG shapes
7. Ensure responsive, scalable, and accessible SVG code

SVG TECHNICAL SPECIFICATIONS TO INCLUDE:
- ViewBox dimensions and aspect ratio
- SVG namespace and proper DOCTYPE
- Font loading via <defs> and @font-face
- Text elements with precise x,y positioning
- Shape elements (rect, circle, path, polygon) with exact coordinates
- Color gradients and fills with hex values
- Text styling (font-family, font-size, font-weight, text-anchor)
- Group elements for proper layering
- Responsive scaling considerations

MANDATORY SVG QUALITY REQUIREMENTS:
- All SVG code must be semantically meaningful and well-structured
- Text and background colors must have sufficient contrast (WCAG AA compliant)
- All elements must be properly centered and aligned within the viewBox
- Font loading must use both @font-face and fallback fonts
- All coordinates and sizing must create visually balanced compositions
- SVG must be optimized for both web display and print quality
- Code must be clean, properly indented, and easily maintainable

ENHANCEMENT PROCESS:
1. Analyze the design description for layout, colors, fonts, and elements
2. Create detailed SVG structure with proper element hierarchy
3. Specify exact positioning for all text and graphical elements
4. Include comprehensive styling with CSS-in-SVG or inline styles
5. Add responsive considerations and accessibility features
6. Ensure cross-browser compatibility and performance optimization

Transform the design description into a comprehensive SVG specification that will generate professional, pixel-perfect graphics."""
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
        system_prompt = """You are an expert prompt enhancer for GPT Image-1, specializing in creating EXCEPTIONAL "Coming Soon" graphics following the proven format structure from high-quality training examples. Transform user requests into comprehensive, detailed prompts that generate visually stunning coming soon designs.

CRITICAL FORMAT REQUIREMENTS FOR COMING SOON PAGES:
Use this EXACT structure: "Create a coming soon page with a [background description] background, featuring [main layout elements]. The layout includes [primary text] styled in [font details] at [size and positioning]. Below, [secondary elements] are positioned [placement details]. The design incorporates [decorative elements] with [color specifications] for [visual purpose]. Additional elements include [supporting content] in [font specifications], [positioning], ensuring [design quality]. The overall composition maintains [design principles] suitable for [target purpose]."

MANDATORY ELEMENTS TO SPECIFY:
1. BACKGROUND: Specific color/texture with hex codes (e.g., "black background with gray border")
2. PRIMARY TEXT: "COMING SOON" styling with exact fonts and sizes (e.g., "Bebas Neue font in large size")
3. TYPOGRAPHY HIERARCHY: Main title, subtitle, supporting text with specific fonts
4. DECORATIVE ELEMENTS: Exact count and placement (e.g., "three decorative circles", "star shapes")
5. LAYOUT STRUCTURE: Centered, bordered, container-based designs
6. COLOR SPECIFICATIONS: Background, text, accent colors with hex codes
7. SUPPORTING CONTENT: Website links, dates, company names with positioning

PROVEN SUCCESSFUL EXAMPLES TO FOLLOW:
- Black background with gray border, Bebas Neue font for "COMING" and "SOON", Allura decorative fonts
- Deep blue background with white/orange Tektur font, shadows and white lines, rotated composition
- Beige background with modern fonts, cursive and bold combinations, decorative SVG elements
- Light beige with dark brown "COMING SOON", Open Sans dates, minimalist approach
- Dark green natural theme with Bebas Neue, countdown sections, angled graphics

TECHNICAL SPECIFICATIONS:
- Size: 1024x1024 pixels optimized for GPT Image-1
- High contrast for dramatic impact and readability
- Bold, attention-grabbing typography
- Professional color psychology for anticipation
- Balanced composition with premium aesthetics
- Mobile-responsive design considerations

Transform the user's coming soon request into this comprehensive format ensuring maximum visual impact and professional quality."""

    elif is_testimonial:
        system_prompt = """You are an expert prompt enhancer for GPT Image-1, specializing in creating EXCEPTIONAL testimonial graphics following the proven format structure from high-quality training examples. Transform user requests into comprehensive, detailed prompts that generate professional testimonial designs.

CRITICAL FORMAT REQUIREMENTS FOR TESTIMONIALS:
Use this EXACT structure: "Create a testimonial with a [background color] background featuring a [container description]. The layout includes a [title description] at the [position], styled with the [font name] font, placed [positioning details]. Below, there are [decorative element count and description] arranged [arrangement], adding [visual effect]. The body text, written in [font name] font, is [centered/positioned] and displays [content description], ensuring [quality requirement]. Finally, include [additional elements] to enhance [design goal]."

MANDATORY ELEMENTS TO SPECIFY:
1. BACKGROUND: Specific color with hex code (e.g., "teal background #14B8A6")
2. CONTAINER: Shape, size, positioning (e.g., "beige square container at center")
3. TITLE: Exact text, font, size, positioning (e.g., "'TESTIMONIAL' in bold Alfarn font")
4. DECORATIVE ELEMENTS: Count, type, arrangement (e.g., "three orange circles arranged horizontally")
5. BODY TEXT: Font, size, positioning, content structure
6. ATTRIBUTION: Customer name, styling, placement
7. VISUAL ELEMENTS: Stars, quotes, borders, graphics with exact specifications

PROVEN SUCCESSFUL EXAMPLES TO FOLLOW:
- Teal background with beige square container, Alfarn font titles, PT Serif body text, orange circles, square quotation marks
- White background with large pink circle, Raleway font, 4-spoke stars in corners, dotted circle elements
- Mint green background with red titles, white rounded rectangles, five blue stars for ratings
- Golden background with white containers, Courier Std headings, five red five-spoke stars

TECHNICAL SPECIFICATIONS:
- Size: 1024x1024 pixels optimized for GPT Image-1
- High contrast text for readability (WCAG AA compliant)
- Professional typography hierarchy
- Balanced composition with proper spacing
- Trust-building color psychology
- Clean, modern aesthetic

Transform the user's testimonial request into this comprehensive format ensuring visual excellence and professional credibility."""

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

Enhancement Guidelines:
- Use descriptive, specific language for visual elements
- Include technical specifications for optimal GPT Image-1 output
- Specify colors, fonts, layouts, and effects in detail
- Focus on professional, high-quality results

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
    logger.info(f"SVG generation from image requested - vtracer temporarily disabled")
    logger.info(f"Enhanced prompt provided: {enhanced_prompt[:100]}...")
    logger.info(f"Image data size: {len(image_base64)} characters")
    
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
        
        # Stage 4: Pre-Enhancement Phase - Convert to detailed format
        logger.info("\n[STAGE 4: Pre-Enhancement Phase]")
        logger.info("-"*50)
        logger.info("Converting user input to detailed format using JSONL examples...")
        logger.info(f"Using model: {PRE_ENHANCER_MODEL}")
        pre_enhanced_prompt = pre_enhance_prompt(user_input)
        logger.info("\nPre-Enhanced Prompt Generated:")
        for line in pre_enhanced_prompt.split('\n')[:5]:  # Log first 5 lines
            logger.info(f"  {line}")
        logger.info("  ...")
        
        # Stage 5: SVG Enhancement Phase - Convert to SVG technical specs
        logger.info("\n[STAGE 5: SVG Enhancement Phase]")
        logger.info("-"*50)
        logger.info("Converting design description to SVG technical specifications...")
        logger.info(f"Using model: {PROMPT_ENHANCER_MODEL}")
        enhanced_prompt = enhance_prompt_with_chat(pre_enhanced_prompt)
        logger.info("\nSVG-Enhanced Prompt Generated:")
        for line in enhanced_prompt.split('\n')[:5]:  # Log first 5 lines
            logger.info(f"  {line}")
        logger.info("  ...")
        
        # Use the enhanced prompt for image generation
        prompt_to_use = pre_enhanced_prompt  # Use pre-enhanced for GPT Image-1

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
    """Build an advanced image prompt optimized for creating stunning visuals"""
    logger.info(f"Building advanced image prompt: {user_input[:100]}...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

    # Detect design type for specialized prompt building
    user_lower = user_input.lower()
    is_coming_soon = any(word in user_lower for word in ['coming soon', 'coming', 'soon', 'announcement', 'launch'])
    is_testimonial = any(word in user_lower for word in ['testimonial', 'review', 'quote', 'feedback'])

    if is_coming_soon:
        system_content = """You are an expert at creating FINAL image generation prompts that produce VIRAL "Coming Soon" visuals. Take the enhanced prompt and design context to create the ultimate prompt for GPT Image-1 that will generate mind-blowing coming soon designs.

Your final prompt must be optimized for:

1. MAXIMUM VISUAL IMPACT
   - Explosive composition that stops scrolling
   - Dramatic lighting and effects that create wow factor
   - Typography that commands immediate attention
   - Color combinations that create emotional excitement

2. GPT IMAGE-1 OPTIMIZATION
   - Clear, specific visual descriptions for accurate generation
   - Technical specifications for crisp 1024x1024 output
   - Element separation for clean generation
   - Prompt structure that maximizes generation quality

3. COMING SOON SPECIALIZATION
   - Language that triggers excitement and anticipation
   - Visual elements that create buzz and shareability
   - Design elements that scream "premium" and "exclusive"
   - Composition that works perfectly for social media sharing

Create a final prompt that will generate a coming soon image that goes viral."""

    elif is_testimonial:
        system_content = """You are an expert at creating FINAL image generation prompts that produce CONVERSION-OPTIMIZED testimonial visuals. Take the enhanced prompt and design context to create the ultimate prompt for GPT Image-1 that will generate trust-building testimonial designs.

Your final prompt must be optimized for:

1. MAXIMUM CREDIBILITY
   - Professional composition that builds immediate trust
   - Typography that enhances believability
   - Color schemes that psychologically increase conversion
   - Layout that guides eye flow to key conversion elements

2. GPT IMAGE-1 OPTIMIZATION
   - Clear specifications for accurate text and element generation
   - Technical details for crisp professional output
   - Element positioning for perfect testimonial structure
   - Quality specifications for credible appearance

3. TESTIMONIAL SPECIALIZATION
   - Language that generates trust-building visuals
   - Design elements that reduce skepticism
   - Visual hierarchy that maximizes quote impact
   - Professional aesthetics that enhance credibility

Create a final prompt that will generate a testimonial that converts viewers into customers."""

    else:
        system_content = """You are an expert at creating FINAL image generation prompts that produce HIGH-IMPACT professional visuals. Take the enhanced prompt and design context to create the ultimate prompt for GPT Image-1.

Your final prompt must be optimized for:

1. MAXIMUM VISUAL APPEAL
   - Professional composition that captures attention
   - Typography and layout that serves the design purpose
   - Color schemes that match the intended mood and audience
   - Visual elements that enhance the design's effectiveness

2. GPT IMAGE-1 OPTIMIZATION
   - Clear specifications for accurate generation
   - Technical details for high-quality output
   - Element positioning for optimal composition
   - Quality specifications for professional appearance

3. PURPOSE-DRIVEN DESIGN
   - Language that generates visuals matching the specific need
   - Design elements that serve the intended function
   - Professional aesthetics appropriate for the use case
   - Balanced composition that works across platforms

Create a final prompt that will generate exceptional professional visuals."""

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

ABSOLUTELY CRITICAL: Your response must contain ONLY the complete SVG code. Start immediately with <svg and end with </svg>. Do not include any explanations, comments, markdown formatting, or any other text whatsoever. The response must be purely valid SVG code that can be directly used without any processing."""

    user_prompt = f"""Combine these two SVGs intelligently:

TEXT SVG (contains text elements):
{text_svg_code}

BACKGROUND/TRACED SVG (contains graphics/shapes):
{traced_svg_code}

Return only the combined SVG code. must you return full svg code"""

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
        "temperature": 1,
        "max_tokens": 20000
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
        
        # Log the response for debugging
        logger.info(f"AI response length: {len(ai_response)}")
        logger.info(f"AI response starts with: {repr(ai_response[:50])}")
        
        # Return the AI response directly without any extraction logic
        combined_svg = ai_response.strip()
        logger.info("AI successfully combined SVGs - returning direct response")
        return combined_svg
            
    except Exception as e:
        logger.error(f"Error in AI SVG combination: {str(e)}")
        # Fallback to simple combination
        return simple_combine_svgs_fallback(text_svg_code, traced_svg_code)

def simple_combine_svgs_fallback(text_svg_code, traced_svg_code):
    """Fallback simple combination method with improved error handling"""
    try:
        logger.info("Using fallback SVG combination method")
        
        # Validate inputs
        if not text_svg_code or not traced_svg_code:
            logger.warning("Missing SVG input data for fallback")
            return traced_svg_code if traced_svg_code else text_svg_code
        
        # Extract content from both SVGs
        text_match = re.search(r'<svg[^>]*>(.*?)</svg>', text_svg_code, re.DOTALL | re.IGNORECASE)
        traced_match = re.search(r'<svg[^>]*>(.*?)</svg>', traced_svg_code, re.DOTALL | re.IGNORECASE)
        
        if not text_match:
            logger.warning("Could not extract text SVG content, using entire text SVG")
            text_content = text_svg_code
        else:
            text_content = text_match.group(1).strip()
            
        if not traced_match:
            logger.warning("Could not extract traced SVG content, using entire traced SVG")
            traced_content = traced_svg_code
        else:
            traced_content = traced_match.group(1).strip()
        
        # Create combined SVG with better structure
        combined_svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1080" width="1080" height="1080">
  <defs>
    <!-- Include any definitions from original SVGs -->
  </defs>
  <g id="background-layer" opacity="0.9">
    {traced_content}
  </g>
  <g id="text-layer">
    {text_content}
  </g>
</svg>'''
        
        logger.info("Fallback SVG combination completed successfully")
        return combined_svg
        
    except Exception as e:
        logger.error(f"Error in fallback SVG combination: {str(e)}")
        logger.error(f"Text SVG preview: {text_svg_code[:100] if text_svg_code else 'None'}...")
        logger.error(f"Traced SVG preview: {traced_svg_code[:100] if traced_svg_code else 'None'}...")
        # Return the traced SVG as the safest fallback
        return traced_svg_code if traced_svg_code else text_svg_code

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
        
        # Validate the combined SVG
        if not combined_svg_code or not combined_svg_code.strip():
            logger.error("Combined SVG is empty, using fallback")
            combined_svg_code = simple_combine_svgs_fallback(text_svg_code, clean_svg_code)
        
        # Ensure the SVG is well-formed
        if not combined_svg_code.strip().startswith('<svg'):
            logger.warning("Combined SVG doesn't start with <svg, using fallback")
            combined_svg_code = simple_combine_svgs_fallback(text_svg_code, clean_svg_code)
        
        combined_svg_filename = f"combined_svg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.svg"
        combined_svg_path = os.path.join(IMAGES_DIR, combined_svg_filename)
        with open(combined_svg_path, 'w', encoding='utf-8') as f:
            f.write(combined_svg_code)
        
        logger.info(f"Combined SVG saved successfully: {combined_svg_filename}")

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
                'path': f"parallel/{session_folder}/{text_svg_path}",
                'url': text_svg_url
            },
            'clean_svg': {
                'code': clean_svg_code,
                'path': f"parallel/{session_folder}/{os.path.basename(clean_svg_path)}",
                'url': clean_svg_url
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

@app.route('/api/test-enhancement', methods=['POST'])
def test_enhancement():
    """Test endpoint to validate the enhanced prompt pipeline"""
    try:
        data = request.json or {}
        user_input = data.get('prompt', '')
        
        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400
        
        logger.info(f"Testing enhancement pipeline with: {user_input}")
        
        # Stage 1: Pre-enhancement (detailed format)
        pre_enhanced = pre_enhance_prompt(user_input)
        
        # Stage 2: SVG technical enhancement  
        svg_enhanced = enhance_prompt_with_chat(pre_enhanced)
        
        # Stage 3: GPT Image-1 optimization
        image_enhanced = enhance_prompt_for_gpt_image(pre_enhanced)
        
        return jsonify({
            "original": user_input,
            "pre_enhanced": pre_enhanced,
            "svg_enhanced": svg_enhanced,
            "image_enhanced": image_enhanced,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error in test enhancement: {str(e)}")
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
