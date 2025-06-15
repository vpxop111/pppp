import requests
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
OPENAI_API_KEY_ENHANCER = os.getenv('OPENAI_API_KEY_ENHANCER')
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
    """Plan the design approach based on user input"""
    logger.info(f"Planning design for: {user_input[:100]}...")
    
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
                "content": """You are a design planner specializing in high-quality posters, logos, and branding materials. Create a structured plan for the user's request that focuses on visual impact, cohesive color themes, and professional composition.

Your plan should include:
1. Design Goals
   - Main purpose and intended use (e.g., poster, logo)
   - Target audience and mood
   - Key message and visual emotion

2. Visual Composition
   - Overall layout structure and focal points
   - Background elements (colors, gradients, textures)
   - Visual hierarchy and balance

3. Typography and Style
   - Font choices, sizes, and pairings
   - Spacing, alignment, and hierarchy

4. Color and Branding
   - Suggested color palette with hex/RGB codes
   - Background and accent color recommendations
   - Mood or theme descriptors (e.g., vibrant, minimal, elegant)

5. Technical and Production Considerations
   - SVG optimization requirements and file size targets
   - Responsive design and browser compatibility
   - Logo integration and branding guidelines

6. Implementation Strategy
   - Component breakdown and creation order
   - Special effects or patterns (shadows, overlays)
   - Testing and validation steps

Be specific, practical, and focus on creating a visually compelling poster or logo that aligns with the user's request."""
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"Design planning error: {response_data}")
        return "Error in design planning"

    return response_data["choices"][0]["message"]["content"]

def generate_design_knowledge(design_plan, user_input):
    """Generate specific design knowledge based on the plan and user input"""
    logger.info("Generating design knowledge...")
    
    url = OPENAI_CHAT_ENDPOINT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_ENHANCER}"
    }

    payload = {
        "model": DESIGN_KNOWLEDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a design knowledge expert specializing in professional poster and logo creation. Based on the design plan and user request, provide detailed insights, best practices, and actionable recommendations to enhance visual quality and impact.

Include:
1. Typography and Branding
   - Font recommendations with pairing suggestions
   - Hierarchy and spacing for emphasis
   - Logo integration tips

2. Color and Mood
   - Cohesive color palette suggestions with hex/RGB values
   - Contrast, accessibility, and readability guidance
   - Background color and texture ideas for mood

3. Layout and Composition
   - Effective grid systems and alignment techniques
   - Focal point emphasis and visual flow strategies
   - Balanced use of whitespace and dynamic elements

4. SVG and Graphic Techniques
   - Path and shape optimization for crisp visuals
   - Use of overlays, shadows, gradients, and textures
   - Grouping and layer organization for clarity

5. Technical and Production
   - Responsive design tips for multiple screen sizes
   - SVG file size optimization and performance
   - Cross-browser compatibility considerations

Provide clear, practical recommendations to elevate the design quality for posters, logos, and branding materials."""
            },
            {
                "role": "user",
                "content": f"Design Plan:\n{design_plan}\n\nUser Request:\n{user_input}"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"Design knowledge generation error: {response_data}")
        return "Error in generating design knowledge"

    return response_data["choices"][0]["message"]["content"]

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
                "content": """You are an expert design prompt enhancer. Enhance the user's design request into a highly detailed, technical specification optimized for creating professional-quality posters, logos, and branding materials.

Your enhanced prompt should include:
1. Layout and Composition
   - Overall structure and visual flow
   - Focal points and visual hierarchy
   - Balance, symmetry, and use of whitespace

2. Typography
   - Font styles and families with pairing recommendations
   - Font sizes with emphasis hierarchy
   - Text alignment and letter spacing

3. Color and Atmosphere
   - Cohesive color palette suggestions with hex/RGB codes
   - Background colors, gradients, and texture ideas
   - Mood and style descriptors (e.g., vibrant, elegant, minimalist)

4. Visual Elements and Branding
   - Background graphics, patterns, or images
   - Icons, symbols, and decorative shapes
   - Logo placement and integration tips

5. Technical Requirements
   - SVG-specific considerations and optimization techniques
   - Responsive design and scaling guidelines
   - Browser compatibility and performance optimizations

6. Poster and Logo Aesthetics
   - Recommendations for cohesion between elements
   - Suggestions for lighting, shadow, and depth
   - Tips for making the design stand out visually

Convert the user's brief into a complete, actionable prompt that maintains their original intent while adding all necessary details to produce a visually compelling and professional SVG design."""
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1500
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
                "content": """You are an advanced SVG design prompt optimizer specializing in high-impact posters and logo creation. Enhance the pre-enhanced design prompt to ensure the resulting SVG is visually stunning, cohesive, and technically precise.

Your optimization should address:
1. Vector and Visual Optimization
   - Emphasize vector-friendly shapes, paths, and curves
   - Define gradients, patterns, and textures for backgrounds
   - Recommend shadows, overlays, and depth for visual appeal

2. Technical Precision
   - Exact dimensions, positions, and aspect ratios
   - Precise color values (HEX/RGB) with contrast considerations
   - Font specifications, including web-safe fallbacks and loading strategies

3. Component and Layer Organization
   - Grouping of elements for reuse and clarity
   - IDs and classes for styling and interactivity
   - Organized layering to control visual stacking and effects

4. Performance and Accessibility
   - Optimize paths to minimize file size and complexity
   - Ensure responsive scaling and cross-browser performance
   - Add ARIA labels and semantic structure for accessibility

5. Poster and Branding Details
   - Specify background elements (colors, gradients, patterns)
   - Suggest lighting, shadow, and texture to create depth
   - Recommend cohesive color themes and mood descriptors
   - Integrate logo placement and branding cues

Ensure the final prompt guides the SVG generator to produce a complete, eye-catching poster or logo design with a well-defined background, color theme, and professional quality."""
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    logger.info(f"Calling OpenAI Chat API for prompt enhancement with model: {PROMPT_ENHANCER_MODEL}")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()

    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response_data}")
        raise Exception(f"OpenAI API error: {response_data.get('error', {}).get('message', 'Unknown error')}")

    return response_data["choices"][0]["message"]["content"] 