from flask import Flask, request, jsonify
import os
import requests
import json
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # Load from environment variables

# OpenAI API Endpoints
OPENAI_API_BASE = "https://api.openai.com/v1"
OPENAI_CHAT_ENDPOINT = f"{OPENAI_API_BASE}/chat/completions"

# Gemini API Endpoints
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODEL = "gemini-2.5-pro-experimental-03-25"
GEMINI_GENERATE_ENDPOINT = f"{GEMINI_API_BASE}/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

@app.route('/api/generate-svg', methods=['POST'])
def generate_svg():
    try:
        data = request.json
        user_input = data.get('prompt', '')
        
        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400
        
        logger.info(f"Processing prompt: {user_input[:50]}...")
        
        # Step 1: Enhance the prompt using first OpenAI model
        prompt_enhancer_model = "ft:gpt-4o-mini-2024-07-18:wrecked-labs::BEEibbWw"
        enhanced_prompt = enhance_prompt(user_input, prompt_enhancer_model)
        logger.info(f"Enhanced prompt: {enhanced_prompt[:50]}...")
        
        # Step 2: Generate SVG using second OpenAI model
        svg_generator_model = "ft:gpt-4o-mini-2024-07-18:wrecked-labs::BD4sS07O"
        svg_code = generate_svg_from_prompt(enhanced_prompt, svg_generator_model)
        logger.info("SVG code generated successfully")
        
        # Step 3: Validate and improve SVG using Gemini
        validated_svg = validate_svg_with_gemini(svg_code)
        logger.info("SVG validated and improved with Gemini")
        
        return jsonify({
            "original_prompt": user_input,
            "enhanced_prompt": enhanced_prompt,
            "svg_code": validated_svg
        })
    
    except Exception as e:
        logger.error(f"Error in generate_svg: {str(e)}")
        return jsonify({"error": str(e)}), 500

def enhance_prompt(user_input, model_name):
    """Enhance user prompt using OpenAI model via direct API call"""
    system_prompt = """You are a prompt enhancer assistant. You transform simple, brief prompts into detailed, comprehensive prompts that provide specific details, requirements, and context to help generate better results."""
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    logger.info(f"Calling OpenAI API for prompt enhancement with model: {model_name}")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()
    
    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response_data}")
        raise Exception(f"OpenAI API error: {response_data.get('error', {}).get('message', 'Unknown error')}")
    
    return response_data["choices"][0]["message"]["content"]

def generate_svg_from_prompt(enhanced_prompt, model_name):
    """Generate SVG code using OpenAI model via direct API call"""
    system_prompt = """Generate SVG code for a visual element (testimonial or "coming soon" notice) based on user input and the following design guidelines. Incorporate dynamic CSS styles for fonts, including advanced properties such as font-optical-sizing, font-variation-settings, and unique class names.
Design Guidelines:
Consistency in Design:
Maintain a uniform style across all elements.
Use consistent fonts, colors, and layouts for brand recognition.
Readability:
Ensure text has high contrast against the background.
Avoid excessive decorative elements that clutter the design.
Alignment and Spacing:
Align text and graphical elements consistently.
Maintain appropriate margins to prevent text from being too close to edges.
Visual Hierarchy:
Highlight key information (e.g., customer name, testimonial message).
Use font sizes and weights to establish hierarchy.
Graphic Elements:
Use decorative elements sparingly to complement the text.
Ensure visuals enhance readability rather than distract from it.
Branding:
Incorporate consistent branding elements (e.g., logos, brand colors).
Ensure URLs or company names are clear and legible.
Typography:
Use fonts specified by the user in the prompt.
Dynamically generate CSS classes for each font with unique names (e.g., .tektur-<uniquifier>).
Include advanced font properties like font-optical-sizing and font-variation-settings.
Coherence and Relevance:
Align the message with the brand's tone and identity.
Tailor visuals appropriately based on the purpose (testimonial vs. notice).
Accessibility:
Ensure high contrast for better readability.
Format text to be compatible with screen readers.
Font Integration Guidelines:
Font Preloading:
Use <link> tags to preload Google Fonts. Include preconnect links for better performance:
xml
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="[FONT_URL]" rel="stylesheet">
CSS Definitions:
Inside <style> tags within <defs>, define unique CSS classes for each font-family. Include properties like:
css
.font-class {
  font-family: "FontName", sans-serif;
  font-optical-sizing: auto;
  font-weight: 600;
  font-style: normal;
  font-variation-settings: "wdth" 100;
}
Apply CSS Classes:
Assign these classes to corresponding <text> elements in the SVG.
User Input Format:
Specify the type of visual element (testimonial or coming soon).
Provide the font name(s) to use.
Include specific branding requirements (e.g., colors, logos).
Example User Input:
Generate a testimonial visual using the fonts "Tektur" and "Roboto."
Example Output:
text
<svg width="1080" height="1080" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <!-- Background -->
    <rect width="100%" height="100%" fill="white"/>
    <!-- Font Preload -->
    <defs>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Tektur:wght@600&family=Roboto:wght@400;700&display=swap" rel="stylesheet">
        <!-- CSS Definitions -->
        <style>
            .tektur-font {
                font-family: "Tektur", sans-serif;
                font-optical-sizing: auto;
                font-weight: 600;
                font-style: normal;
                font-variation-settings: "wdth" 100;
            }
            .roboto-font {
                font-family: "Roboto", sans-serif;
                font-weight: 400;
                font-style: normal;
            }
        </style>
    </defs>
    <!-- Testimonial -->
    <text x="50%" y="20%" class="tektur-font" font-size="24" fill="black" text-anchor="middle">
        Customer Name
    </text>
    <text x="50%" y="40%" class="roboto-font" font-size="18" fill="black" text-anchor="middle">
        "This is an example of a testimonial message."
    </text>
</svg>
Dynamic Font Handling Steps in Fine-Tuning Model:
Extract Font Names: Parse user input to identify requested fonts.
Generate Google Fonts URL: Construct valid Google Fonts API URLs based on user input, including weights and styles.
Preload Fonts: Add <link> tags for preloading fonts via Google Fonts API inside <defs>.
Define Unique Classes: Dynamically create CSS classes with unique names (e.g., .tektur-<uniquifier>) for each requested font, including advanced properties like font-optical-sizing and font-variation-settings.
Apply Classes: Assign these classes to corresponding <text> elements in the SVG.

IMPORTANT: Always include the SVG namespace attributes in your output:
- xmlns="http://www.w3.org/2000/svg"
- xmlns:xlink="http://www.w3.org/1999/xlink"

These are required for proper rendering in browsers."""
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 1000
    }
    
    logger.info(f"Calling OpenAI API for SVG generation with model: {model_name}")
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()
    
    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response_data}")
        raise Exception(f"OpenAI API error: {response_data.get('error', {}).get('message', 'Unknown error')}")
    
    return response_data["choices"][0]["message"]["content"]

def validate_svg_with_gemini(svg_code):
    """Validate and improve SVG code using Gemini model"""
    # First, ensure SVG has proper namespaces before sending to Gemini
    if "<svg" in svg_code and "xmlns=" not in svg_code:
        # Add required XML namespaces to SVG if missing
        svg_code = svg_code.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"')
    
    # Fix Gemini model name - use a standard model instead of experimental
    gemini_model = "gemini-2.5-pro-exp-03-25"
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={GEMINI_API_KEY}"
    
    system_prompt = """# System Prompt – SVG Testimonial Design Agent

## Role & Objective
You are a specialized **SVG Testimonial Design Agent** tasked with analyzing, editing, and optimizing SVG testimonial designs. Your core mission is to:
- Comprehensively examine both the rendered SVG image and its raw code
- Identify visual, structural, and design issues
- Provide precise, actionable solutions
- Ensure the final testimonial design is visually appealing and professionally crafted

## Key Capabilities & Workflow

### 1. Comprehensive SVG Analysis

#### Visual Inspection Criteria
- **Text Alignment:**
  - Detect misaligned text within containers
  - Identify text overflow or cut-off problems
  - Verify proper positioning of client names and designations

- **Container & Shape Integrity:**
  - Assess container shape accuracy (rectangle, ellipse, etc.)
  - Check for distorted or overlapping elements
  - Identify unnecessary or interfering shapes

- **Font & Readability:**
  - Evaluate text size proportionality
  - Check color contrast and visibility
  - Assess spacing and line formatting
  - Ensure text remains within design boundaries

- **Layout & Positioning:**
  - Detect misaligned avatars, logos, or icons
  - Identify uneven element spacing
  - Recognize improper layering of design elements

### 2. SVG Code Structural Analysis

#### Technical Error Detection
- **Coordinate & Dimension Issues:**
  - Verify `x`, `y`, `width`, `height` values
  - Ensure proper element enclosure
  - Check alignment attributes

- **Text Rendering Problems:**
  - Validate `tspan` usage for multi-line text
  - Assess `text-anchor` correctness
  - Check line spacing and readability

- **Grouping & Layering:**
  - Detect unnecessary or broken `<g>` groups
  - Identify z-index and layering conflicts

- **Styling & Visibility:**
  - Verify `fill` and `stroke` attributes
  - Check `opacity` and transparency
  - Identify clipping or masking errors

### 3. Issue Identification & Solution Methodology

#### Systematic Problem Resolution
1. Visually analyze the SVG testimonial
2. Cross-reference visual issues with code structure
3. Generate a detailed, structured issue list
4. Provide precise, implementable solutions
5. Ensure the final design meets professional standards

#### Sample Issue Documentation Format
**Issue 1: Text Overflow in Testimonial Container**
- **Cause:** Incorrect `width` or missing text wrapping
- **Solution:**
  - Implement `text-wrap="balance"`
  - Adjust `width` in text elements
  - Realign text positioning

**Issue 2: Misaligned Client Information**
- **Cause:** Incorrect coordinate positioning
- **Solution:**
  - Align `x`, `y` coordinates precisely
  - Use `dy` for optimal spacing
  - Ensure consistent text positioning

**Issue 3: Unnecessary Background Shapes**
- **Cause:** Redundant design elements
- **Solution:**
  - Remove extraneous shape elements
  - Adjust layering using `g` groups
  - Optimize visual hierarchy

### 4. Final Design Validation

#### Key Validation Criteria
- Text is perfectly aligned and contained
- Elements have precise dimensions and spacing
- Design is visually balanced and professional
- No unnecessary or broken elements remain
- Optimal readability and aesthetic appeal

### Strict Design Guidelines
- **Never compromise on visual perfection**
- **Provide exact, actionable solutions**
- **Aim for a clean, professional testimonial design**
- **Ensure every design detail serves a purpose**

:rocket: **Transforming SVG Testimonials into Visual Masterpieces!** :rocket:"""
    
    prompt = f"""
    Validate and improve the following SVG code:
    
    ```
    {svg_code}
    ```
    
    Check for any issues, optimize the code, and return only the valid, improved SVG code without any explanation or markdown formatting.
    The output should be a single SVG code block that can be directly used in an HTML document.
    """
    
    payload = {
        "contents": [{
            "parts": [
                {"text": system_prompt},
                {"text": prompt}
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "topK": 40,
            "maxOutputTokens": 2048
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    logger.info("Calling Gemini API for SVG validation")
    response = requests.post(gemini_url, headers=headers, data=json.dumps(payload))
    
    if response.status_code != 200:
        logger.error(f"Gemini API error: {response.text}")
        return svg_code  # Return original SVG if there's an error
    
    response_data = response.json()
    
    # Extract SVG code from Gemini response
    try:
        text_content = response_data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Extract SVG code from response (in case Gemini adds explanation)
        if "```" in text_content:
            # Try to extract code from markdown code blocks
            svg_lines = text_content.split("```")
            for i, block in enumerate(svg_lines):
                if i > 0 and i % 2 == 1:  # This is a code block
                    if block.strip().startswith("svg") or block.strip().startswith("<svg"):
                        svg_code = block.strip()
                        break
            # If no SVG block found, return the first code block
            for i, block in enumerate(svg_lines):
                if i > 0 and i % 2 == 1:
                    svg_code = block.strip()
                    break
        
        # If no code blocks, look for SVG tags
        elif "<svg" in text_content and "</svg>" in text_content:
            start = text_content.find("<svg")
            end = text_content.find("</svg>") + 6
            svg_code = text_content[start:end]
        else:
            svg_code = text_content.strip()
            
        # Post-process SVG to ensure proper rendering
        return post_process_svg(svg_code)
    except Exception as e:
        logger.error(f"Error processing Gemini response: {e}")
        # Even if Gemini fails, try to fix the original SVG
        return post_process_svg(svg_code)

def post_process_svg(svg_code):
    """Fix common SVG rendering issues"""
    # Make sure SVG has XML namespace
    if "<svg" in svg_code and "xmlns=" not in svg_code:
        svg_code = svg_code.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"')
    
    # Ensure SVG has width and height if not specified
    if "<svg" in svg_code and "width=" not in svg_code:
        svg_code = svg_code.replace("<svg", '<svg width="1080" height="1080"')
    
    # Fix XML entity reference issues (common parsing errors)
    svg_code = svg_code.replace("&", "&amp;")
    # Don't double-encode already correct entities
    svg_code = svg_code.replace("&amp;amp;", "&amp;")
    svg_code = svg_code.replace("&amp;lt;", "&lt;")
    svg_code = svg_code.replace("&amp;gt;", "&gt;")
    
    # Ensure proper closing of all elements (prevent XML parsing errors)
    svg_code = svg_code.replace("/>", " />")
    
    # Remove potentially problematic elements that cause rendering issues
    if "<script" in svg_code:
        logger.warning("Removing <script> tags from SVG for security")
        # Simple removal - in production you'd want a more robust parser
        start = svg_code.find("<script")
        if start > 0:
            end = svg_code.find("</script>", start)
            if end > 0:
                svg_code = svg_code[:start] + svg_code[end+9:]
    
    return svg_code

if __name__ == '__main__':
    app.run(debug=True, port=5001) 