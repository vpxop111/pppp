from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO
from PIL import Image
import pytesseract
import os
import uuid
from datetime import datetime
import vtracer
from shared_functions import (
    check_vector_suitability,
    plan_design,
    generate_design_knowledge,
    pre_enhance_prompt,
    enhance_prompt_with_chat
)
from utils import (
    generate_image_with_gpt,
    save_svg,
    logger,
    IMAGES_DIR
)
import numpy as np
import remove_text_simple
import png_to_svg_converter
from openai import OpenAI
import requests
import re
import logging
from utils import (
    PARALLEL_OUTPUTS_DIR,
    save_image,
    generate_image_with_gpt
)
import sys
import traceback

# Configure logging
logger = logging.getLogger(__name__)

def init_parallel_pipeline():
    """Initialize the parallel SVG pipeline"""
    try:
        logger.info("Initializing parallel SVG pipeline")
        
        # Try to verify vtracer is available, but don't fail if it's not
        vtracer_available = False
        try:
            import vtracer
            if hasattr(vtracer, 'convert_image_to_svg_path') or hasattr(vtracer, 'convert_image_to_svg_py'):
                vtracer_available = True
                logger.info("vtracer module found and verified")
            else:
                logger.warning("vtracer module found but missing expected functions")
        except ImportError:
            logger.warning("vtracer module not available - parallel SVG features will be limited")
        
        # Verify directories exist
        if not os.path.exists(PARALLEL_OUTPUTS_DIR):
            os.makedirs(PARALLEL_OUTPUTS_DIR, exist_ok=True)
            logger.info(f"Created PARALLEL_OUTPUTS_DIR: {PARALLEL_OUTPUTS_DIR}")
        
        if vtracer_available:
            logger.info("Parallel SVG pipeline initialized successfully with vtracer support")
        else:
            logger.info("Parallel SVG pipeline initialized with limited functionality (no vtracer)")
        
        return vtracer_available
    except Exception as e:
        logger.error(f"Failed to initialize parallel SVG pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        # Don't raise the exception - allow the app to continue without parallel pipeline
        logger.warning("Continuing without parallel SVG pipeline support")
        return False

# Instantiate a GPT client for chat completions
chat_client = OpenAI()

def build_advanced_image_prompt(user_input, design_context):
    """Build an advanced image prompt optimized for parallel SVG processing"""

    # Analyze user input for design type and requirements
    user_lower = user_input.lower()

    # Design type detection with more specific categories
    design_types = {
        'poster': ['poster', 'flyer', 'announcement', 'event', 'coming soon', 'promotion'],
        'logo': ['logo', 'brand', 'company', 'business', 'startup', 'identity'],
        'card': ['card', 'testimonial', 'review', 'quote', 'recommendation'],
        'banner': ['banner', 'header', 'cover', 'social media', 'facebook', 'instagram'],
        'infographic': ['infographic', 'chart', 'data', 'statistics', 'info'],
        'certificate': ['certificate', 'award', 'diploma', 'achievement'],
        'invitation': ['invitation', 'invite', 'party', 'wedding', 'event'],
        'menu': ['menu', 'restaurant', 'food', 'cafe', 'dining'],
        'brochure': ['brochure', 'pamphlet', 'leaflet', 'booklet']
    }

    detected_type = 'general'
    for design_type, keywords in design_types.items():
        if any(keyword in user_lower for keyword in keywords):
            detected_type = design_type
            break

    # Extract key elements from design context
    context_elements = []
    if design_context:
        # Extract key phrases from design context
        context_lines = design_context.split('\n')
        for line in context_lines[:10]:  # First 10 lines usually contain key info
            if any(keyword in line.lower() for keyword in ['color', 'font', 'style', 'layout', 'theme']):
                context_elements.append(line.strip())

    # Build prompt components
    prompt_parts = []

    # 1. Core request with design type optimization
    if detected_type == 'poster':
        prompt_parts.append(f"Create a professional poster design: {user_input}")
        prompt_parts.append("Design requirements: Bold typography, clear hierarchy, eye-catching visuals, structured layout")
        prompt_parts.append("Visual style: High-impact graphics, vibrant colors, professional composition, marketing-focused")
    elif detected_type == 'logo':
        prompt_parts.append(f"Create a distinctive logo design: {user_input}")
        prompt_parts.append("Design requirements: Simple memorable shapes, scalable graphics, clean typography, brand identity")
        prompt_parts.append("Visual style: Minimalist approach, strong contrast, vector-friendly elements, timeless design")
    elif detected_type == 'card':
        prompt_parts.append(f"Create an elegant card design: {user_input}")
        prompt_parts.append("Design requirements: Professional layout, readable typography, trustworthy appearance, balanced composition")
        prompt_parts.append("Visual style: Clean background, subtle elegance, credible aesthetics, testimonial-focused")
    elif detected_type == 'banner':
        prompt_parts.append(f"Create a dynamic banner design: {user_input}")
        prompt_parts.append("Design requirements: Horizontal layout, social media optimized, engaging visuals, clear messaging")
        prompt_parts.append("Visual style: Platform-appropriate, scroll-stopping appeal, brand consistency, modern aesthetics")
    else:
        prompt_parts.append(f"Create a professional graphic design: {user_input}")
        prompt_parts.append("Design requirements: Versatile layout, clear visual hierarchy, professional appearance, multi-purpose design")
        prompt_parts.append("Visual style: Modern aesthetics, balanced composition, adaptable elements, universal appeal")

    # 2. Technical specifications for optimal SVG conversion
    prompt_parts.append("Technical specs: 1024x1024 resolution, high contrast elements, clear edge definition, distinct boundaries")
    prompt_parts.append("SVG optimization: Vector-friendly graphics, clean background separation, text-image distinction, sharp details")

    # 3. Quality and aesthetic requirements
    prompt_parts.append("Quality standards: Professional finish, polished appearance, commercial-grade design, publication-ready")
    prompt_parts.append("Color approach: Vibrant yet balanced palette, good contrast ratios, harmonious color scheme, brand-appropriate")

    # 4. Add context elements if available
    if context_elements:
        context_summary = " | ".join(context_elements[:3])  # Top 3 context elements
        prompt_parts.append(f"Design context: {context_summary}")

    # 5. Parallel processing optimization
    prompt_parts.append("Processing optimization: Clear text-background separation, distinct graphic elements, OCR-friendly text placement")

    # Combine all parts with separators
    final_prompt = " || ".join(prompt_parts)

    # Ensure prompt length is manageable
    if len(final_prompt) > 1200:
        # Keep the most important parts
        essential_parts = prompt_parts[:4]  # Core request + technical specs
        final_prompt = " || ".join(essential_parts)

    return final_prompt

def process_ocr_svg(image_data):
    """Generate a text-only SVG using GPT-4.1-mini by passing the image directly to the chat API."""
    # Base64-encode the PNG image
    img_b64 = base64.b64encode(image_data).decode('utf-8')
    # Build prompts matching app.py's generate_svg_from_image style
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
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY_SVG')}"
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
    # Save the original image bytes to a temporary PNG file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_input_path = f"temp_input_{timestamp}_{uuid.uuid4()}.png"
    with open(temp_input_path, "wb") as f:
        f.write(image_data)

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

    # Read the generated SVG content
    with open(output_svg_path, 'r') as f:
        clean_svg_code = f.read()

    # Cleanup the temporary input file
    os.remove(temp_input_path)

    # Return clean SVG code and paths
    return clean_svg_code, output_svg_path, edited_png_path

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

def combine_svgs(text_svg_code, traced_svg_code):
    """Combine text and path SVGs using GPT-4o-mini to produce a unified SVG."""
    import time
    logger.info('Stage 8: Combining SVGs using HTTP API')
    logger.info('Stage 8.1: Starting SVG combination process')

    # Reduce content size to fit within token limits
    original_text_size = len(text_svg_code.encode('utf-8')) if isinstance(text_svg_code, str) else len(text_svg_code)
    original_path_size = len(traced_svg_code.encode('utf-8')) if isinstance(traced_svg_code, str) else len(traced_svg_code)
    logger.info(f'Stage 8.2: Original sizes - Text SVG: {original_text_size} bytes, Traced SVG: {original_path_size} bytes')

    # Reduce traced SVG size significantly as it's usually the largest
    reduced_traced_svg = reduce_svg_content(traced_svg_code, max_chars=30000)
    reduced_text_svg = reduce_svg_content(text_svg_code, max_chars=5000)

    reduced_text_size = len(reduced_text_svg.encode('utf-8'))
    reduced_path_size = len(reduced_traced_svg.encode('utf-8'))
    logger.info(f'Stage 8.3: Reduced sizes - Text SVG: {reduced_text_size} bytes, Traced SVG: {reduced_path_size} bytes')

    # Use a simpler combination approach if content is still too large
    total_size = reduced_text_size + reduced_path_size
    if total_size > 40000:  # Still too large, use simple combination
        logger.info('Stage 8.4: Content still large, using simple combination approach')
        return simple_combine_svgs(reduced_text_svg, reduced_traced_svg)

    logger.info('Stage 8.4: Preparing HTTP API request')
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
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }
    payload = {
        'model': 'gpt-4o-mini',
        'messages': messages,
        'temperature': 0,
        'max_tokens': 4000
    }

    start_time = time.time()
    logger.info('Stage 8.5: Sending HTTP request to OpenAI API')
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        api_response_time = time.time() - start_time
        logger.info(f'Stage 8.6: HTTP response received in {api_response_time:.2f} seconds')

        if resp.status_code != 200:
            logger.error(f'Stage 8 ERROR: {resp.status_code} - {resp.text}')
            logger.info('Stage 8.7: Falling back to simple combination')
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
        logger.info(f'Stage 8.8: Combined SVG size: {combined_size} bytes')
        total_time = time.time() - start_time
        logger.info(f'Stage 8.9: SVG combination completed in {total_time:.2f} seconds total')
        return combined_svg

    except Exception as e:
        logger.error(f'Stage 8 API ERROR: {str(e)}')
        logger.info('Stage 8.10: Falling back to simple combination')
        return simple_combine_svgs(reduced_text_svg, reduced_traced_svg)

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

    logger.info('Stage 8.11: Using advanced SVG combination')

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

    logger.info(f'Stage 8.12: Extracted elements - Traced: {len(traced_elements["paths"])} paths, Text: {len(text_elements["texts"])} texts')

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

    logger.info(f'Stage 8.13: Advanced combination completed - Final size: {len(combined_svg)} characters')
    return combined_svg

def generate_parallel_svg_pipeline(data):
    """Generate SVG using parallel pipeline with vtracer"""
    try:
        user_input = data.get('prompt', '')
        skip_enhancement = data.get('skip_enhancement', False)

        if not user_input:
            return {'error': 'No prompt provided'}

        logger.info('=== PARALLEL SVG PIPELINE START ===')
        logger.info(f'Processing prompt: {user_input[:100]}...')

        # Check if vtracer is available
        vtracer_available = False
        try:
            import vtracer
            if hasattr(vtracer, 'convert_image_to_svg_path') or hasattr(vtracer, 'Config'):
                vtracer_available = True
        except ImportError:
            pass
        
        if not vtracer_available:
            logger.warning('vtracer not available - falling back to standard SVG generation')
            # Fallback to the standard generation method from shared_functions
            from shared_functions import (
                check_vector_suitability,
                plan_design,
                generate_design_knowledge,
                pre_enhance_prompt,
                enhance_prompt_with_chat
            )
            
            # Stage 1: Vector Suitability Check
            vector_suitability = check_vector_suitability(user_input)
            if vector_suitability.get('not_suitable', False):
                return {
                    'error': 'Not suitable for SVG',
                    'guidance': vector_suitability.get('guidance'),
                    'stage': 'vector_check'
                }
            
            # Stages 2-5: Design planning and enhancement
            design_plan = plan_design(user_input)
            design_knowledge = generate_design_knowledge(design_plan, user_input)
            design_context = f"""Design Plan:\n{design_plan}\n\nDesign Knowledge:\n{design_knowledge}\n\nOriginal Request:\n{user_input}"""
            
            if not skip_enhancement:
                pre_enhanced = pre_enhance_prompt(design_context)
                enhanced_prompt = enhance_prompt_with_chat(pre_enhanced)
            else:
                enhanced_prompt = user_input
            
            # Stage 6: Generate image using GPT
            image_base64, image_filename = generate_image_with_gpt(enhanced_prompt)
            
            # Stage 7: Use a simple PNG to SVG converter as fallback
            try:
                import png_to_svg_converter
                image_data = base64.b64decode(image_base64)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                input_filename = f"fallback_input_{timestamp}_{uuid.uuid4().hex[:8]}.png"
                input_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, input_filename)
                
                with open(input_filepath, "wb") as f:
                    f.write(image_data)
                
                output_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, f"fallback_output_{timestamp}.svg")
                png_to_svg_converter.convert_png_to_svg(input_filepath, output_filepath)
                
                with open(output_filepath, 'r') as f:
                    svg_code = f.read()
                
                svg_filename = save_svg(svg_code, prefix='fallback_svg')
                
                return {
                    'original_prompt': user_input,
                    'image_url': f'/static/images/{image_filename}',
                    'svg_code': svg_code,
                    'svg_path': svg_filename,
                    'stage': 'fallback_complete',
                    'note': 'Generated using fallback method (vtracer not available)'
                }
                
            except Exception as fallback_error:
                logger.error(f"Fallback SVG generation failed: {str(fallback_error)}")
                # Return just the image if SVG conversion fails
                return {
                    'original_prompt': user_input,
                    'image_url': f'/static/images/{image_filename}',
                    'svg_code': None,
                    'svg_path': None,
                    'stage': 'image_only',
                    'note': 'SVG conversion unavailable - image generated successfully'
                }

        # Original vtracer-based processing
        logger.info('Using vtracer for SVG conversion')
        
        # Generate image using GPT
        image_base64, image_filename = generate_image_with_gpt(user_input)
        logger.info(f'Image generated: {image_filename}')

        # Save the image for vtracer processing
        image_data = base64.b64decode(image_base64)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"parallel_input_{timestamp}_{uuid.uuid4().hex[:8]}.png"
        input_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, input_filename)
        
        with open(input_filepath, "wb") as f:
            f.write(image_data)
        logger.info(f'Input image saved: {input_filepath}')

        # Process with vtracer
        output_svg = f"parallel_output_{timestamp}_{uuid.uuid4().hex[:8]}.svg"
        output_filepath = os.path.join(PARALLEL_OUTPUTS_DIR, output_svg)

        config = vtracer.Config(
            color_mode="color",
            hierarchical=False,
            filter_speckle=4,
            corner_threshold=60,
            length_threshold=4.0,
            splice_threshold=45,
            path_precision=8,
        )
        
        logger.info('Starting vtracer conversion...')
        vtracer.convert_image_to_svg_path(
            input_filepath,
            output_filepath,
            config
        )
        logger.info(f'Vtracer conversion complete: {output_filepath}')

        # Read the generated SVG
        with open(output_filepath, 'r') as f:
            svg_code = f.read()

        # Save final SVG and return
        svg_filename = save_svg(svg_code, prefix='parallel_svg')
        logger.info(f'Final SVG saved: {svg_filename}')
        
        return {
            'original_prompt': user_input,
            'image_url': f'/static/images/{image_filename}',
            'svg_code': svg_code,
            'svg_path': svg_filename,
            'stage': 'complete'
        }

    except Exception as e:
        logger.error(f"Error in parallel SVG pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        return {'error': str(e), 'stage': 'failed'}, 500

 