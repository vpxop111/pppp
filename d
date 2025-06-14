from flask import request, jsonify
from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO
from PIL import Image
import pytesseract
import os
import uuid
from datetime import datetime
import vtracer
from app import (
    app,
    check_vector_suitability,
    plan_design,
    generate_design_knowledge,
    pre_enhance_prompt,
    enhance_prompt_with_chat,
    generate_image_with_gpt,
    save_svg,
    logger,
    IMAGES_DIR
)
import numpy as np
import remove_text_combined
import png_to_svg_converter

# Directory for parallel pipeline outputs
PARALLEL_OUTPUTS_DIR = os.path.join(IMAGES_DIR, 'parallel')
os.makedirs(PARALLEL_OUTPUTS_DIR, exist_ok=True)

def process_ocr_svg(image_data):
    """Process OCR and generate text-based SVG"""
    image = Image.open(BytesIO(image_data))
    # Extract text bounding boxes
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    elements = []
    
    for i, txt in enumerate(ocr_data.get('text', [])):
        text = txt.strip()
        if not text:
            continue
        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]
        # sample color at center
        cx = x + w // 2
        cy = y + h // 2
        color_pixel = image.getpixel((cx, cy))
        # Handle RGB/RGBA tuple color format
        if isinstance(color_pixel, tuple) and len(color_pixel) >= 3:
            r, g, b = color_pixel[:3]
            color = f'#{r:02x}{g:02x}{b:02x}'
        else:
            color = '#000000'
        font_size = h
        # y+h to adjust baseline
        elements.append(
            f'<text x="{x}" y="{y + h}" fill="{color}" '
            f'font-size="{font_size}" font-family="sans-serif">{text}</text>'
        )
    
    width, height = image.size
    svg_code = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{"".join(elements)}</svg>'
    )
    
    # Save SVG
    svg_filename = save_svg(svg_code, prefix='text_svg')
    return svg_code, svg_filename

def create_mask(image_data):
    """Create an alpha mask from white text in the input image"""
    # Load and convert to grayscale
    img = Image.open(BytesIO(image_data)).convert("RGB")
    gray = img.convert("L")
    
    # Threshold to create binary mask - higher threshold for white text
    thr = 240  # Adjusted for white text
    mask_arr = np.array(gray)
    mask_binary = (mask_arr > thr).astype(np.uint8) * 255
    mask = Image.fromarray(mask_binary, mode="L")
    
    # Convert to RGBA with alpha channel
    mask_rgba = mask.convert("RGBA")
    mask_rgba.putalpha(mask)
    
    # Save mask to BytesIO
    mask_bytes = BytesIO()
    mask_rgba.save(mask_bytes, format='PNG')
    mask_bytes.seek(0)
    return mask_bytes

def process_clean_svg(image_data):
    """Process text removal and convert to clean SVG"""
    # Save the original image bytes to a temporary PNG file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_input_path = f"temp_input_{timestamp}_{uuid.uuid4()}.png"
    with open(temp_input_path, "wb") as f:
        f.write(image_data)

    # Create a mask for text removal using remove_text_combined
    mask_path = remove_text_combined.create_mask(temp_input_path)

    # Remove text from the image using remove_text_combined
    edited_png_path = remove_text_combined.remove_text(temp_input_path, mask_path)

    # Convert the edited PNG to SVG using png_to_svg_converter
    output_svg_path = png_to_svg_converter.convert_png_to_svg(edited_png_path)

    # Read the generated SVG content
    with open(output_svg_path, 'r') as f:
        clean_svg_code = f.read()

    # Cleanup only the temporary input file
    os.remove(temp_input_path)

    # Return clean SVG code and paths for mask and edited PNG
    return clean_svg_code, output_svg_path, mask_path, edited_png_path

@app.route('/api/generate-parallel-svg', methods=['POST'])
def generate_parallel_svg():
    """Pipeline: Stages 1-6 image gen, then parallel Stage 7: OCR+SVG and Clean SVG generation"""
    data = request.json or {}
    user_input = data.get('prompt', '')
    skip_enhancement = data.get('skip_enhancement', False)

    if not user_input:
        return jsonify({'error': 'No prompt provided'}), 400

    logger.info('=== PARALLEL SVG PIPELINE START ===')

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
    image_base64, image_filename = generate_image_with_gpt(enhanced_prompt)
    image_data = base64.b64decode(image_base64)

    # Stage 7: Parallel Processing
    logger.info('Stage 7: Parallel Processing - OCR+SVG and Clean SVG')
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        ocr_future = executor.submit(process_ocr_svg, image_data)
        clean_future = executor.submit(process_clean_svg, image_data)
        
        # Get results
        text_svg_code, text_svg_path = ocr_future.result()
        clean_svg_code, clean_svg_path, mask_path, edited_png_path = clean_future.result()

    # Create a session subfolder and move outputs there
    session_folder = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    output_folder = os.path.join(PARALLEL_OUTPUTS_DIR, session_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Move generated image into session folder
    src_image = os.path.join(IMAGES_DIR, image_filename)
    dst_image = os.path.join(output_folder, image_filename)
    os.rename(src_image, dst_image)

    # Move text SVG into session folder
    src_text_svg = os.path.join(IMAGES_DIR, text_svg_path)
    dst_text_svg = os.path.join(output_folder, text_svg_path)
    os.rename(src_text_svg, dst_text_svg)

    # Move cleaned SVG into session folder
    if not os.path.isabs(clean_svg_path):
        src_clean_svg = os.path.join(os.getcwd(), clean_svg_path)
    else:
        src_clean_svg = clean_svg_path
    dst_clean_svg = os.path.join(output_folder, os.path.basename(clean_svg_path))
    os.rename(src_clean_svg, dst_clean_svg)

    # Move mask image into session folder
    src_mask = mask_path if os.path.isabs(mask_path) else mask_path
    dst_mask = os.path.join(output_folder, os.path.basename(mask_path))
    os.rename(src_mask, dst_mask)
    # Base URL for parallel outputs
    base_url = '/static/images/parallel'
    mask_url = f"{base_url}/{session_folder}/{os.path.basename(mask_path)}"

    # Move cleaned PNG (converter input) into session folder
    src_edited_png = edited_png_path if os.path.isabs(edited_png_path) else edited_png_path
    dst_edited_png = os.path.join(output_folder, os.path.basename(edited_png_path))
    os.rename(src_edited_png, dst_edited_png)
    edited_png_url = f"{base_url}/{session_folder}/{os.path.basename(edited_png_path)}"

    # Construct URLs for client access
    image_url = f"{base_url}/{session_folder}/{image_filename}"
    text_svg_url = f"{base_url}/{session_folder}/{text_svg_path}"
    clean_svg_url = f"{base_url}/{session_folder}/{os.path.basename(clean_svg_path)}"

    return jsonify({
        'original_prompt': user_input,
        'image_url': image_url,
        'mask': {
            'path': f"parallel/{session_folder}/{os.path.basename(mask_path)}",
            'url': mask_url
        },
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
        'stage': 7
    })

if __name__ == '__main__':
    # Run standalone on port 5004
    app.run(host='127.0.0.1', port=5004, debug=True) 