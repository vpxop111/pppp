from flask import request, jsonify
from app import (
    app,
    check_vector_suitability,
    plan_design,
    generate_design_knowledge,
    pre_enhance_prompt,
    enhance_prompt_with_chat,
    generate_image_with_gpt,
    save_svg,
    logger
)
import base64
from io import BytesIO
from PIL import Image
import pytesseract
import os
import uuid
from datetime import datetime
from openai import OpenAI

# GPT client for text extraction
chat_client = OpenAI()

@app.route('/api/generate-text-svg', methods=['POST'])
def generate_image_text_svg():
    """Pipeline: Stages 1–6 image gen, then Stage 7: text OCR & SVG generation"""
    data = request.json or {}
    user_input = data.get('prompt', '')
    skip_enhancement = data.get('skip_enhancement', False)

    if not user_input:
        return jsonify({'error': 'No prompt provided'}), 400

    logger.info('=== IMAGE→TEXT→SVG PIPELINE START ===')

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

    # Stage 7: OCR and SVG generation via GPT-4.1-mini
    logger.info('Stage 7: OCR and SVG Generation via GPT-4.1-mini')
    # Decode and save the image to disk
    image_data = base64.b64decode(image_base64)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f"text_input_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    input_filepath = os.path.join(IMAGES_DIR, input_filename)
    with open(input_filepath, "wb") as f:
        f.write(image_data)
    # Determine dimensions
    img = Image.open(input_filepath)
    width, height = img.size
    view_box = f"0 0 {width} {height}"
    image_url = f"/static/images/{input_filename}"
    # Prepare prompt
    messages = [
        {"role": "system", "content": (
            "You are a precise SVG code generator. "
            "Given image dimensions and a list of text elements with their positions (x, y), font sizes, and fill colors, "
            "output only a valid SVG string with the correct <svg> wrapper and one <text> tag per element, "
            "using each 'text' value exactly as provided (no changes). Do not include explanations or markup extras."
        )},
        {"role": "user", "content": (
            f"Here is an image at URL: {image_url}. The image width is {width}px and height is {height}px. "
            "Generate a valid SVG that begins with:\n"
            f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"{view_box}\">\n"
            "Then include one <text> element per visible text string, each with appropriate x, y coordinates, "
            "font-family=\"sans-serif\", font-size matching the original, and fill color in hex. Close the SVG at the end. "
            "Return only the complete SVG code."
        )}
    ]
    response = chat_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=1
    )
    svg_code = response.choices[0].message.content.strip()
    # Save and return
    svg_filename = save_svg(svg_code, prefix='text_svg')
    return jsonify({
        'original_prompt': user_input,
        'image_url': f'/static/images/{image_filename}',
        'svg_code': svg_code,
        'svg_path': svg_filename,
        'stage': 7
    })

if __name__ == '__main__':
    # Run standalone on port 5003
    app.run(host='127.0.0.1', port=5003, debug=True) 