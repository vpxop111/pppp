from flask import request, jsonify
from app import (
    app,
    check_vector_suitability,
    plan_design,
    generate_design_knowledge,
    pre_enhance_prompt,
    enhance_prompt_with_chat,
    generate_image_with_gpt,
    logger
)

@app.route('/api/generate-image', methods=['POST'])
def generate_image_only():
    """Image-only generation endpoint covering Stages 1–6:
    1. Vector Suitability Check
    2. Design Planning
    3. Design Knowledge Generation
    4. Pre-Enhancement (optional)
    5. Technical Enhancement (optional)
    6. Image Generation via GPT Image-1
    """
    data = request.json or {}
    user_input = data.get('prompt', '')
    skip_enhancement = data.get('skip_enhancement', False)

    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    logger.info("===== IMAGE-ONLY PIPELINE: Starting request =====")

    # Stage 1: Vector Suitability Check
    logger.info("Stage 1: Vector Suitability Check")
    vector_suitability = check_vector_suitability(user_input)
    if vector_suitability.get('not_suitable', False):
        return jsonify({
            "error": "Not suitable for SVG",
            "guidance": vector_suitability.get('guidance'),
            "stage": 1
        }), 400

    # Stage 2: Design Planning
    logger.info("Stage 2: Design Planning")
    design_plan = plan_design(user_input)

    # Stage 3: Design Knowledge Generation
    logger.info("Stage 3: Design Knowledge Generation")
    design_knowledge = generate_design_knowledge(design_plan, user_input)

    # Combine context for prompt enhancements
    design_context = f"""Design Plan:\n{design_plan}\n\nDesign Knowledge and Best Practices:\n{design_knowledge}\n\nOriginal Request:\n{user_input}"""

    # Stage 4 & 5: Prompt Enhancements
    if skip_enhancement:
        pre_enhanced_prompt = user_input
        enhanced_prompt = user_input
    else:
        logger.info("Stage 4: Pre-Enhancement")
        pre_enhanced_prompt = pre_enhance_prompt(design_context)
        logger.info("Stage 5: Technical Enhancement")
        enhanced_prompt = enhance_prompt_with_chat(pre_enhanced_prompt)

    # Stage 6: Image Generation via GPT-Image
    logger.info("Stage 6: Image Generation via GPT-Image")
    image_base64, image_filename = generate_image_with_gpt(enhanced_prompt)

    # Build response
    return jsonify({
        "original_prompt": user_input,
        "design_plan": design_plan,
        "design_knowledge": design_knowledge,
        "pre_enhanced_prompt": pre_enhanced_prompt,
        "enhanced_prompt": enhanced_prompt,
        "image_base64": image_base64,
        "image_url": f"/static/images/{image_filename}",
        "stage": 6
    })

if __name__ == '__main__':
    # Run this image-only pipeline standalone on port 5002
    app.run(host='127.0.0.1', port=5002, debug=True) 