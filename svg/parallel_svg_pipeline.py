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
        # Handle RGB tuple color format
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