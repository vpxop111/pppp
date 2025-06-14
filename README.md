# AI SVG Generator Pipeline

This server implements a pipeline that processes user input through multiple AI models to generate SVG images:

1. User input is enhanced by an OpenAI model (prompt enhancer)
2. Enhanced prompt is sent to another OpenAI model to generate SVG code
3. SVG code is validated and improved by Google's Gemini model

## Setup and Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the server:

```bash
python app.py
```

The server will run on http://localhost:5000.

## API Usage

Send a POST request to `/api/generate-svg` with a JSON body containing a prompt:

```json
{
  "prompt": "Draw a mountain landscape"
}
```

The response will contain:
- `original_prompt`: The user's original input
- `enhanced_prompt`: The enhanced prompt created by the first AI model
- `svg_code`: The validated and improved SVG code

## Client Integration

See `client_integration.js` for an example of how to call the API from a client-side application.

## Notes

- API keys are hardcoded in the `app.py` file for simplicity. In a production environment, use environment variables.
- The server has CORS enabled so it can be called from browser-based applications. 