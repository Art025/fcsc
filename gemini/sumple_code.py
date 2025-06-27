from google import genai
from google.genai import types
from PIL import Image


client = genai.Client(api_key="GEMINI_API_KEY")  # @param {type:"string"}
MODEL_ID = "gemini-2.5-flash"  # @param ["gemini-1.5-flash-latest","gemini-2.5-flash-lite-preview-06-17","gemini-2.5-flash","gemini-2.5-pro"] {"allow-input":true}

# Load and resize image
img = Image.open("tool.png")
img = img.resize(
    (800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS
)  # Resizing to speed-up rendering

# Analyze the image using Gemini
image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        img,
        """
          Point to no more than 10 items in the image, include spill.
          The answer should follow the json format: [{"point": , "label": }, ...]. The points are in [y, x] format normalized to 0-1000.
        """,
    ],
    config=types.GenerateContentConfig(temperature=0.5),
)

# Check response
print(image_response.text)
