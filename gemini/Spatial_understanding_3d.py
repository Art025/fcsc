import os
from google import genai
from google.genai import types
from PIL import Image

# 環境変数からAPIキーを取得
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.5-flash"  # @param ["gemini-1.5-flash-latest","gemini-2.5-flash-lite-preview-06-17","gemini-2.5-flash","gemini-2.5-pro"] {"allow-input":true}

# Load the selected image
img = Image.open("kitchen.jpg")

# Analyze the image using Gemini
image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        img,
        """
          Detect the 3D bounding boxes of no more than 10 items.
          Output a json list where each entry contains the object name in "label" and its 3D bounding box in "box_3d"
          The 3D bounding box format should be [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw].
        """,
    ],
    config=types.GenerateContentConfig(temperature=0.5),
)

# Check response
print(image_response.text)
