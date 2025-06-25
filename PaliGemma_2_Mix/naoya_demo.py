import os
import io

import requests
import numpy as np
import PIL
import kagglehub
import keras
import keras_hub

os.environ["KERAS_BACKEND"] = "jax"


model_id = kagglehub.model_download("keras/paligemma2/keras/pali_gemma2_mix_3b_448")

keras.config.set_floatx("bfloat16")
pali_gemma_lm = keras_hub.models.PaliGemmaCausalLM.from_preset(model_id)


def read_image(url):
    contents = io.BytesIO(requests.get(url).content)
    image = PIL.Image.open(contents)
    image = np.array(image)
    # Remove alpha channel if neccessary.
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


img = read_image("/home/roboworks/fcsc2025/PaliGemma_2_Mix/car.jpg")

prompt = "answer en where is the cow standing?\n"
output = pali_gemma_lm.generate(
    inputs={
        "images": img,
        "prompts": prompt,
    }
)
print(output)
