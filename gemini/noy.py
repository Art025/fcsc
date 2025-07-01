# onigiri.pngのバウンディングボックスを作成した結果をedtected_onigiri.jpgに保存する

import os
import cv2  # OpenCVをインポート
from google import genai
from google.genai import types
from PIL import Image
import time  # timeモジュールをインポート
import json  # jsonモジュールをインポート

# 環境変数からAPIキーを取得
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.5-flash"  # @param ["gemini-1.5-flash-latest","gemini-2.5-flash-lite-preview-06-17","gemini-2.5-flash","gemini-2.5-pro"] {"allow-input":true}


def detect(frame, target_label):
    """検出する関数

    Args:
        frame: カメラからのフレーム
        target_label: 検出したいラベル名

    Returns:
        tuple: 検出結果のフラグ、ラベル名、2Dバウンディングボックスの座標

    2Dバウンディングボックスの座標は(x1, y1, x2, y2)の形式で返す。
    もし検出されなかった場合は、(False, None, None)
    """

    if not isinstance(frame, Image.Image):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
    else:
        img = frame

    try:
        image_response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                img,
                f"Detect the 2d bounding boxes of the {target_label}.",
            ],
            config=types.GenerateContentConfig(
                temperature=0.1, response_mime_type="application/json"
            ),
        )

        response_text = image_response.text.strip()
        print(f"Gemini Raw Response: {response_text}")  # デバッグ出力

        detected_items = json.loads(response_text)
        if isinstance(detected_items, list):
            for item in detected_items:
                label = item.get("label", "").lower()
                box_2d = item.get("box_2d")

                if target_label in label and box_2d:
                    box_2d = [v / 1000 for v in box_2d]
                    h, w, _ = frame.shape
                    x1 = int(box_2d[0] * w)
                    y1 = int(box_2d[1] * h)
                    x2 = int(box_2d[2] * w)
                    y2 = int(box_2d[3] * h)
                    return True, item.get("label"), (x1, y1, x2, y2)
        else:
            print("Geminiからの応答がJSONリストではありませんでした。")

    except json.JSONDecodeError:
        print("Geminiからの応答が有効なJSONではありませんでした。")

    return False, None, None


def draw_bounding_box(frame, box, label):
    """バウンディングボックスを描画する関数

    Args:
        frame: カメラからのフレーム
        box: 2Dバウンディングボックスの座標 (x1, y1, x2, y2)
        label: ラベル名
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )


def main():
    img = cv2.imread("onigiri.png")

    dets = detect(img, "rice_ball")

    if dets[0]:
        print(f"「{dets[1]}」を検出しました。バウンディングボックス: {dets[2]}")
        draw_bounding_box(img, dets[2], dets[1])
        cv2.imwrite("detected_onigiri.jpg", img)


if __name__ == "__main__":
    main()
