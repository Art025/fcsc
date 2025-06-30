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

# USBカメラの初期化
cap = cv2.VideoCapture(0)  # 0は通常、デフォルトのWebカメラを指します

if not cap.isOpened():
    raise IOError(
        "Webカメラを開けませんでした。カメラが接続されているか確認してください。"
    )

print("カメラを起動しました。'q'キーを押すと終了します。")

last_detection_time = time.time()  # 最後の検出時間を記録する変数を初期化
detection_interval = 5  # 検出間隔を5秒に設定

while True:
    ret, frame = cap.read()  # カメラからフレームを読み込む
    if not ret:
        print("フレームを読み込めませんでした。")
        break

    current_time = time.time()  # 現在時刻を取得

    # 5秒ごとに検出を実行
    if current_time - last_detection_time >= detection_interval:
        # OpenCVのBGR形式からPILのRGB形式に変換
        # Gemini APIはPIL形式の画像を必要とします
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)

        # Analyze the image using Gemini
        try:
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

            # Geminiからの応答をJSONとしてパース
            try:
                detected_items = json.loads(image_response.text)
                # 検出されたアイテムの中に「onigiri」があるかチェック
                for item in detected_items:
                    if (
                        item.get("label") == "Onigiri"
                        or item.get("label") == "onigiri"
                        or item.get("label") == "rice ball"
                    ):
                        print("「rice ball」を検出しました。カメラを停止します。")
                        # onigiriが検出されたらループを抜ける
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()  # プログラムを終了
            except json.JSONDecodeError:
                print("Geminiからの応答が有効なJSONではありませんでした。")

        except Exception as e:
            print(f"Gemini API呼び出し中にエラーが発生しました: {e}")

        last_detection_time = current_time  # 検出時間を更新

    # カメラの映像を表示 (デバッグ用)
    cv2.imshow("Camera Feed", frame)

    # 'q'キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# カメラとウィンドウを解放
cap.release()
cv2.destroyAllWindows()
