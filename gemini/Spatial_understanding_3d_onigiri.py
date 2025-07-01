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

# メインループを制御するためのフラグ
stop_program = False

while not stop_program:
    ret, frame = cap.read()  # カメラからフレームを読み込む
    if not ret:
        print("フレームを読み込めませんでした。")
        break

    current_time = time.time()  # 現在時刻を取得

    # OpenCVのBGR形式からPILのRGB形式に変換
    # Gemini APIはPIL形式の画像を必要とします
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)

    # Geminiを使用して画像を分析
    try:
        image_response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                img,
                """
                Detect the 2D bounding boxes of "rice ball".
                Output a json list where each entry contains the object name in "label" and its 2D bounding box in "box_2d"
                """,
            ],
            config=types.GenerateContentConfig(
                temperature=0.1, response_mime_type="application/json"
            ),
        )

        response_text = image_response.text.strip()
        print(f"Gemini Raw Response: {response_text}")  # デバッグ出力

        onigiri_detected_flag = False

        # JSONとしてパースを試みる
        try:
            detected_items = json.loads(response_text)
            if isinstance(detected_items, list):
                for item in detected_items:
                    label = item.get("label", "").lower()
                    box_3d = item.get("box_2d")

                    # 「onigiri」または「rice ball」の様々な形式をチェック
                    if "rice ball" in label and box_3d:
                        print(
                            f"「{item.get('label')}」を検出しました。カメラを停止します。"
                        )
                        onigiri_detected_flag = True

                        # バウンディングボックスの描画
                        # 3Dボックスの中心とサイズから2Dの左上と右下の座標を計算
                        # frameの幅と高さを取得
                        h, w, _ = frame.shape

                        # # 正規化された座標をピクセル座標に変換
                        # # Geminiの出力が正規化されていると仮定
                        # (
                        #     x_center_norm,
                        #     y_center_norm,
                        #     _,
                        #     x_size_norm,
                        #     y_size_norm,
                        #     _,
                        #     _,
                        #     _,
                        #     _,
                        # ) = box_3d

                        # x_center = int(x_center_norm * w)
                        # y_center = int(y_center_norm * h)
                        # x_size = int(x_size_norm * w)
                        # y_size = int(y_size_norm * h)

                        # x1 = int(x_center - x_size / 2)
                        # y1 = int(y_center - y_size / 2)
                        # x2 = int(x_center + x_size / 2)
                        # y2 = int(y_center + y_size / 2)

                        # 2Dバウンディングボックスの座標を取得
                        box_3d = [v / 1000 for v in box_3d]

                        x1 = int(box_3d[0] * w)
                        y1 = int(box_3d[1] * h)
                        x2 = int(box_3d[2] * w)
                        y2 = int(box_3d[3] * h)

                        # バウンディングボックスを描画 (緑色、太さ2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # ラベルテキストを描画
                        cv2.putText(
                            frame,
                            item.get("label"),
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )

                        break  # おにぎりが見つかったら内側のループを抜ける
            else:
                print("Geminiからの応答がJSONリストではありませんでした。")

        except json.JSONDecodeError:
            print("Geminiからの応答が有効なJSONではありませんでした。")
            # フォールバック: JSONパースが失敗した場合、生レスポンステキストからキーワードをチェック
            if (
                "onigiri" in response_text.lower()
                or "rice ball" in response_text.lower()
            ):
                print(
                    "「onigiri」または「rice ball」をテキストから検出しました。カメラを停止します。"
                )
                onigiri_detected_flag = True

        if onigiri_detected_flag:
            stop_program = True  # メインループを終了するためにフラグをTrueに設定

    except Exception as e:
        print(f"Gemini API呼び出し中にエラーが発生しました: {e}")

    last_detection_time = current_time  # 検出時間を更新

    # カメラの映像を表示 (デバッグ用)
    cv2.imshow("Camera Feed", frame)

    # 'q'キーが押されたらループを抜ける
    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop_program = True

# カメラとウィンドウを解放
cap.release()
cv2.destroyAllWindows()

if stop_program:
    print("プログラムが終了しました。")