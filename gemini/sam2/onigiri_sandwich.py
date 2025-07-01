import os
import cv2  # OpenCVをインポート
from google import genai
from google.genai import types
from PIL import Image
import time  # timeモジュールをインポート
import json  # jsonモジュールをインポート
from ultralytics import SAM

# 環境変数からAPIキーを取得
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.5-flash"  # @param ["gemini-1.5-flash-latest","gemini-2.5-flash-lite-preview-06-17","gemini-2.5-flash","gemini-2.5-pro"] {"allow-input":true}


def detect(frame, target_label):
    """検出する関数"""
    if not isinstance(frame, Image.Image):
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
                    h, w, _ = frame.shape
                    x1 = int(box_2d[0] / 1000 * w)
                    y1 = int(box_2d[1] / 1000 * h)
                    x2 = int(box_2d[2] / 1000 * w)
                    y2 = int(box_2d[3] / 1000 * h)
                    return True, item.get("label"), (x1, y1, x2, y2)
        else:
            print("Geminiからの応答がJSONリストではありませんでした。")

    except json.JSONDecodeError:
        print("Geminiからの応答が有効なJSONではありませんでした。")
    except Exception as e:
        print(f"Gemini検出中にエラーが発生しました: {e}")

    return False, None, None


def draw_label_on_image(
    frame, position, label, color=(255, 0, 0)
):  # デフォルト色を赤に変更
    """指定された位置にラベルテキストを描画する関数"""
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    input_image_path = "onigirisand.jpg"  # 複数の物体を含む画像を想定
    target_labels = ["rice_ball", "sandwich"]  # 検出したいオブジェクトのリスト

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Ultralytics SAMモデルのロード (一度だけロード)
    try:
        sam_model = SAM("sam2.1_b.pt")
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return

    # 各ターゲットラベルに対して処理を実行
    for target_label in target_labels:
        print(f"\n--- Detecting: {target_label} ---")
        output_detected_image_path = (
            f"gemini_detected_{target_label}.jpg"  # Geminiのバウンディングボックスあり
        )
        output_sam_image_path = f"sam_segmented_with_gemini_label_{target_label}.jpg"  # SAMのセグメンテーション＋Geminiラベル

        # Gemini APIでバウンディングボックスを検出
        detected_flag, gemini_label, detected_bbox_gemini = detect(
            img, target_label
        )  # 変数名を変更

        if detected_flag:
            print(
                f"「{gemini_label}」を検出しました。Geminiバウンディングボックス: {detected_bbox_gemini}"
            )
            # Gemini検出結果の画像を保存 (赤いバウンディングボックスとラベルで)
            img_gemini_drawn = img.copy()
            x1, y1, x2, y2 = detected_bbox_gemini
            cv2.rectangle(
                img_gemini_drawn, (x1, y1), (x2, y2), (255, 0, 0), 2
            )  # 赤色の四角
            cv2.putText(
                img_gemini_drawn,
                gemini_label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.imwrite(output_detected_image_path, img_gemini_drawn)
            print(
                f"Geminiによる検出結果を {output_detected_image_path} に保存しました。"
            )

            # 検出されたバウンディングボックスを使用してSAMで推論
            results = sam_model(img, bboxes=[x1, y1, x2, y2])

            try:
                plot_img_sam = results[0].plot()

                # Geminiのラベルを描画するY座標を調整 (SAMのボックスの上)
                label_y_position = y1 - 10
                if label_y_position < 0:
                    label_y_position = 15

                # Geminiのラベルを描画 (バウンディングボックスは描画しない)
                draw_label_on_image(
                    plot_img_sam,
                    (x1, label_y_position),
                    gemini_label,
                    color=(255, 0, 0),
                )  # 赤色でラベルを描画

                cv2.imwrite(output_sam_image_path, plot_img_sam)
                print(
                    f"SAMによるセグメンテーションとGeminiラベルの結果を {output_sam_image_path} に保存しました。"
                )
            except Exception as e:
                print(f"SAMの結果描画または保存中にエラーが発生しました: {e}")
                if plot_img_sam is None:
                    print("SAMのプロット画像が生成されませんでした。")
        else:
            print(f"{target_label}は検出されませんでした。")


if __name__ == "__main__":
    main()
