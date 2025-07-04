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
                    # box_2dは正規化された座標なので、画像のサイズに合わせて変換
                    # Gemini APIのbox_2dは [x_min, y_min, x_max, y_max] で、各値が0-1000の範囲
                    # これを0-1の範囲に変換し、さらに画像のピクセル値に変換
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
    input_image_path = "onigiri.png"
    output_detected_image_path = "detected_onigiri.jpg"
    output_sam_image_path = "sam_detected_onigiri.jpg"

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Gemini APIでバウンディングボックスを検出
    dets = detect(img, "rice_ball")
    detected_bbox = None

    if dets[0]:
        detected_bbox = dets[2]
        print(f"「{dets[1]}」を検出しました。バウンディングボックス: {detected_bbox}")
        # 検出結果を画像に描画して保存
        draw_bounding_box(img.copy(), detected_bbox, dets[1])  # .copy() で元のimgを保護
        cv2.imwrite(output_detected_image_path, img)
    else:
        print("おにぎりは検出されませんでした。")
        return

    # Ultralytics SAMモデルのロード
    try:
        sam_model = SAM("sam2.1_b.pt")
        # sam_model.info() # モデル情報の表示（オプション）
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return

    # 検出されたバウンディングボックスを使用してSAMで推論
    if detected_bbox:
        # SAMモデルのbboxes引数は [x1, y1, x2, y2] のリストを期待
        x1, y1, x2, y2 = detected_bbox
        # Ultralytics SAMはBGR画像を期待するため、imgをそのまま渡す
        results = sam_model(input_image_path, bboxes=[x1, y1, x2, y2])

        # 結果を画像に描画して保存
        # resultsオブジェクトにはplotメソッドがあり、検出結果を描画できる
        # SAMはセグメンテーションマスクを生成するため、それを視覚化する
        try:
            # results.plot()はNumPy配列を返す
            plot_img = results[0].plot()
            cv2.imwrite(output_sam_image_path, plot_img)
            print(f"SAMによる検出結果を {output_sam_image_path} に保存しました。")
        except Exception as e:
            print(f"SAMの結果描画または保存中にエラーが発生しました: {e}")
    else:
        print("SAMで処理するためのバウンディングボックスが見つかりませんでした。")


if __name__ == "__main__":
    main()


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


def draw_bounding_box(frame, box, label, color=(0, 255, 0)):
    """バウンディングボックスを描画する関数"""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    input_image_path = "onigiri.png"
    output_detected_image_path = "gemini_detected_onigiri.jpg"  # ファイル名を変更
    output_sam_image_path = "sam_segmented_onigiri.jpg"  # ファイル名を変更

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Gemini APIでバウンディングボックスを検出
    detected_flag, gemini_label, detected_bbox = detect(img, "rice_ball")

    if detected_flag:
        print(
            f"「{gemini_label}」を検出しました。バウンディングボックス: {detected_bbox}"
        )
        # Gemini検出結果の画像を保存
        img_gemini_drawn = img.copy()  # 元のimgを変更しないようにコピー
        draw_bounding_box(img_gemini_drawn, detected_bbox, gemini_label)
        cv2.imwrite(output_detected_image_path, img_gemini_drawn)
        print(f"Geminiによる検出結果を {output_detected_image_path} に保存しました。")
    else:
        print("おにぎりは検出されませんでした。")
        return

    # Ultralytics SAMモデルのロード
    try:
        sam_model = SAM("sam2.1_b.pt")
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return

    # 検出されたバウンディングボックスを使用してSAMで推論
    if detected_bbox:
        x1, y1, x2, y2 = detected_bbox
        results = sam_model(input_image_path, bboxes=[x1, y1, x2, y2])

        try:
            # results.plot()はNumPy配列を返す
            plot_img_sam = results[0].plot()  # SAMのデフォルトの描画

            # ここでGeminiが検出したラベルをSAMの描画結果に追加する
            # SAMのplot_img_samはBGR形式のOpenCV画像なので、draw_bounding_boxが使える
            draw_bounding_box(
                plot_img_sam, detected_bbox, gemini_label, color=(0, 0, 255)
            )  # 青色で描画

            cv2.imwrite(output_sam_image_path, plot_img_sam)
            print(
                f"SAMによるセグメンテーションとGeminiラベルの結果を {output_sam_image_path} に保存しました。"
            )
        except Exception as e:
            print(f"SAMの結果描画または保存中にエラーが発生しました: {e}")
    else:
        print("SAMで処理するためのバウンディングボックスが見つかりませんでした。")


if __name__ == "__main__":
    main()