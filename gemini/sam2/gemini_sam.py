import os
import cv2  # OpenCVをインポート
from google import genai
from google.genai import types
from PIL import Image
import time  # timeモジュールをインポート
import json  # jsonモジュールをインポート
from ultralytics import SAM  # SAMをインポート

# 環境変数からAPIキーを取得
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.5-flash"  # @param ["gemini-1.5-flash-latest","gemini-2.5-flash-lite-preview-06-17","gemini-2.5-flash","gemini-2.5-pro"] {"allow-input":true}

# SAMモデルのロード
# 必要に応じて、他のSAMモデルの重みファイル（例: sam_l.pt, sam_h.ptなど）を指定してください
sam_model = SAM("sam2.1_b.pt")


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
        img = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )  # SAMのためにRGBに変換
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
                    # Geminiのbox_2dは正規化されているため、元の画像サイズにスケール
                    # frame.shapeは(height, width, channels)の順
                    h, w, _ = frame.shape

                    # box_2dの値を直接使用できるよう変更 (Geminiモデルの出力に合わせて調整)
                    # モデルの出力によっては[x_min, y_min, x_max, y_max]の順番で正規化された値が返る想定
                    # したがって、直接 int(box_2d[0] * w) のように変換
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


def draw_bounding_box(frame, box, label, color=(0, 255, 0)):
    """バウンディングボックスを描画する関数"""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_segmentation_mask(frame, mask, color=(0, 0, 255), alpha=0.5):
    """セグメンテーションマスクを画像に重ねて描画する関数"""
    # マスクを3チャンネルに変換
    mask_colored = cv2.merge([mask * color[0], mask * color[1], mask * color[2]])

    # 元のフレームとマスクをブレンド
    blended_frame = cv2.addWeighted(
        frame, 1 - alpha, mask_colored.astype(frame.dtype), alpha, 0
    )
    return blended_frame


def main():
    img_path = "onigiri.png"
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Could not read image at {img_path}")
        return

    # おにぎりを検出
    found, label, bbox = detect(img, "rice_ball")

    if found:
        print(f"「{label}」を検出しました。バウンディングボックス: {bbox}")

        # オリジナルのバウンディングボックスを描画 (緑色)
        draw_bounding_box(img, bbox, label, color=(0, 255, 0))

        # SAMでセグメンテーションを実行
        # OpenCVのBGR画像をPILのRGBに変換してSAMに渡す
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        sam_results = sam_model(
            pil_img, bboxes=[bbox]
        )  # bboxesは[x1, y1, x2, y2]の形式

        # セグメンテーション結果の処理
        if sam_results and sam_results[0].masks:
            # 最初のマスクを取得（SAMは複数のマスクを返す可能性があるため）
            mask = sam_results[0].masks.data[0].cpu().numpy().astype("uint8") * 255
            print("SAMによるセグメンテーションマスクを検出しました。")

            # マスクを元の画像に重ねて描画 (赤色)
            img = draw_segmentation_mask(img, mask, color=(0, 0, 255), alpha=0.4)
        else:
            print("SAMによるセグメンテーションマスクは検出されませんでした。")

        cv2.imwrite("detected_onigiri_and_segmented.jpg", img)
        print("処理結果を detected_onigiri_and_segmented.jpg に保存しました。")
    else:
        print("おにぎりを検出できませんでした。")


if __name__ == "__main__":
    main()
