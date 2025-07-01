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


# バウンディングボックスは描画しないので、この関数はシンプルにラベルを描画するためにのみ使用するか、
# もしくは新しいヘルパー関数を定義する
def draw_label_on_image(
    frame, position, label, color=(255, 0, 0)
):  # デフォルト色を赤に変更
    """指定された位置にラベルテキストを描画する関数"""
    cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    input_image_path = "onigiri.png"
    output_detected_image_path = (
        "gemini_detected_onigiri.jpg"  # Geminiのバウンディングボックスあり
    )
    output_sam_image_path = (
        "sam_segmented_with_gemini_label.jpg"  # SAMのセグメンテーション＋Geminiラベル
    )

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not load image from {input_image_path}")
        return

    # Gemini APIでバウンディングボックスを検出
    detected_flag, gemini_label, detected_bbox_gemini = detect(
        img, "rice_ball"
    )  # 変数名を変更

    if detected_flag:
        print(
            f"「{gemini_label}」を検出しました。Geminiバウンディングボックス: {detected_bbox_gemini}"
        )
        # Gemini検出結果の画像を保存 (赤いバウンディングボックスとラベルで)
        img_gemini_drawn = img.copy()
        # draw_bounding_boxは今回の要件では使わないが、元のコードとの比較のためにコメントアウト
        # draw_bounding_box(img_gemini_drawn, detected_bbox_gemini, gemini_label, color=(255, 0, 0)) # 赤色
        # ラベルとバウンディングボックスを描画する関数をここで直接呼び出す代わりに、
        # draw_bounding_box_and_labelなどの新しい関数を作成することも可能。
        # 今回は、SAMの結果にラベルのみ追加するため、この部分は元の通りで良い。

        # Gemini単体での検出結果（赤色のバウンディングボックスとラベル）を保存
        # draw_bounding_boxを再利用して、Geminiの検出結果を赤色で表示
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
    if detected_bbox_gemini:  # Geminiで検出されたバウンディングボックスをSAMに渡す
        x1, y1, x2, y2 = detected_bbox_gemini

        # SAMはinput_image_pathまたはimg (numpy array) のどちらでも受け取れる
        # 既にimgを読み込んでいるので、ここではimgを直接渡します
        # bboxesはSAMがセグメンテーションのプロンプトとして利用
        results = sam_model(img, bboxes=[x1, y1, x2, y2])

        try:
            # results.plot()はSAMのデフォルトの描画（青いバウンディングボックスと0 0.XX）を含むNumPy配列を返す
            plot_img_sam = results[0].plot()

            # SAMが描画したバウンディングボックスの座標を取得
            # 通常、results[0].boxes.xyxy にバウンディングボックスの座標が含まれる
            # SAMはプロンプトで与えられたバウンディングボックスに対してセグメンテーションを行うため、
            # 描画されるバウンディングボックスは元のプロンプトと一致すると期待される

            # ここでは、SAMが描画したバウンディングボックスのy座標の上部を利用して、
            # その少し上にGeminiのラベルを描画します
            # SAMのplot()で描画されたボックスの正確な位置を取得するのが最も確実ですが、
            # Geminiから得たbboxをSAMに渡しているので、そのbboxのy1座標を利用します。

            # Geminiのラベルを描画するY座標を調整 (SAMのボックスの上)
            # SAMが描画したバウンディングボックスのy1座標を基準にする
            label_y_position = y1 - 10  # バウンディングボックスの少し上に
            if label_y_position < 0:  # 画像の上端からはみ出さないように調整
                label_y_position = 15

            # Geminiのラベルを描画 (バウンディングボックスは描画しない)
            draw_label_on_image(
                plot_img_sam, (x1, label_y_position), gemini_label, color=(255, 0, 0)
            )  # 赤色でラベルを描画

            cv2.imwrite(output_sam_image_path, plot_img_sam)
            print(
                f"SAMによるセグメンテーションとGeminiラベルの結果を {output_sam_image_path} に保存しました。"
            )
        except Exception as e:
            print(f"SAMの結果描画または保存中にエラーが発生しました: {e}")
            # エラー時にSAMの描画ができなかった場合でも、Geminiラベルを描画する試みを続けるためのエラーハンドリング
            # 例えば、SAMが結果を返さなかった場合、plot_img_sam がNoneになる可能性も考慮する
            if plot_img_sam is None:
                print("SAMのプロット画像が生成されませんでした。")
    else:
        print("SAMで処理するためのバウンディングボックスが見つかりませんでした。")


if __name__ == "__main__":
    main()