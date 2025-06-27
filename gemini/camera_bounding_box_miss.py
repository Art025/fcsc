import google.generativeai as genai
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np  # numpyをインポート
import cv2  # OpenCVライブラリをインポート
import re

# あなたのGemini APIキーを設定してください
# 環境変数に設定することを推奨します: os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def analyze_image_with_3d_detection(pil_image: Image.Image):
    """
    PIL画像をGeminiで分析し、3Dバウンディングボックス（姿勢情報含む）を検出します。

    Args:
        pil_image (PIL.Image.Image): 分析するPIL形式の画像。
    Returns:
        tuple: (PIL.Image.Image, list) 検出結果が描画された画像と、パースされたオブジェクトのリスト。
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        # === プロンプトの調整 ===
        # 3Dバウンディングボックス全体が難しい場合でも、姿勢情報（yaw, pitch, roll）を強調
        prompt_parts = [
            "この画像から物体を検出し、その2Dバウンディングボックス情報と、**最も重要として、各物体の空間的な向き（姿勢、orientation）**を、yaw, pitch, roll の度数で具体的に記載してください。3Dバウンディングボックスの中心や寸法が推定できない場合でも、可能な限り姿勢情報（rotation: yaw, pitch, roll）を提供してください。例: Object: cup, 2D Box: [10, 20, 50, 60], Rotation(yaw=45deg, pitch=10deg, roll=0deg). 出力は簡潔に、各オブジェクトの情報が1行で完結するようにしてください。"
        ]

        print("Analyzing single camera frame...")
        response = model.generate_content([pil_image] + prompt_parts)

        # --- 応答テキストの表示とパース ---
        parsed_objects = []
        if response.text:
            print("\n--- Gemini Response ---")
            print(response.text)  # Geminiからの生のテキスト応答を表示

            lines = response.text.split("\n")
            for line in lines:
                obj_info = {}
                # まず Object: の形式を試す
                label_match = re.search(r"Object: ([^,]+)", line)
                if label_match:
                    obj_info["label"] = label_match.group(1).strip()
                else:  # Label: の形式を試す
                    label_match = re.search(r"Label: ([^,]+)", line)
                    if label_match:
                        obj_info["label"] = label_match.group(1).strip()

                # 2D Boxの正規表現をより柔軟に、括弧や追加のテキストを許容するように変更
                box2d_match = re.search(
                    r"2D Bounding Box.*?:?\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", line
                )
                if box2d_match:
                    obj_info["bbox_2d"] = [
                        int(box2d_match.group(i)) for i in range(1, 5)
                    ]

                rotation_match = re.search(
                    r"Rotation\(yaw=(-?\d+\.?\d*)deg, pitch=(-?\d+\.?\d*)deg, roll=(-?\d+\.?\d*)deg\)",
                    line,
                )
                if rotation_match:
                    obj_info["rotation"] = {
                        "yaw": float(rotation_match.group(1)),
                        "pitch": float(rotation_match.group(2)),
                        "roll": float(rotation_match.group(3)),
                    }
                # 3D CenterとDimは、Geminiが提供しない可能性が高いため、ここではオプションとして扱う
                center_match = re.search(r"Center\((\d+), (\d+), (\d+)\)", line)
                if center_match:
                    obj_info["center_3d"] = [
                        int(center_match.group(i)) for i in range(1, 4)
                    ]

                dim_match = re.search(r"Dim\((\d+), (\d+), (\d+)\)", line)
                if dim_match:
                    obj_info["dim_3d"] = [int(dim_match.group(i)) for i in range(1, 4)]

                if (
                    obj_info.get("label")
                    or obj_info.get("bbox_2d")
                    or obj_info.get("rotation")
                ):  # いずれかの情報があれば追加
                    parsed_objects.append(obj_info)

            # 重複エントリを防ぐため、検出されたオブジェクトを整理（同じラベルで複数の行がある場合など）
            unique_objects = {}
            for obj in parsed_objects:
                label = obj.get("label", "Unknown")
                if label not in unique_objects:
                    unique_objects[label] = obj
                else:
                    unique_objects[label].update(obj)  # 情報をマージまたは更新
            parsed_objects = list(unique_objects.values())

        # --- パース結果の表示 ---
        print("\n--- Parsed Object Data ---")
        if parsed_objects:
            for obj in parsed_objects:
                print(f"Object: {obj.get('label', 'N/A')}")
                if "bbox_2d" in obj:
                    print(f"  2D Bounding Box: {obj['bbox_2d']}")
                if "rotation" in obj:
                    print(f"  3D Rotation (Yaw, Pitch, Roll): {obj['rotation']}")
                if "center_3d" in obj:
                    print(f"  3D Center: {obj['center_3d']}")
                if "dim_3d" in obj:
                    print(f"  3D Dimensions: {obj['dim_3d']}")
        else:
            print("No structured object data could be parsed from the response.")

        # PIL画像への描画
        draw = ImageDraw.Draw(pil_image)
        try:
            # フォントのパスを適切に設定（Linuxの場合など）
            font = ImageFont.truetype("arial.ttf", 20)  # Windows/macOS
        except IOError:
            print("Font 'arial.ttf' not found, using default font.")
            font = ImageFont.load_default()

        if parsed_objects:
            for obj in parsed_objects:
                x1, y1, x2, y2 = obj.get(
                    "bbox_2d", [0, 0, pil_image.width, pil_image.height]
                )  # bbox_2dがない場合は画像全体を使用
                label = obj.get("label", "Unknown")
                rotation_info = obj.get("rotation", {})

                # バウンディングボックスの描画（bbox_2d があれば）
                if "bbox_2d" in obj:
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                display_text = f"{label}"
                if rotation_info:
                    display_text += f"\nY:{rotation_info['yaw']:.1f}° P:{rotation_info['pitch']:.1f}° R:{rotation_info['roll']:.1f}°"
                else:
                    display_text += "\nNo 3D Rot."  # 3D回転情報がない場合

                # テキストの描画位置を調整（バウンディングボックスがなくても表示できるよう）
                text_x = x1 if "bbox_2d" in obj else 10
                text_y = y1 - 25 if "bbox_2d" in obj else 10
                text_y = max(0, text_y)  # 画面上端を超えないように

                # テキストの背景を描画して見やすくする
                try:
                    text_bbox = draw.textbbox((text_x, text_y), display_text, font=font)
                    draw.rectangle(text_bbox, fill=(255, 255, 255, 128))
                except AttributeError:
                    print(
                        "Pillow version might be too old for textbbox. Skipping text background."
                    )

                draw.text((text_x, text_y), display_text, fill="red", font=font)

        return pil_image, parsed_objects

    except Exception as e:
        print(f"An error occurred during image analysis: {e}")
        return pil_image, []  # エラー時も元の画像を返す


# --- メイン実行部分 ---
if __name__ == "__main__":
    # カメラの初期化
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Taking a single picture for analysis...")

    # カメラから1フレームだけ読み込む
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    # OpenCVのBGR画像をPILのRGB画像に変換
    pil_img_for_gemini = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Geminiで分析し、検出結果が描画されたPIL画像と検出情報を取得
    processed_pil_img, detections = analyze_image_with_3d_detection(pil_img_for_gemini)

    # 処理されたPIL画像をNumpy配列（OpenCV形式）に変換
    display_cv_img = cv2.cvtColor(np.array(processed_pil_img), cv2.COLOR_RGB2BGR)

    # === ここからが姿勢制御の概念的な部分（一度だけ実行） ===
    if detections:
        print("\n--- Detected Object Pose Information ---")
        for obj in detections:
            if "rotation" in obj:
                label = obj.get("label", "Unknown")
                yaw = obj["rotation"]["yaw"]
                pitch = obj["rotation"]["pitch"]
                roll = obj["rotation"]["roll"]

                print(
                    f"[{label}] 姿勢情報: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°, Roll={roll:.1f}°"
                )

                # ここに具体的な姿勢制御ロジックを記述します（一度だけ実行）
                # 例: ロボットの関節角度を調整する、仮想環境のオブジェクトを回転させるなど
                # 注意: これらの値はカメラからの相対的な見え方に基づくため、正確なロボット制御には追加のキャリブレーションや深度情報が必要になることが多いです。

                if abs(yaw) > 30:
                    print(
                        f"  -> {label} のヨー角が大きい (Y:{yaw:.1f}°)。調整を検討してください。"
                    )
                if abs(pitch) > 20:
                    print(
                        f"  -> {label} のピッチ角が大きい (P:{pitch:.1f}°)。調整を検討してください。"
                    )
                # ロール角も同様にチェック可能
            else:
                print(
                    f"[{obj.get('label', 'Unknown')}] 回転情報が見つかりませんでした。"
                )
    else:
        print("オブジェクトが検出されませんでした。")

    # --- 姿勢制御の概念的な部分ここまで ---

    # 検出結果が描画された画像をOpenCVウィンドウに表示
    cv2.imshow("Detection Result", display_cv_img)
    print("\nPress any key on the 'Detection Result' window to close it.")

    # キーが押されるまでウィンドウを表示し続ける
    cv2.waitKey(0)

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()
