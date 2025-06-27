import google.generativeai as genai
import os
from PIL import Image
import numpy as np
import cv2
import time
import re

# あなたのGemini APIキーを設定してください
# 環境変数に設定することを推奨します: os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def analyze_image_with_3d_detection(pil_image: Image.Image):
    """
    PIL画像をGeminiで分析し、3Dバウンディングボックス（姿勢情報含む）を検出します。
    この関数は検出されたオブジェクトのリストのみを返します。
    描画は呼び出し元で行います。

    Args:
        pil_image (PIL.Image.Image): 分析するPIL形式の画像。
    Returns:
        list: パースされたオブジェクトのリスト。
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        # プロンプトの調整: 3D姿勢情報を明確に要求
        prompt_parts = [
            "この画像から物体を検出し、その2Dバウンディングボックス情報と、**最も重要として、各物体の空間的な向き（姿勢、orientation）**を、yaw, pitch, roll の度数で具体的に記載してください。3Dバウンディングボックスの中心や寸法が推定できない場合でも、可能な限り姿勢情報（rotation: yaw, pitch, roll）を提供してください。例: Object: cup, 2D Box: [10, 20, 50, 60], Rotation(yaw=45deg, pitch=10deg, roll=0deg). 出力は簡潔に、各オブジェクトの情報が1行で完結するようにしてください。"
        ]

        print("Analyzing camera frame...")
        response = model.generate_content([pil_image] + prompt_parts)

        parsed_objects = []
        if response.text:
            print("\n--- Gemini Response ---")
            print(response.text)

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
            # もし同じラベルで複数のエントリが見つかった場合、新しい情報で上書きする
            unique_objects = {}
            for obj in parsed_objects:
                label = obj.get("label", "Unknown")
                if label not in unique_objects:
                    unique_objects[label] = obj
                else:
                    unique_objects[label].update(obj)  # 情報をマージまたは更新
            parsed_objects = list(unique_objects.values())

        # パース結果の表示
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

        return parsed_objects

    except Exception as e:
        print(f"An error occurred during image analysis: {e}")
        return []  # エラー時は空のリストを返す


# --- メイン実行部分 ---
if __name__ == "__main__":
    # カメラの初期化
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Press 'q' to quit.")

    last_analysis_time = time.time()
    analysis_interval = 5  # 例: 5秒ごとにGeminiで分析

    cv2.namedWindow(
        "Camera Feed with Detection", cv2.WINDOW_NORMAL
    )  # ウィンドウをリサイズ可能にする

    # 最後に検出されたオブジェクト情報を保持する変数
    # Geminiが検出できなかった場合でも、直前の検出結果を表示し続けるため
    last_parsed_detections = []

    while True:
        ret, frame = cap.read()  # フレームを読み込む
        if not ret:
            print("Failed to grab frame.")
            break

        current_time = time.time()

        # 表示用のフレームを現在のフレームで初期化
        display_frame = frame.copy()

        # --- Geminiによる分析の実行 ---
        if current_time - last_analysis_time >= analysis_interval:
            print("\n--- Sending frame to Gemini ---")
            # OpenCVのBGR画像をPILのRGB画像に変換
            pil_img_for_gemini = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Geminiで分析し、検出情報を取得
            detections = analyze_image_with_3d_detection(pil_img_for_gemini)
            last_analysis_time = current_time

            # 検出結果を更新
            last_parsed_detections = detections

        # --- 検出結果の描画（常に実行） ---
        # Geminiに送らないフレームの場合でも、前回の検出結果を表示し続ける
        if last_parsed_detections:
            for obj in last_parsed_detections:
                # bbox_2dがない場合は画像全体を使用
                x1, y1, x2, y2 = obj.get(
                    "bbox_2d", [0, 0, frame.shape[1], frame.shape[0]]
                )
                label = obj.get("label", "Unknown")
                rotation_info = obj.get("rotation", {})

                # 2D バウンディングボックスの描画
                if "bbox_2d" in obj:
                    cv2.rectangle(
                        display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2
                    )  # 赤色の矩形, 太さ2

                # 表示テキストの準備
                display_text = f"{label}"
                if rotation_info:
                    yaw = rotation_info["yaw"]
                    pitch = rotation_info["pitch"]
                    roll = rotation_info["roll"]
                    display_text += f"\nY:{yaw:.1f}° P:{pitch:.1f}° R:{roll:.1f}°"
                else:
                    display_text += "\nNo 3D Rot."

                # テキストの描画位置を調整
                text_x = x1
                text_y = y1 - 10  # バウンディングボックスの上に表示

                # テキストが画像の上端からはみ出さないように調整
                if text_y < 0:
                    text_y = y2 + 20  # バウンディングボックスの下に表示

                # テキストを複数行で表示するための準備
                # cv2.putText は複数行テキストを直接サポートしていないため、行ごとに描画する
                lines = display_text.split("\n")
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                line_height = 25  # 各行の高さの目安

                for i, line in enumerate(lines):
                    # テキストのサイズを取得して背景ボックスを描画
                    (text_width, text_height), baseline = cv2.getTextSize(
                        line, font, font_scale, font_thickness
                    )

                    # 背景ボックスの位置計算 (テキストの左上から)
                    bg_x1 = text_x
                    bg_y1 = text_y + i * line_height - text_height - baseline
                    bg_x2 = text_x + text_width
                    bg_y2 = text_y + i * line_height + baseline

                    # 背景ボックスを描画 (半透明)
                    overlay = display_frame.copy()
                    # 描画範囲をフレーム内に限定する（オプション）
                    bg_x1 = max(0, bg_x1)
                    bg_y1 = max(0, bg_y1)
                    bg_x2 = min(frame.shape[1], bg_x2)
                    bg_y2 = min(frame.shape[0], bg_y2)

                    if bg_x1 < bg_x2 and bg_y1 < bg_y2:  # 有効な矩形であるかチェック
                        cv2.rectangle(
                            overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1
                        )  # 白色の背景
                    alpha = 0.6
                    display_frame = cv2.addWeighted(
                        overlay, alpha, display_frame, 1 - alpha, 0
                    )

                    # テキストを描画
                    cv2.putText(
                        display_frame,
                        line,
                        (text_x, text_y + i * line_height),
                        font,
                        font_scale,
                        (0, 0, 255),
                        font_thickness,
                        cv2.LINE_AA,
                    )  # 赤色のテキスト

                # === ここからが姿勢制御の概念的な部分 ===
                if "rotation" in obj:
                    label = obj.get("label", "Unknown")
                    yaw = obj["rotation"]["yaw"]
                    pitch = obj["rotation"]["pitch"]
                    roll = obj["rotation"]["roll"]

                    # ロギングは引き続き行う
                    # print(f"[{label}] 検出された姿勢情報: Yaw={yaw:.1f}°, Pitch={pitch:.1f}°, Roll={roll:.1f}°")

                    # ここに具体的な姿勢制御ロジックを記述します
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

                # --- 姿勢制御の概念的な部分ここまで ---

        # 常に更新されたdisplay_frameを表示
        cv2.imshow("Camera Feed with Detection", display_frame)

        # 'q' キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()