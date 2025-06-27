import google.generativeai as genai
import os
from PIL import Image
import numpy as np
import cv2
import time
import re

# Gemini APIキーを設定してください。環境変数に設定することを推奨します。
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def analyze_image_with_3d_detection(pil_image: Image.Image):
    """
    PIL画像をGeminiで分析し、オブジェクトの2Dバウンディングボックスと姿勢情報を検出します。
    この関数は検出されたオブジェクトのリストのみを返します。
    描画は呼び出し元で行います。

    Args:
        pil_image (PIL.Image.Image): 分析するPIL形式の画像。
    Returns:
        list: パースされたオブジェクトのリスト。
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

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
                label_match = re.search(r"Object: ([^,]+)", line)
                if label_match:
                    obj_info["label"] = label_match.group(1).strip()
                else:
                    label_match = re.search(r"Label: ([^,]+)", line)
                    if label_match:
                        obj_info["label"] = label_match.group(1).strip()

                box2d_match = re.search(
                    r"(?:2D Bounding Box|2D Box).*?:?\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]",
                    line,
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
                ):
                    parsed_objects.append(obj_info)

            unique_objects = {}
            for obj in parsed_objects:
                label = obj.get("label", "Unknown")
                if label not in unique_objects:
                    unique_objects[label] = obj
                else:
                    unique_objects[label].update(obj)
            parsed_objects = list(unique_objects.values())

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
        return []


# --- メイン実行部分 ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("Press 'q' to quit. Press 'r' to resume camera if paused.")

    last_analysis_time = time.time()
    analysis_interval = 5  # 例: 5秒ごとにGeminiで分析

    cv2.namedWindow("Camera Feed with Detection", cv2.WINDOW_NORMAL)

    # 最後に検出されたオブジェクト情報を保持する変数
    last_parsed_detections = []
    # カメラを一時停止するかどうかのフラグ
    pause_camera_feed = False
    # おにぎりが検出されて一時停止中かどうかのフラグ
    onigiri_detected_and_paused = False
    # 一時停止時におにぎり情報を保持する変数
    paused_onigiri_info = []
    # 最後に読み込んだフレームを保持 (一時停止時に表示するため)
    last_frame = None

    while True:
        if not pause_camera_feed:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            last_frame = frame.copy()  # 新しいフレームを取得したら保存

        # 表示用のフレームを現在のフレーム（または一時停止中の最終フレーム）で初期化
        display_frame = (
            last_frame.copy()
            if last_frame is not None
            else np.zeros((480, 640, 3), dtype=np.uint8)
        )

        current_time = time.time()

        # --- Geminiによる分析の実行 ---
        # カメラが一時停止中でない場合のみGemini分析を実行
        if not pause_camera_feed and (
            current_time - last_analysis_time >= analysis_interval
        ):
            print("\n--- Sending frame to Gemini ---")
            pil_img_for_gemini = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            detections = analyze_image_with_3d_detection(pil_img_for_gemini)
            last_analysis_time = current_time

            last_parsed_detections = detections

            # おにぎり検出チェック
            found_onigiri = False
            onigiri_to_highlight = []
            for obj in last_parsed_detections:
                # "onigiri" または "rice ball" を検出ラベルに含むかチェック
                if "label" in obj and (
                    "onigiri" in obj["label"].lower()
                    or "rice ball" in obj["label"].lower()
                ):
                    found_onigiri = True
                    onigiri_to_highlight.append(obj)

            if found_onigiri:
                print("おにぎりを検出しました！カメラを一時停止します。")
                pause_camera_feed = True
                onigiri_detected_and_paused = True
                paused_onigiri_info = (
                    onigiri_to_highlight  # 検出されたおにぎり情報を保存
                )

        # --- 検出結果の描画 ---
        if onigiri_detected_and_paused:
            # おにぎり検出による一時停止中の場合、検出されたおにぎりだけを描画
            for obj in paused_onigiri_info:
                x1, y1, x2, y2 = obj.get(
                    "bbox_2d", [0, 0, display_frame.shape[1], display_frame.shape[0]]
                )
                label = obj.get("label", "Unknown")
                rotation_info = obj.get("rotation", {})

                # バウンディングボックスを強調して描画 (例: 緑色で太く)
                if "bbox_2d" in obj:
                    cv2.rectangle(
                        display_frame, (x1, y1), (x2, y2), (0, 255, 0), 4
                    )  # 緑色, 太さ4

                # テキストの準備
                display_text = f"{label}"
                if rotation_info:
                    yaw = rotation_info["yaw"]
                    pitch = rotation_info["pitch"]
                    roll = rotation_info["roll"]
                    display_text += f"\nY:{yaw:.1f}° P:{pitch:.1f}° R:{roll:.1f}°"
                else:
                    display_text += "\nNo 3D Rot."
                display_text += (
                    "\n[Paused] Press 'r' to resume."  # 一時停止中のメッセージ
                )

                text_x = x1
                text_y = y1 - 10
                if text_y < 0:
                    text_y = y2 + 20

                lines = display_text.split("\n")
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                line_height = 25

                for i, line in enumerate(lines):
                    (text_width, text_height), baseline = cv2.getTextSize(
                        line, font, font_scale, font_thickness
                    )
                    bg_x1 = text_x
                    bg_y1 = text_y + i * line_height - text_height - baseline
                    bg_x2 = text_x + text_width
                    bg_y2 = text_y + i * line_height + baseline

                    overlay = display_frame.copy()
                    bg_x1 = max(0, bg_x1)
                    bg_y1 = max(0, bg_y1)
                    bg_x2 = min(display_frame.shape[1], bg_x2)
                    bg_y2 = min(display_frame.shape[0], bg_y2)

                    if bg_x1 < bg_x2 and bg_y1 < bg_y2:
                        cv2.rectangle(
                            overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1
                        )
                    alpha = 0.6
                    display_frame = cv2.addWeighted(
                        overlay, alpha, display_frame, 1 - alpha, 0
                    )
                    cv2.putText(
                        display_frame,
                        line,
                        (text_x, text_y + i * line_height),
                        font,
                        font_scale,
                        (0, 255, 0),
                        font_thickness,
                        cv2.LINE_AA,
                    )  # 緑色のテキスト

        elif last_parsed_detections:
            # 通常モードの場合、前回の検出結果を全て描画
            for obj in last_parsed_detections:
                x1, y1, x2, y2 = obj.get(
                    "bbox_2d", [0, 0, display_frame.shape[1], display_frame.shape[0]]
                )
                label = obj.get("label", "Unknown")
                rotation_info = obj.get("rotation", {})

                if "bbox_2d" in obj:
                    cv2.rectangle(
                        display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2
                    )  # 赤色の矩形

                display_text = f"{label}"
                if rotation_info:
                    yaw = rotation_info["yaw"]
                    pitch = rotation_info["pitch"]
                    roll = rotation_info["roll"]
                    display_text += f"\nY:{yaw:.1f}° P:{pitch:.1f}° R:{roll:.1f}°"
                else:
                    display_text += "\nNo 3D Rot."

                text_x = x1
                text_y = y1 - 10
                if text_y < 0:
                    text_y = y2 + 20

                lines = display_text.split("\n")
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                line_height = 25

                for i, line in enumerate(lines):
                    (text_width, text_height), baseline = cv2.getTextSize(
                        line, font, font_scale, font_thickness
                    )
                    bg_x1 = text_x
                    bg_y1 = text_y + i * line_height - text_height - baseline
                    bg_x2 = text_x + text_width
                    bg_y2 = text_y + i * line_height + baseline

                    overlay = display_frame.copy()
                    bg_x1 = max(0, bg_x1)
                    bg_y1 = max(0, bg_y1)
                    bg_x2 = min(display_frame.shape[1], bg_x2)
                    bg_y2 = min(display_frame.shape[0], bg_y2)

                    if bg_x1 < bg_x2 and bg_y1 < bg_y2:
                        cv2.rectangle(
                            overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1
                        )
                    alpha = 0.6
                    display_frame = cv2.addWeighted(
                        overlay, alpha, display_frame, 1 - alpha, 0
                    )
                    cv2.putText(
                        display_frame,
                        line,
                        (text_x, text_y + i * line_height),
                        font,
                        font_scale,
                        (0, 0, 255),
                        font_thickness,
                        cv2.LINE_AA,
                    )

        cv2.imshow("Camera Feed with Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        # 'q' キーが押されたら終了
        if key == ord("q"):
            break
        # 'r' キーが押され、一時停止中であれば再開
        elif key == ord("r") and pause_camera_feed:
            print("カメラを再開します。")
            pause_camera_feed = False
            onigiri_detected_and_paused = False
            paused_onigiri_info = []  # 保存していたおにぎり情報をクリア

    cap.release()
    cv2.destroyAllWindows()
