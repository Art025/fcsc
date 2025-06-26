import google.generativeai as genai
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io
import re
import cv2  # OpenCVライブラリをインポート
import time  # 時間関連の処理のためにインポート

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

        prompt_parts = [
            "この画像から物体を検出し、それぞれの2Dおよび3Dバウンディングボックス情報を出力してください。3Dバウンディングボックスは、オブジェクトの中心座標 (x, y, z)、寸法 (width, height, depth)、および回転 (yaw, pitch, roll) を含めてください。各オブジェクトのラベルも提供してください。特に、検出された各オブジェクトの yaw, pitch, roll の値を度数で具体的に記載してください。例: Object: cup, 2D Box: [10, 20, 50, 60], 3D Box: Center(100, 200, 30), Dim(20, 30, 40), Rotation(yaw=45deg, pitch=10deg, roll=0deg)."
        ]

        print("Analyzing camera frame...")
        response = model.generate_content([pil_image] + prompt_parts)

        # --- 応答テキストの表示とパース ---
        parsed_objects = []
        if response.text:
            print("\n--- Gemini Response ---")
            print(response.text)  # Geminiからの生のテキスト応答を表示

            lines = response.text.split("\n")
            for line in lines:
                obj_info = {}
                label_match = re.search(r"Object: ([^,]+)", line)
                if label_match:
                    obj_info["label"] = label_match.group(1).strip()

                box2d_match = re.search(r"2D Box: \[(\d+), (\d+), (\d+), (\d+)\]", line)
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

                if obj_info:
                    parsed_objects.append(obj_info)

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

        # --- 視覚化 ---
        draw = ImageDraw.Draw(pil_image)
        try:
            # フォントのパスを指定するか、システムデフォルトを使用
            # Linuxの場合: /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf など
            font = ImageFont.truetype("arial.ttf", 20)  # Windows/macOS
        except IOError:
            font = ImageFont.load_default()  # デフォルトフォント

        if parsed_objects:
            for obj in parsed_objects:
                if "bbox_2d" in obj:
                    x1, y1, x2, y2 = obj["bbox_2d"]
                    label = obj.get("label", "Unknown")
                    rotation_info = obj.get("rotation", {})

                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                    display_text = f"{label}"
                    if rotation_info:
                        display_text += f"\nY:{rotation_info['yaw']:.1f}° P:{rotation_info['pitch']:.1f}° R:{rotation_info['roll']:.1f}°"

                    # テキストの背景を描画して見やすくする
                    # draw.textbbox は Pillow 9.2.0 以降で利用可能
                    try:
                        text_bbox = draw.textbbox(
                            (x1, y1 - 25), display_text, font=font
                        )
                        draw.rectangle(text_bbox, fill=(255, 255, 255, 128))
                    except AttributeError:  # textbboxがない場合のフォールバック
                        print(
                            "Pillow version might be too old for textbbox. Skipping text background."
                        )

                    draw.text((x1, y1 - 25), display_text, fill="red", font=font)

        return pil_image, parsed_objects

    except Exception as e:
        print(f"An error occurred during image analysis: {e}")
        return pil_image, []  # エラー時も元の画像を返す


# --- メイン実行部分 ---
if __name__ == "__main__":
    # カメラの初期化
    # 通常、0 はデフォルトのカメラを指します。複数のカメラがある場合は 1, 2... と試してください。
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

    while True:
        ret, frame = cap.read()  # フレームを読み込む
        if not ret:
            print("Failed to grab frame.")
            break

        # OpenCVのBGR画像をPILのRGB画像に変換
        # Gemini APIはPIL形式の画像を受け取るため、この変換が必要です
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        current_time = time.time()
        if current_time - last_analysis_time >= analysis_interval:
            print("\n--- Sending frame to Gemini ---")
            processed_pil_img, detections = analyze_image_with_3d_detection(pil_img)
            last_analysis_time = current_time

            # Matplotlibで結果を表示
            plt.imshow(processed_pil_img)
            plt.axis("off")
            plt.title("Detected Objects with Rotation Info")
            plt.show(block=False)  # ノンブロッキング表示
            plt.pause(0.1)  # UIイベント処理のための短いポーズ

            # 必要なら、OpenCVウィンドウにも検出結果を描画（ここでは簡易的に）
            # PIL画像をOpenCV形式に戻す
            processed_cv_img = cv2.cvtColor(
                cv2.UMat(
                    processed_pil_img.copy()
                ).get(),  # PIL画像をUMatに変換してからnumpy arrayに戻す
                cv2.COLOR_RGB2BGR,
            )
            cv2.imshow("Camera Feed with Detection", processed_cv_img)
        else:
            # Geminiに送らないフレームはそのまま表示
            cv2.imshow("Camera Feed with Detection", frame)

        # 'q' キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()
    plt.close("all")  # 全てのMatplotlibウィンドウを閉じる