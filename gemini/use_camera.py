import google.generativeai as genai
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io
import re  # 正規表現モジュールを追加

# あなたのGemini APIキーを設定してください
# 環境変数に設定することを推奨します: os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def analyze_image_with_3d_detection(image_path: str):
    """
    指定された画像をGeminiで分析し、3Dバウンディングボックス（姿勢情報含む）を検出します。

    Args:
        image_path (str): 分析する画像のパス。
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        img = Image.open(image_path)

        prompt_parts = [
            "この画像から物体を検出し、それぞれの2Dおよび3Dバウンディングボックス情報を出力してください。3Dバウンディングボックスは、オブジェクトの中心座標 (x, y, z)、寸法 (width, height, depth)、および回転 (yaw, pitch, roll) を含めてください。各オブジェクトのラベルも提供してください。特に、検出された各オブジェクトの yaw, pitch, roll の値を度数で具体的に記載してください。例: Object: cup, 2D Box: [10, 20, 50, 60], 3D Box: Center(100, 200, 30), Dim(20, 30, 40), Rotation(yaw=45deg, pitch=10deg, roll=0deg)."
        ]

        print(f"Analyzing {image_path}...")
        response = model.generate_content([img] + prompt_parts)

        print("\n--- Detection Results ---")
        if response.candidates:
            # --- 応答テキストの表示とパース ---
            parsed_objects = []
            if response.text:
                print(response.text)  # Geminiからの生のテキスト応答を表示

                lines = response.text.split("\n")
                for line in lines:
                    obj_info = {}
                    # Object Label
                    label_match = re.search(r"Object: ([^,]+)", line)
                    if label_match:
                        obj_info["label"] = label_match.group(1).strip()

                    # 2D Box Parsing
                    box2d_match = re.search(
                        r"2D Box: \[(\d+), (\d+), (\d+), (\d+)\]", line
                    )
                    if box2d_match:
                        obj_info["bbox_2d"] = [
                            int(box2d_match.group(i)) for i in range(1, 5)
                        ]

                    # 3D Rotation Parsing (yaw, pitch, roll)
                    # "Rotation(yaw=XXdeg, pitch=YYdeg, roll=ZZdeg)" のパターンを想定
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

                    # 3D Center Parsing (x, y, z)
                    center_match = re.search(r"Center\((\d+), (\d+), (\d+)\)", line)
                    if center_match:
                        obj_info["center_3d"] = [
                            int(center_match.group(i)) for i in range(1, 4)
                        ]

                    # 3D Dimension Parsing (width, height, depth)
                    dim_match = re.search(r"Dim\((\d+), (\d+), (\d+)\)", line)
                    if dim_match:
                        obj_info["dim_3d"] = [
                            int(dim_match.group(i)) for i in range(1, 4)
                        ]

                    if obj_info:  # 何らかの情報がパースできたら追加
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
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            if parsed_objects:
                for obj in parsed_objects:
                    if "bbox_2d" in obj:
                        x1, y1, x2, y2 = obj["bbox_2d"]
                        label = obj.get("label", "Unknown")
                        rotation_info = obj.get("rotation", {})

                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                        # ラベルと回転情報を2Dボックスの上に表示
                        display_text = f"{label}"
                        if rotation_info:
                            display_text += f"\nY:{rotation_info['yaw']:.1f}° P:{rotation_info['pitch']:.1f}° R:{rotation_info['roll']:.1f}°"

                        # テキストの背景を描画して見やすくする (オプション)
                        text_bbox = draw.textbbox(
                            (x1, y1 - 25), display_text, font=font
                        )
                        draw.rectangle(
                            text_bbox, fill=(255, 255, 255, 128)
                        )  # 半透明の白背景
                        draw.text((x1, y1 - 25), display_text, fill="red", font=font)

            plt.imshow(img)
            plt.axis("off")
            plt.title("Detected Objects with Rotation Info")
            plt.show()

        else:
            print("No candidates found in the response.")
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(
                    f"Prompt was blocked due to: {response.prompt_feedback.block_reason}"
                )
                print(f"Safety ratings: {response.prompt_feedback.safety_ratings}")

    except Exception as e:
        print(f"An error occurred: {e}")


# --- 実行例 ---
if __name__ == "__main__":
    test_image_path = "物体検出元画像.png"  # あなたの画像パスを指定してください

    # テスト画像を作成 (必要に応じてコメントアウトして実際の画像パスを設定)
    if not os.path.exists(test_image_path):
        dummy_img = Image.new("RGB", (600, 400), color="white")
        d = ImageDraw.Draw(dummy_img)
        # 立方体を模した図形（奥行きを少し意識）
        d.polygon(
            [(100, 100), (200, 100), (200, 200), (100, 200)],
            fill="lightgray",
            outline="black",
        )  # 手前面
        d.line([(100, 100), (120, 80)], fill="black")  # 奥行き線
        d.line([(200, 100), (220, 80)], fill="black")
        d.line([(200, 200), (220, 180)], fill="black")
        d.line([(100, 200), (120, 180)], fill="black")
        d.polygon(
            [(120, 80), (220, 80), (220, 180), (120, 180)], outline="black"
        )  # 奥面
        d.text((150, 150), "Box", fill=(0, 0, 0), font=ImageFont.load_default())

        # 円筒を模した図形（少し傾ける）
        d.ellipse((350, 150, 450, 200), fill="skyblue", outline="black")  # 上の楕円
        d.line([(350, 175), (350, 250)], fill="black")  # 左側面
        d.line([(450, 175), (450, 250)], fill="black")  # 右側面
        d.ellipse((350, 225, 450, 275), fill="skyblue", outline="black")  # 下の楕円
        d.text((380, 200), "Cylinder", fill=(0, 0, 0), font=ImageFont.load_default())

        dummy_img.save(test_image_path)
        print(f"Created a dummy image: {test_image_path}")

    # 画像を分析
    analyze_image_with_3d_detection(test_image_path)
