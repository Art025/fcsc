import google.generativeai as genai
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import io

# あなたのGemini APIキーを設定してください
# 環境変数に設定することを推奨します: os.environ.get("GEMINI_API_KEY")
# 例: os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def analyze_image_with_3d_detection(image_path: str):
    """
    指定された画像をGemini 2.5で分析し、3Dバウンディングボックスを検出します。

    Args:
        image_path (str): 分析する画像のパス。
    """
    try:
        # モデルの初期化 (Gemini 1.5 Flash または Gemini 1.5 Pro を推奨)
        # 3D検出はGemini 1.5 Proの方が良い結果を出す可能性がありますが、Flashはより高速です。
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        # または model = genai.GenerativeModel('gemini-1.5-pro-latest')

        # 画像の読み込み
        img = Image.open(image_path)

        # プロンプトの準備
        # 3D検出を明示的に要求するプロンプトを記述します。
        prompt_parts = [
            "この画像から物体を検出し、それぞれの2Dおよび3Dバウンディングボックス情報を出力してください。3Dバウンディングボックスは、オブジェクトの中心座標 (x, y, z)、寸法 (width, height, depth)、および回転 (yaw, pitch, roll) を含めてください。各オブジェクトのラベルも提供してください。"
        ]

        # コンテンツの生成
        print(f"Analyzing {image_path}...")
        response = model.generate_content([img] + prompt_parts)

        # 結果の表示
        print("\n--- Detection Results ---")
        if response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "parts"):
                    for part in candidate.parts:
                        # safety_settingsでブロックされていないか確認
                        if part.text:
                            print(part.text)
                        # 3D検出結果は通常、Structured dataまたはCode Blockとして返されます。
                        # ここでは簡略化のためにtext部分を出力していますが、
                        # 実際のアプリケーションではJSONや構造化されたデータをパースする必要があります。
                        # 詳細な3Dバウンディングボックス情報は、
                        # `response.candidates[0].function_call`や
                        # `response.candidates[0].text`内の特定のフォーマットで返されます。
                        # 必要に応じて、responseオブジェクトの構造を確認してください。
                if candidate.finish_reason == "SAFETY":
                    print("Content potentially blocked by safety settings.")

            # --- 視覚化の試み (簡略版) ---
            # ここでは、Geminiのレスポンスからバウンディングボックス情報を「パースする」必要があります。
            # レスポンスの形式は変動する可能性があるため、これはあくまで例です。
            # 実際のアプリケーションでは、より堅牢なパースロジックが必要です。

            # 簡単な描画のための準備
            draw = ImageDraw.Draw(img)
            # フォントの読み込み (システムにインストールされているフォントを使用)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            if response.text:
                # レスポンステキストからバウンディングボックス情報を抽出し、描画する (非常に簡略化)
                # 例えば、"Object: chair, 2D Box: [x1, y1, x2, y2]" のような形式を想定
                lines = response.text.split("\n")
                for line in lines:
                    if "2D Box:" in line:
                        try:
                            # 例: "Object: chair, 2D Box: [100, 50, 200, 150]"
                            parts = (
                                line.split("2D Box: [")[1].replace("]", "").split(", ")
                            )
                            x1, y1, x2, y2 = map(int, parts)
                            label = line.split(", 2D Box:")[0].replace("Object: ", "")

                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            draw.text((x1, y1 - 25), label, fill="red", font=font)
                        except Exception as e:
                            print(f"Could not parse 2D box from line: {line} - {e}")

            plt.imshow(img)
            plt.axis("off")
            plt.title("Detected Objects (2D Boxes - Simplified)")
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
    # ここに分析したい画像のパスを指定してください
    # 例: current_directory/my_image.jpg
    # または、手元に画像がない場合は、簡単なテスト画像を生成してみましょう。

    # テスト画像を作成 (必要に応じてコメントアウトして実際の画像パスを設定)
    test_image_path = "test_image.png"
    if not os.path.exists(test_image_path):
        dummy_img = Image.new("RGB", (600, 400), color="white")
        d = ImageDraw.Draw(dummy_img)
        d.ellipse((50, 50, 200, 200), fill="red", outline="black")
        d.rectangle((300, 100, 500, 300), fill="blue", outline="black")
        d.text((10, 10), "This is a test image.", fill=(0, 0, 0))
        dummy_img.save(test_image_path)
        print(f"Created a dummy image: {test_image_path}")

    # 画像を分析
    analyze_image_with_3d_detection(test_image_path)

    # 別の画像を試す場合は、以下のコメントを解除してパスを修正してください。
    # analyze_image_with_3d_detection("path/to/your/another_image.jpg")
