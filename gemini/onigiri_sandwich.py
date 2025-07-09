import os
import json
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont  # ImageDrawとImageFontをインポート
import IPython  # 画像表示のためにIPythonをインポート

# 環境変数からAPIキーを取得
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY 環境変数が設定されていません。")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.5-flash"

# 画像ファイルのパス
image_path = "onigirisand.jpg"

# 画像を読み込む
try:
    img = Image.open(image_path)
except FileNotFoundError:
    print(
        f"エラー: '{image_path}' が見つかりません。画像ファイルがスクリプトと同じディレクトリにあるか確認してください。"
    )
    exit()

# Geminiモデルを使用して画像を分析
# プロンプトを修正し、2Dバウンディングボックスを [x_min, y_min, x_max, y_max] 形式で要求します。
image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        img,
        """
          Detect the bounding boxes of the onigiri and the sandwich.
          Output a json list where each entry contains the object name in "label" and its 2D bounding box in "box_2d".
          The 2D bounding box format should be [x_min, y_min, x_max, y_max].
        """,
    ],
    config=types.GenerateContentConfig(temperature=0.5),
)

# Gemini APIからの応答を表示
json_output_str = image_response.text
print("Gemini API Response:")
print(json_output_str)

# JSON出力を解析
try:
    detections = json.loads(json_output_str)
except json.JSONDecodeError as e:
    print(f"JSONデコードエラー: {e}")
    print("JSON文字列の整形を試みます...")
    # モデルの出力が完全にJSON形式でない場合があるため、開始と終了のブラケットを探して整形を試みる
    start_idx = json_output_str.find("[")
    end_idx = json_output_str.rfind("]")
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        cleaned_json_str = json_output_str[start_idx : end_idx + 1]
        try:
            detections = json.loads(cleaned_json_str)
            print("整形されたJSONの再解析に成功しました。")
        except json.JSONDecodeError as e_cleaned:
            print(f"整形後も解析に失敗しました: {e_cleaned}")
            detections = []  # 失敗した場合は空のリストにフォールバック
    else:
        detections = []  # 失敗した場合は空のリストにフォールバック

# 描画オブジェクトを作成
draw = ImageDraw.Draw(img)

# バウンディングボックスの色を定義
colors = ["red", "green", "blue", "purple", "orange"]  # 色を循環して使用

# ラベル用のフォントを読み込む
try:
    # システムにarial.ttfがない場合があるため、一般的なフォントを試すか、デフォルトを使用
    font = ImageFont.truetype("arial.ttf", 25)  # フォントサイズを調整可能
except IOError:
    font = ImageFont.load_default()
    print("arial.ttfを読み込めませんでした。デフォルトフォントを使用します。")

# バウンディングボックスとラベルを描画
for i, detection in enumerate(detections):
    label = detection.get("label", "Unknown")
    # プロンプトで "box_2d" を要求しているため、そのキーを探します。
    # もしモデルが "_2d" のような別のキーを返す場合は、ここを修正する必要があります。
    bbox = detection.get("box_2d")

    if bbox and len(bbox) == 4:
        # バウンディングボックスの形式を [x_min, y_min, x_max, y_max] と仮定
        x_min, y_min, x_max, y_max = bbox
        color = colors[i % len(colors)]

        # 四角形を描画
        draw.rectangle(
            [x_min, y_min, x_max, y_max], outline=color, width=4
        )  # 線の太さを調整

        # ラベルを描画
        # テキストをボックスの左上隅の少し上に配置
        text_x = x_min
        text_y = y_min - 30  # 垂直方向の位置を調整
        if text_y < 0:  # テキストが画像の上端からはみ出さないように調整
            text_y = y_min + 5

        draw.text((text_x, text_y), label, fill=color, font=font)
    else:
        print(
            f"不正な形式のバウンディングボックスをスキップしました: ラベル={label}, bbox={bbox}"
        )

# バウンディングボックスが描画された画像を表示
# Jupyter Notebookのような環境で実行すると、画像が直接表示されます。
# スクリプトとして実行すると、システムのデフォルト画像ビューアで画像が開きます。
img.show()
