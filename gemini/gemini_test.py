import google.generativeai as genai
import os

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("エラー: 環境変数 'GEMINI_API_KEY' が設定されていません。")
    print(
        "APIキーを設定するには、ターミナルで 'export GEMINI_API_KEY=\"YOUR_API_KEY\"' を実行してください。"
    )
    exit()

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)  # 利用可能なモデル名を確認してください

response = model.generate_content("Explain how AI works in a few words")
print(response.text)
