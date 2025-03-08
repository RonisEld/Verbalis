import google.generativeai as genai
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# プロジェクトのルートディレクトリを取得
root_dir = Path(__file__).parent.parent

# configurationディレクトリの.envファイルを読み込む
dotenv_path = os.path.join(root_dir, 'configuration', '.env')
load_dotenv(dotenv_path)

# APIキーが取得できたか確認
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("エラー: GEMINI_API_KEYが見つかりません。.envファイルを確認してください。")
    sys.exit(1)

print(f"APIキーが読み込まれました: {api_key[:5]}...")
genai.configure(api_key=api_key)

# 利用可能なモデルを表示
for model in genai.list_models():
    if "gemini" in model.name:
        print(f"モデル名: {model.name}, 表示名: {model.display_name}")
