import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 利用可能なモデルを表示
for model in genai.list_models():
    if "gemini" in model.name:
        print(f"モデル名: {model.name}, 表示名: {model.display_name}")
