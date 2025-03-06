"""
Voice Chat - Gemini APIとStyle-BERT-VITS2を組み合わせたチャットボット

このモジュールはGoogle Gemini 2.0 Flash APIを使用してテキスト応答を生成し、
Style-BERT-VITS2 APIを使用して音声に変換するコマンドラインチャットボットを提供します。
"""

import os
import sys
import json
import asyncio
import requests
import tempfile
import subprocess
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any

# .envファイルから環境変数を読み込む
load_dotenv()

# Gemini API設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("エラー: GEMINI_API_KEYが設定されていません。.envファイルを確認してください。")
    sys.exit(1)

# Style-BERT-VITS2 API設定
SBV2_API_URL = "http://localhost:8000"
DEFAULT_SPEAKER_ID = int(os.getenv("DEFAULT_SPEAKER_ID", "0"))
DEFAULT_MODEL_ID = int(os.getenv("DEFAULT_MODEL_ID", "0"))

# Gemini APIの設定
genai.configure(api_key=GEMINI_API_KEY)

class VoiceChat:
    """
    Gemini APIとStyle-BERT-VITS2を組み合わせたチャットボットクラス
    """
    
    def __init__(self):
        """
        チャットボットの初期化
        """
        # Geminiモデルの設定
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 2048,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # Gemini 2.0 Flash モデルを使用
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # チャット履歴の初期化
        self.chat_session = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": ["こんにちは。あなたは日本語で会話するAIアシスタントです。"]
                },
                {
                    "role": "model",
                    "parts": ["こんにちは！私は日本語で会話するAIアシスタントです。どのようにお手伝いできますか？"]
                }
            ]
        )
        
        # 利用可能なモデルを取得
        self.available_models = self._get_available_models()
        
        # 音声設定
        self.speaker_id = DEFAULT_SPEAKER_ID
        self.model_id = DEFAULT_MODEL_ID
        
        print(f"利用可能な音声モデル:")
        for model in self.available_models.get("models", []):
            print(f"  ID: {model['id']} - 名前: {model['name']}")
        print(f"デフォルトモデルID: {self.available_models.get('default_model_id', 0)}")
        print(f"現在選択中のモデルID: {self.model_id}")
        print(f"現在選択中の話者ID: {self.speaker_id}")
        print("=" * 50)
    
    def _get_available_models(self) -> Dict:
        """
        利用可能な音声モデルを取得する
        
        Returns:
            Dict: 利用可能なモデルの情報
        """
        try:
            response = requests.get(f"{SBV2_API_URL}/models")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"エラー: 音声モデルの取得に失敗しました。Style-BERT-VITS2 APIサーバーが起動しているか確認してください。")
            print(f"詳細: {str(e)}")
            return {"models": [], "default_model_id": 0}
    
    async def generate_response(self, user_input: str) -> str:
        """
        ユーザー入力に対する応答を生成する
        
        Args:
            user_input: ユーザーの入力テキスト
            
        Returns:
            str: Gemini APIからの応答テキスト
        """
        try:
            response = await asyncio.to_thread(
                self.chat_session.send_message,
                user_input
            )
            return response.text
        except Exception as e:
            print(f"エラー: Gemini APIからの応答生成に失敗しました。")
            print(f"詳細: {str(e)}")
            return "申し訳ありません。応答の生成中にエラーが発生しました。"
    
    async def text_to_speech(self, text: str) -> Optional[bytes]:
        """
        テキストを音声に変換する
        
        Args:
            text: 音声に変換するテキスト
            
        Returns:
            Optional[bytes]: WAV形式の音声データ、エラー時はNone
        """
        try:
            params = {
                "text": text,
                "speaker_id": self.speaker_id,
                "model_id": self.model_id
            }
            
            response = requests.get(f"{SBV2_API_URL}/voice", params=params)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            print(f"エラー: 音声合成に失敗しました。")
            print(f"詳細: {str(e)}")
            return None
    
    async def play_audio(self, audio_data: bytes) -> None:
        """
        音声データを再生する
        
        Args:
            audio_data: WAV形式の音声データ
        """
        if not audio_data:
            return
        
        # 一時ファイルに音声データを書き込む
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(audio_data)
        
        try:
            # OSに応じた音声再生コマンドを実行
            if sys.platform == "win32":
                subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{temp_file_path}').PlaySync()"])
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["afplay", temp_file_path])
            else:  # Linux
                subprocess.run(["aplay", temp_file_path])
        except Exception as e:
            print(f"エラー: 音声の再生に失敗しました。")
            print(f"詳細: {str(e)}")
        finally:
            # 一時ファイルを削除
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    async def process_command(self, command: str) -> bool:
        """
        コマンドを処理する
        
        Args:
            command: 処理するコマンド
            
        Returns:
            bool: 会話を続けるかどうか
        """
        if command.startswith("/model "):
            try:
                model_id = int(command.split(" ")[1])
                model_ids = [m["id"] for m in self.available_models.get("models", [])]
                if model_id in model_ids:
                    self.model_id = model_id
                    print(f"モデルIDを {model_id} に変更しました。")
                else:
                    print(f"エラー: モデルID {model_id} は利用できません。")
            except (IndexError, ValueError):
                print("エラー: 正しいモデルIDを指定してください。例: /model 0")
        
        elif command.startswith("/speaker "):
            try:
                self.speaker_id = int(command.split(" ")[1])
                print(f"話者IDを {self.speaker_id} に変更しました。")
            except (IndexError, ValueError):
                print("エラー: 正しい話者IDを指定してください。例: /speaker 0")
        
        elif command == "/models":
            print("利用可能なモデル:")
            for model in self.available_models.get("models", []):
                print(f"  ID: {model['id']} - 名前: {model['name']}")
        
        elif command == "/help":
            print("利用可能なコマンド:")
            print("  /model <id>  - 使用するモデルIDを変更")
            print("  /speaker <id> - 使用する話者IDを変更")
            print("  /models      - 利用可能なモデル一覧を表示")
            print("  /help        - このヘルプを表示")
            print("  /exit        - チャットを終了")
        
        elif command == "/exit":
            print("チャットを終了します。")
            return False
        
        else:
            print(f"不明なコマンド: {command}")
            print("利用可能なコマンドを確認するには /help と入力してください。")
        
        return True

    async def chat_loop(self) -> None:
        """
        チャットのメインループ
        """
        print("=" * 50)
        print("Voice Chat - Gemini + Style-BERT-VITS2")
        print("=" * 50)
        print("チャットを開始します。終了するには /exit と入力してください。")
        print("コマンド一覧を表示するには /help と入力してください。")
        print("=" * 50)
        
        continue_chat = True
        
        while continue_chat:
            user_input = input("\nあなた > ").strip()
            
            # 空の入力はスキップ
            if not user_input:
                continue
            
            # コマンド処理
            if user_input.startswith("/"):
                continue_chat = await self.process_command(user_input)
                continue
            
            print("応答を生成中...")
            response_text = await self.generate_response(user_input)
            
            print(f"\nAI > {response_text}")
            
            print("音声を生成中...")
            audio_data = await self.text_to_speech(response_text)
            
            if audio_data:
                print("音声を再生中...")
                await self.play_audio(audio_data)


async def main():
    """
    メイン関数
    """
    voice_chat = VoiceChat()
    await voice_chat.chat_loop()


if __name__ == "__main__":
    asyncio.run(main()) 