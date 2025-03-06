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
import io
from pydub import AudioSegment
import numpy as np
import wave
import simpleaudio as sa
import argparse

# .envファイルから環境変数を読み込む
load_dotenv()

# Gemini API設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
    print("エラー: GEMINI_API_KEYが設定されていません。")
    print("1. .envファイルを作成し、以下の内容を追加してください：")
    print("   GEMINI_API_KEY=あなたのGemini APIキー")
    print("2. https://ai.google.dev/ からAPIキーを取得できます。")
    sys.exit(1)

# テスト用のダミーAPIキーの場合は警告を表示
if GEMINI_API_KEY == "dummy_api_key_for_testing":
    print("警告: テスト用のダミーAPIキーを使用しています。実際のAPIキーに置き換えてください。")
    print("実際のAPIキーは https://ai.google.dev/ から取得できます。")

# デフォルト設定
DEFAULT_SPEAKER_ID = int(os.getenv("DEFAULT_SPEAKER_ID", "0"))
DEFAULT_MODEL_ID = int(os.getenv("DEFAULT_MODEL_ID", "0"))

# アプリケーション設定をインポート
from configuration.appconfig import (
    DEFAULT_VOLUME,
    CHARACTER_PROMPTS_DIR,
    DEFAULT_CHARACTER,
    CHARACTER_COMMON_SETTINGS,
    PORT,
)

# サーバー接続設定
API_HOST = "localhost"  # APIサーバーのホスト（0.0.0.0ではなくlocalhostを使用）

# Gemini APIの設定
genai.configure(api_key=GEMINI_API_KEY)

class VoiceChat:
    """
    Gemini APIとStyle-BERT-VITS2を組み合わせたチャットボットクラス
    """
    
    def __init__(self, character_name: str = DEFAULT_CHARACTER):
        """
        チャットボットの初期化
        
        Args:
            character_name: 使用するキャラクター設定の名前
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
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        # Gemini 2.0 Flash モデルを使用
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # キャラクター設定を読み込む
        self.character_name = character_name
        character_prompt = self._load_character_prompt(character_name)
        
        # チャット履歴の初期化（キャラクター設定を含める）
        self.chat_session = self.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [f"こんにちは。あなたは以下の設定に従って会話するAIアシスタントです。この設定を厳守してください：\n\n{character_prompt}"]
                },
                {
                    "role": "model",
                    "parts": ["かしこまりました。設定に従って会話いたします。どのようにお手伝いできますか？"]
                }
            ]
        )
        
        # 利用可能なモデルを取得
        self.available_models = self._get_available_models()
        
        # 音声設定
        self.speaker_id = DEFAULT_SPEAKER_ID
        self.model_id = DEFAULT_MODEL_ID
        self.volume = DEFAULT_VOLUME  # 音量設定を追加
        
        print(f"利用可能な音声モデル:")
        for model in self.available_models.get("models", []):
            print(f"  ID: {model['id']} - 名前: {model['name']}")
        print(f"デフォルトモデルID: {self.available_models.get('default_model_id', 0)}")
        print(f"現在選択中のモデルID: {self.model_id}")
        print(f"現在選択中の話者ID: {self.speaker_id}")
        print(f"現在の音量: {self.volume}")
        print(f"現在のキャラクター設定: {self.character_name}")
        print("=" * 50)
    
    def _load_character_prompt(self, character_name: str) -> str:
        """
        キャラクタープロンプトファイルを読み込む
        
        Args:
            character_name: キャラクター設定の名前
            
        Returns:
            str: キャラクタープロンプトの内容（共通設定を含む）
        """
        prompt_path = os.path.join(CHARACTER_PROMPTS_DIR, f"{character_name}.txt")
        
        # ファイルが存在しない場合はデフォルトを使用
        if not os.path.exists(prompt_path) and character_name != DEFAULT_CHARACTER:
            print(f"警告: キャラクター '{character_name}' が見つかりません。デフォルトを使用します。")
            return self._load_character_prompt(DEFAULT_CHARACTER)
            
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                character_prompt = f.read().strip()
            # 共通設定とキャラクタープロンプトを組み合わせる
            combined_prompt = f"【共通設定】\n{CHARACTER_COMMON_SETTINGS.strip()}\n\n【個別設定】\n{character_prompt}"
            return combined_prompt
            
        except Exception as e:
            print(f"エラー: キャラクタープロンプトの読み込みに失敗しました: {str(e)}")
            # デフォルトのプロンプト
            default_prompt = "あなたは親切で丁寧な日本語AIアシスタントです。"
            return f"【共通設定】\n{CHARACTER_COMMON_SETTINGS.strip()}\n\n【個別設定】\n{default_prompt}"
    def _get_available_models(self) -> Dict:
        """
        利用可能な音声モデルの一覧を取得する
        
        Returns:
            Dict: モデル情報の辞書
        """
        try:
            # APIサーバーに接続を試みる
            max_port = PORT + 10  # 最大10ポート試す
            current_port = PORT
            
            while current_port <= max_port:
                try:
                    api_url = f"http://{API_HOST}:{current_port}/models"
                    response = requests.get(api_url, timeout=5)
                    response.raise_for_status()
                    return response.json()
                except requests.RequestException as e:
                    if "Connection refused" in str(e) and current_port < max_port:
                        # 接続拒否の場合は次のポートを試す
                        current_port += 1
                    else:
                        # その他のエラーまたは最後のポートでも失敗した場合
                        raise
            
            # すべてのポートで失敗
            raise requests.RequestException(f"すべてのポート（{PORT}〜{max_port}）への接続に失敗しました。")
            
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
        # テスト用のダミーAPIキーの場合はモックレスポンスを返す
        if GEMINI_API_KEY == "dummy_api_key_for_testing":
            return "これはテスト用のモックレスポンスです。実際のAPIキーを設定すると、Gemini APIからの応答が返されます。"
            
        try:
            # 通常モードで応答を生成
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
            
            # APIサーバーに接続を試みる
            max_port = PORT + 10  # 最大10ポート試す
            current_port = PORT
            
            while current_port <= max_port:
                try:
                    api_url = f"http://{API_HOST}:{current_port}/voice"
                    response = requests.get(api_url, params=params, timeout=5)
                    response.raise_for_status()
                    return response.content
                except requests.RequestException as e:
                    if "Connection refused" in str(e) and current_port < max_port:
                        # 接続拒否の場合は次のポートを試す
                        current_port += 1
                    else:
                        # その他のエラーまたは最後のポートでも失敗した場合
                        raise
            
            # すべてのポートで失敗
            raise requests.RequestException(f"すべてのポート（{PORT}〜{max_port}）への接続に失敗しました。")
            
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
        
        try:
            # WAVデータを解析
            with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                # WAVファイルのパラメータを取得
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                
                # 音声データを読み込む
                raw_data = wf.readframes(n_frames)
            
            # 音量を調整
            try:
                # バイトデータをnumpy配列に変換
                if sample_width == 2:  # 16-bit
                    audio_array = np.frombuffer(raw_data, dtype=np.int16)
                elif sample_width == 4:  # 32-bit
                    audio_array = np.frombuffer(raw_data, dtype=np.int32)
                else:
                    # その他のフォーマットはサポートしない
                    audio_array = np.frombuffer(raw_data, dtype=np.int16)
                
                # 音量調整（0.0〜1.0の範囲）
                volume_factor = self.volume * 2  # 0.0→0倍, 0.5→1倍, 1.0→2倍
                audio_array = (audio_array * volume_factor).astype(audio_array.dtype)
                
                # numpy配列をバイトデータに戻す
                raw_data = audio_array.tobytes()
            except Exception as e:
                # 音量調整に失敗した場合は元のデータを使用
                print(f"音量調整に失敗しました: {str(e)}")
            
            print(f"音声再生中... (音量: {self.volume:.1f})")
            
            # simpleaudioで再生
            try:
                wave_obj = sa.WaveObject(raw_data, channels, sample_width, framerate)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            except Exception as e:
                print(f"simpleaudioでの再生に失敗しました: {str(e)}")
                
                # フォールバック: 一時ファイルを使用
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file_path = temp_file.name
                        temp_file.write(audio_data)
                    
                    # OSに応じた音声再生コマンドを実行
                    if sys.platform == "win32":
                        # Windowsの場合、PowerShellを使用して再生
                        subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{temp_file_path}').PlaySync()"], check=True)
                    elif sys.platform == "darwin":  # macOS
                        # macOSの場合、afplayコマンドを使用
                        subprocess.run(["afplay", temp_file_path], check=True)
                    else:  # Linux
                        # Linuxの場合、aplayコマンドを使用
                        subprocess.run(["aplay", temp_file_path], check=True)
                    
                    # 一時ファイルを削除
                    os.unlink(temp_file_path)
                except Exception as e2:
                    print(f"フォールバック再生にも失敗しました: {str(e2)}")
            
        except Exception as e:
            print(f"エラー: 音声の再生に失敗しました。")
            print(f"詳細: {str(e)}")
    
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
        
        elif command.startswith("/volume "):
            try:
                volume = float(command.split(" ")[1])
                if 0.0 <= volume <= 1.0:
                    self.volume = volume
                    print(f"音量を {volume} に変更しました。")
                else:
                    print("エラー: 音量は0.0から1.0の間で指定してください。")
            except (IndexError, ValueError):
                print("エラー: 正しい音量を指定してください。例: /volume 0.8")
        
        elif command.startswith("/character "):
            try:
                character_name = command.split(" ")[1]
                # キャラクター設定を読み込む
                character_prompt = self._load_character_prompt(character_name)
                
                # 新しいチャットセッションを開始
                self.character_name = character_name
                self.chat_session = self.model.start_chat(
                    history=[
                        {
                            "role": "user",
                            "parts": [f"こんにちは。あなたは以下の設定に従って会話するAIアシスタントです。この設定を厳守してください：\n\n{character_prompt}"]
                        },
                        {
                            "role": "model",
                            "parts": ["かしこまりました。設定に従って会話いたします。どのようにお手伝いできますか？"]
                        }
                    ]
                )
                print(f"キャラクター設定を '{character_name}' に変更しました。")
            except IndexError:
                print("エラー: 正しいキャラクター名を指定してください。例: /character friendly")
        
        elif command == "/characters":
            print("利用可能なキャラクター設定:")
            for file in os.listdir(CHARACTER_PROMPTS_DIR):
                if file.endswith(".txt"):
                    character_name = os.path.splitext(file)[0]
                    print(f"  - {character_name}" + (" (現在選択中)" if character_name == self.character_name else ""))
        
        elif command == "/models":
            print("利用可能なモデル:")
            for model in self.available_models.get("models", []):
                print(f"  ID: {model['id']} - 名前: {model['name']}")
        
        elif command == "/help":
            print("利用可能なコマンド:")
            print("  /model <id>  - 使用するモデルIDを変更")
            print("  /speaker <id> - 使用する話者IDを変更")
            print("  /volume <value> - 音声の音量を変更（0.0〜1.0）")
            print("  /character <name> - キャラクター設定を変更")
            print("  /characters  - 利用可能なキャラクター設定一覧を表示")
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
            
            # 音声合成と再生
            audio_data = await self.text_to_speech(response_text)
            if audio_data:
                await self.play_audio(audio_data)


async def main():
    """
    メイン関数
    """
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Voice Chat - Gemini + Style-BERT-VITS2")
    parser.add_argument("--character", "-c", type=str, default=DEFAULT_CHARACTER,
                        help=f"使用するキャラクター設定の名前（デフォルト: {DEFAULT_CHARACTER}）")
    args = parser.parse_args()
    
    # 利用可能なキャラクター一覧を表示
    print("利用可能なキャラクター設定:")
    for file in os.listdir(CHARACTER_PROMPTS_DIR):
        if file.endswith(".txt"):
            character_name = os.path.splitext(file)[0]
            print(f"  - {character_name}")
    
    # キャラクター設定を指定してチャットボットを初期化
    chatclient = VoiceChat(character_name=args.character)
    await chatclient.chat_loop()


if __name__ == "__main__":
    asyncio.run(main()) 