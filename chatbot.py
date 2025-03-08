"""
Voice Chat - Gemini APIとStyle-BERT-VITS2を直接組み込んだチャットボット

このモジュールはGoogle Gemini 2.0 Flash APIを使用してテキスト応答を生成し、
Style-BERT-VITS2を直接使用して音声に変換するコマンドラインチャットボットを提供します。APIサーバーを経由せず、より高速な応答を実現します。
"""

import os
import sys
import json
import asyncio
import tempfile
import subprocess
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import io
import numpy as np
import wave
import simpleaudio as sa
import argparse
import hashlib
from time import time
import logging
import glob

# Style-BERT-VITS2のインポート
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
)

# 自作モジュールのインポート
from modules.model_manager import ModelManager
from modules.voice_chat import DirectVoiceChat
from modules.utils import setup_logging, load_bert_models, check_api_key, get_env_path

# .envファイルから環境変数を読み込む
dotenv_path = get_env_path()
load_dotenv(dotenv_path)

# Gemini API設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
help_message = """
1. configuration/.envファイルを作成し、以下の内容を追加してください：
   GEMINI_API_KEY=あなたのGemini APIキー
2. https://ai.google.dev/ からAPIキーを取得できます。
"""

if not check_api_key(GEMINI_API_KEY, "gemini_api_key", help_message):
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
    MODEL_DIR,
    USE_GPU,
    VERBOSE,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    DEFAULT_SDP_RATIO,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_ASSIST_TEXT_WEIGHT,
    BERT_MODEL_NAME,
    MAX_CACHE_SIZE,
)

# Gemini APIの設定
genai.configure(api_key=GEMINI_API_KEY)

# ロギング設定
logger = logging.getLogger(__name__)
setup_logging()

# BERTモデルの読み込み
load_bert_models(Languages.JP, BERT_MODEL_NAME)

# モデル管理クラスのインスタンス化
model_manager = ModelManager(MODEL_DIR, USE_GPU, VERBOSE, MAX_CACHE_SIZE)

class VoiceChat(DirectVoiceChat):
    """
    Gemini APIとStyle-BERT-VITS2を直接使用したチャットボットクラス
    
    このクラスはDirectVoiceChatを拡張し、コマンドライン対話機能を追加します。
    """
    
    def __init__(self, character_name: str = DEFAULT_CHARACTER):
        """
        VoiceChatクラスの初期化
        
        Args:
            character_name: 使用するキャラクター設定の名前
        """
        super().__init__(
            character_name=character_name,
            character_prompts_dir=CHARACTER_PROMPTS_DIR,
            character_common_settings=CHARACTER_COMMON_SETTINGS,
            model_manager=model_manager
        )
        
        # 音声設定
        self.speaker_id = DEFAULT_SPEAKER_ID
        self.model_id = DEFAULT_MODEL_ID
        self.volume = DEFAULT_VOLUME
        
        # TTS設定
        self.style = DEFAULT_STYLE
        self.style_weight = DEFAULT_STYLE_WEIGHT
        self.sdp_ratio = DEFAULT_SDP_RATIO
        self.noise = DEFAULT_NOISE
        self.noise_w = DEFAULT_NOISEW
        self.length = DEFAULT_LENGTH
        self.line_split = DEFAULT_LINE_SPLIT
        self.split_interval = DEFAULT_SPLIT_INTERVAL
        self.assist_text_weight = DEFAULT_ASSIST_TEXT_WEIGHT
        
        # 利用可能なモデルを表示
        models = model_manager.get_available_models()["models"]
        print(f"利用可能な音声モデル:")
        for model in models:
            print(f"  ID: {model['id']} - 名前: {model['name']}")
        print(f"デフォルトモデルID: {model_manager.default_model_id}")
        print(f"現在選択中のモデルID: {self.model_id}")
        print(f"現在選択中の話者ID: {self.speaker_id}")
        print(f"現在の音量: {self.volume}")
        print(f"現在のキャラクター設定: {self.character_name}")
        print("=" * 50)
    
    async def process_command(self, command: str) -> bool:
        """
        コマンドを処理する
        
        Args:
            command: 処理するコマンド
            
        Returns:
            bool: 処理が成功したかどうか
        """
        # コマンドの解析
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False
            
        cmd = cmd_parts[0].lower()
        args = cmd_parts[1:]
        
        # 終了コマンド
        if cmd in ["exit", "quit", "q", "終了"]:
            print("チャットを終了します。")
            return True
            
        # ヘルプコマンド
        elif cmd in ["help", "h", "?", "ヘルプ"]:
            self._show_help()
            return False
            
        # 話者ID変更
        elif cmd in ["speaker", "話者"]:
            if not args:
                print(f"現在の話者ID: {self.speaker_id}")
                return False
                
            try:
                speaker_id = int(args[0])
                self.speaker_id = speaker_id
                print(f"話者IDを {speaker_id} に変更しました。")
            except ValueError:
                print("話者IDは整数で指定してください。")
            return False
            
        # モデルID変更
        elif cmd in ["model", "モデル"]:
            if not args:
                print(f"現在のモデルID: {self.model_id}")
                return False
                
            try:
                model_id = int(args[0])
                # モデルの存在確認
                models = model_manager.get_available_models()["models"]
                if model_id < 0 or model_id >= len(models):
                    print(f"モデルID {model_id} は存在しません。")
                    print(f"利用可能なモデルID: 0〜{len(models)-1}")
                    return False
                    
                self.model_id = model_id
                print(f"モデルIDを {model_id} に変更しました。")
            except ValueError:
                print("モデルIDは整数で指定してください。")
            return False
            
        # 音量変更
        elif cmd in ["volume", "vol", "音量"]:
            if not args:
                print(f"現在の音量: {self.volume}")
                return False
                
            try:
                volume = float(args[0])
                if volume < 0:
                    print("音量は0以上の値を指定してください。")
                    return False
                    
                self.volume = volume
                print(f"音量を {volume} に変更しました。")
            except ValueError:
                print("音量は数値で指定してください。")
            return False
            
        # スタイル変更
        elif cmd in ["style", "スタイル"]:
            if not args:
                print(f"現在のスタイル: {self.style}")
                return False
                
            style = args[0]
            self.style = style
            print(f"スタイルを {style} に変更しました。")
            return False
            
        # スタイルの重み変更
        elif cmd in ["styleweight", "スタイル重み"]:
            if not args:
                print(f"現在のスタイル重み: {self.style_weight}")
                return False
                
            try:
                weight = float(args[0])
                if weight < 0 or weight > 1:
                    print("スタイル重みは0〜1の範囲で指定してください。")
                    return False
                    
                self.style_weight = weight
                print(f"スタイル重みを {weight} に変更しました。")
            except ValueError:
                print("スタイル重みは数値で指定してください。")
            return False
            
        # SDP比率変更
        elif cmd in ["sdp", "sdpratio"]:
            if not args:
                print(f"現在のSDP比率: {self.sdp_ratio}")
                return False
                
            try:
                ratio = float(args[0])
                if ratio < 0 or ratio > 1:
                    print("SDP比率は0〜1の範囲で指定してください。")
                    return False
                    
                self.sdp_ratio = ratio
                print(f"SDP比率を {ratio} に変更しました。")
            except ValueError:
                print("SDP比率は数値で指定してください。")
            return False
            
        # ノイズ量変更
        elif cmd in ["noise", "ノイズ"]:
            if not args:
                print(f"現在のノイズ量: {self.noise}")
                return False
                
            try:
                noise = float(args[0])
                if noise < 0 or noise > 1:
                    print("ノイズ量は0〜1の範囲で指定してください。")
                    return False
                    
                self.noise = noise
                print(f"ノイズ量を {noise} に変更しました。")
            except ValueError:
                print("ノイズ量は数値で指定してください。")
            return False
            
        # ノイズ幅変更
        elif cmd in ["noisew", "ノイズ幅"]:
            if not args:
                print(f"現在のノイズ幅: {self.noise_w}")
                return False
                
            try:
                noise_w = float(args[0])
                if noise_w < 0 or noise_w > 1:
                    print("ノイズ幅は0〜1の範囲で指定してください。")
                    return False
                    
                self.noise_w = noise_w
                print(f"ノイズ幅を {noise_w} に変更しました。")
            except ValueError:
                print("ノイズ幅は数値で指定してください。")
            return False
            
        # 長さ変更
        elif cmd in ["length", "長さ"]:
            if not args:
                print(f"現在の長さ: {self.length}")
                return False
                
            try:
                length = float(args[0])
                if length <= 0:
                    print("長さは0より大きい値を指定してください。")
                    return False
                    
                self.length = length
                print(f"長さを {length} に変更しました。")
            except ValueError:
                print("長さは数値で指定してください。")
            return False
            
        # 自動分割設定
        elif cmd in ["split", "分割"]:
            if not args:
                print(f"現在の自動分割設定: {self.line_split}")
                return False
                
            if args[0].lower() in ["on", "true", "yes", "1"]:
                self.line_split = True
                print("自動分割をオンにしました。")
            elif args[0].lower() in ["off", "false", "no", "0"]:
                self.line_split = False
                print("自動分割をオフにしました。")
            else:
                print("自動分割の設定には on/off を指定してください。")
            return False
            
        # 分割間隔変更
        elif cmd in ["splitinterval", "分割間隔"]:
            if not args:
                print(f"現在の分割間隔: {self.split_interval}")
                return False
                
            try:
                interval = float(args[0])
                if interval <= 0:
                    print("分割間隔は0より大きい値を指定してください。")
                    return False
                    
                self.split_interval = interval
                print(f"分割間隔を {interval} に変更しました。")
            except ValueError:
                print("分割間隔は数値で指定してください。")
            return False
            
        # キャラクター変更
        elif cmd in ["character", "キャラクター"]:
            if not args:
                print(f"現在のキャラクター: {self.character_name}")
                return False
                
            character_name = args[0]
            # キャラクタープロンプトを読み込む
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
            
            self.character_name = character_name
            print(f"キャラクターを {character_name} に変更しました。")
            return False
            
        # 設定表示
        elif cmd in ["settings", "設定"]:
            self._show_settings()
            return False
            
        # 未知のコマンド
        else:
            print(f"未知のコマンド: {cmd}")
            print("利用可能なコマンドを表示するには 'help' と入力してください。")
            return False
    
    def _show_help(self):
        """
        ヘルプメッセージを表示する
        """
        help_text = """
利用可能なコマンド:
  help, h, ?, ヘルプ                    - このヘルプを表示
  exit, quit, q, 終了                   - チャットを終了
  speaker, 話者 [ID]                    - 話者IDを変更/表示
  model, モデル [ID]                    - モデルIDを変更/表示
  volume, vol, 音量 [値]                - 音量を変更/表示
  style, スタイル [名前]                - スタイルを変更/表示
  styleweight, スタイル重み [0-1]       - スタイルの重みを変更/表示
  sdp, sdpratio [0-1]                   - SDP比率を変更/表示
  noise, ノイズ [0-1]                   - ノイズ量を変更/表示
  noisew, ノイズ幅 [0-1]                - ノイズ幅を変更/表示
  length, 長さ [値]                     - 音声の長さを変更/表示
  split, 分割 [on/off]                  - 自動分割の設定を変更/表示
  splitinterval, 分割間隔 [値]          - 分割間隔を変更/表示
  character, キャラクター [名前]        - キャラクター設定を変更/表示
  settings, 設定                        - 現在の設定を表示
        """
        print(help_text)
    
    def _show_settings(self):
        """
        現在の設定を表示する
        """
        settings = f"""
現在の設定:
  キャラクター: {self.character_name}
  モデルID: {self.model_id}
  話者ID: {self.speaker_id}
  音量: {self.volume}
  スタイル: {self.style}
  スタイル重み: {self.style_weight}
  SDP比率: {self.sdp_ratio}
  ノイズ量: {self.noise}
  ノイズ幅: {self.noise_w}
  長さ: {self.length}
  自動分割: {self.line_split}
  分割間隔: {self.split_interval}
        """
        print(settings)
    
    async def chat_loop(self):
        """
        チャットループを実行する
        """
        print("\nVerbalis を起動しました。")
        print("チャットを終了するには 'exit' と入力してください。")
        print("コマンド一覧を表示するには 'help' と入力してください。")
        print("=" * 50)
        
        while True:
            try:
                # ユーザー入力を取得
                user_input = input("\nあなた: ")
                
                # 空の入力はスキップ
                if not user_input.strip():
                    continue
                
                # コマンドの場合
                if user_input.startswith("/"):
                    command = user_input[1:]  # スラッシュを除去
                    exit_chat = await self.process_command(command)
                    if exit_chat:
                        break
                    continue
                
                # 応答を生成
                print("応答を生成中...")
                response = await self.generate_response(user_input)
                
                # 応答を表示
                print(f"\n{self.character_name}: {response}")
                
                # 音声に変換して再生
                print("音声を生成中...")
                audio_data = await self.text_to_speech(
                    text=response,
                    style=self.style,
                    style_weight=self.style_weight,
                    sdp_ratio=self.sdp_ratio,
                    noise=self.noise,
                    noise_w=self.noise_w,
                    length=self.length,
                    line_split=self.line_split
                )
                
                if audio_data:
                    await self.play_audio(audio_data)
                else:
                    print("音声の生成に失敗しました。")
                
            except KeyboardInterrupt:
                print("\nチャットを終了します。")
                break
                
            except Exception as e:
                logger.error(f"エラーが発生しました: {str(e)}")
                print(f"エラーが発生しました: {str(e)}")

async def main():
    """
    メイン関数
    """
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Verbalis")
    parser.add_argument("--character", "-c", type=str, default=DEFAULT_CHARACTER,
                        help=f"使用するキャラクター設定の名前（デフォルト: {DEFAULT_CHARACTER}）")
    parser.add_argument("--speaker", "-s", type=int, default=DEFAULT_SPEAKER_ID,
                        help=f"話者ID（デフォルト: {DEFAULT_SPEAKER_ID}）")
    parser.add_argument("--model", "-m", type=int, default=DEFAULT_MODEL_ID,
                        help=f"モデルID（デフォルト: {DEFAULT_MODEL_ID}）")
    parser.add_argument("--volume", "-v", type=float, default=DEFAULT_VOLUME,
                        help=f"音量（デフォルト: {DEFAULT_VOLUME}）")
    args = parser.parse_args()
    
    # チャットボットの初期化
    chat = VoiceChat(character_name=args.character)
    chat.speaker_id = args.speaker
    chat.model_id = args.model
    chat.volume = args.volume
    
    # チャットループを実行
    await chat.chat_loop()

if __name__ == "__main__":
    asyncio.run(main()) 