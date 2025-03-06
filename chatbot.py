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
)

# Gemini APIの設定
genai.configure(api_key=GEMINI_API_KEY)

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectTTS:
    """
    Style-BERT-VITS2を直接使用したテキスト音声合成クラス
    
    このクラスはStyle-BERT-VITS2モデルを直接ラップし、テキストから音声を生成する機能と
    生成した音声をキャッシュする機能を提供します。APIサーバーを経由せずに高速に動作します。
    """
    
    def __init__(self, model_file: str, config_file: str, style_file: str, use_gpu: bool = False, verbose: bool = False):
        """
        DirectTTSクラスの初期化
        
        Args:
            model_file: モデルファイルのパス
            config_file: 設定ファイルのパス
            style_file: スタイルベクトルファイルのパス
            use_gpu: GPUを使用するかどうか
            verbose: 詳細なログを出力するかどうか
        """
        self.tts_model = TTSModel(
            model_path=model_file,
            config_path=config_file,
            style_vec_path=style_file,
            device="cuda" if use_gpu else "cpu"
        )
        self.verbose = verbose
        self.cache = {}
        self.cache_keys = []  # キャッシュキーの順序を保持
        
        # 最大キャッシュサイズ
        self.max_cache_size = 100

    def generate_cache_key(self, text: str, speaker_id: int, style: str) -> str:
        """
        キャッシュキーを生成する
        
        Args:
            text: 音声に変換するテキスト
            speaker_id: 話者ID
            style: スタイル名
            
        Returns:
            生成されたキャッシュキー（MD5ハッシュ）
        """
        unique_string = f"{text}_{speaker_id}_{style}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    async def tts(self, text: str, speaker_id: int, style: str, **kwargs) -> bytes:
        """
        テキストから音声を生成する
        
        Args:
            text: 音声に変換するテキスト
            speaker_id: 話者ID
            style: スタイル名
            **kwargs: その他のパラメータ
            
        Returns:
            WAV形式の音声データ（バイト列）
            
        Raises:
            Exception: 音声生成中にエラーが発生した場合
        """
        # 短いテキストの場合は処理を高速化
        if len(text) <= 10:
            kwargs['line_split'] = False  # 短いテキストは分割しない
        
        cache_key = self.generate_cache_key(text, speaker_id, style)

        # キャッシュをチェック
        if cache_key in self.cache:
            if self.verbose:
                logger.info(f"キャッシュヒット: {speaker_id}/{style}: {text}")
            
            # キャッシュの使用順を更新
            if cache_key in self.cache_keys:
                self.cache_keys.remove(cache_key)
            self.cache_keys.append(cache_key)
            
            return self.cache[cache_key]

        # 別スレッドでTTSを実行
        try:
            start_time = time()
            rate, audio = await asyncio.to_thread(self.tts_model.infer, text=text, speaker_id=speaker_id, style=style, **kwargs)
            if self.verbose:
                logger.info(f"音声生成完了（{time() - start_time:.2f}秒）: {text}")

        except Exception as ex:
            logger.error(f"音声生成エラー: {str(ex)}")
            raise ex

        # WAV形式に変換
        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(1)      # モノラル
                wf.setsampwidth(2)      # 16ビット
                wf.setframerate(rate)   # サンプリングレート
                wf.writeframes(audio)
            buffer.seek(0)

            # キャッシュに保存
            audio_data = buffer.read()
            
            # キャッシュサイズを制限
            if len(self.cache_keys) >= self.max_cache_size:
                # 最も古いキャッシュを削除
                oldest_key = self.cache_keys.pop(0)
                del self.cache[oldest_key]
                
            self.cache[cache_key] = audio_data
            self.cache_keys.append(cache_key)
            
            return audio_data
            
        except Exception as ex:
            logger.error(f"WAV変換エラー: {str(ex)}")
            raise ex 

class ModelManager:
    """
    Style-BERT-VITS2モデルの管理クラス
    
    このクラスはモデルファイルの検索、読み込み、管理を行います。
    """
    
    def __init__(self, model_dir: str = MODEL_DIR, use_gpu: bool = USE_GPU, verbose: bool = VERBOSE):
        """
        ModelManagerクラスの初期化
        
        Args:
            model_dir: モデルディレクトリのパス
            use_gpu: GPUを使用するかどうか
            verbose: 詳細なログを出力するかどうか
        """
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.models = {}  # モデルのキャッシュ
        
        # モデル情報をスキャン
        self.model_info = self._scan_models()
        
    def _scan_models(self) -> Dict:
        """
        モデルディレクトリをスキャンして利用可能なモデルを検索する
        
        モデル構造:
        model_assets/
          ├── モデル名1/
          │   ├── モデル名1.pth または モデル名1.safetensors
          │   ├── config.json
          │   └── style_vectors.npy
          └── モデル名2/
              ├── モデル名2.pth または モデル名2.safetensors
              ├── config.json
              └── style_vectors.npy
        
        Returns:
            Dict: モデル情報の辞書
        """
        models = []
        default_model_id = 0
        
        try:
            # モデルディレクトリが存在しない場合は作成
            os.makedirs(self.model_dir, exist_ok=True)
            
            # 方法1: サブディレクトリベースの検索（apiserver.pyのget_available_models関数と同様）
            model_dirs = glob.glob(os.path.join(self.model_dir, "*"))
            for model_dir in sorted(model_dirs):
                if os.path.isdir(model_dir):
                    model_name = os.path.basename(model_dir)
                    
                    # モデルファイルを検索（.safetensorsと.pth）
                    model_file = None
                    safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
                    pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
                    
                    if safetensors_files:
                        model_file = safetensors_files[0]  # 最初に見つかったsafetensorsファイルを使用
                    elif pth_files:
                        model_file = pth_files[0]  # safetensorsがなければpthファイルを使用
                    
                    if not model_file:
                        logger.warning(f"モデルディレクトリ '{model_dir}' にモデルファイルが見つかりません。スキップします。")
                        continue
                    
                    # 設定ファイルを検索
                    config_file = os.path.join(model_dir, "config.json")
                    if not os.path.exists(config_file):
                        # モデル名と同じ名前の設定ファイルを検索
                        config_files = glob.glob(os.path.join(model_dir, "*.json"))
                        if config_files:
                            config_file = config_files[0]
                        else:
                            logger.warning(f"モデル '{model_name}' の設定ファイル (.json) が見つかりません。スキップします。")
                            continue
                    
                    # スタイルファイルを検索
                    style_file = os.path.join(model_dir, "style_vectors.npy")
                    if not os.path.exists(style_file):
                        # 他のパターンのスタイルファイルを検索
                        style_files = glob.glob(os.path.join(model_dir, "*_style.npy"))
                        if style_files:
                            style_file = style_files[0]
                        else:
                            logger.warning(f"モデル '{model_name}' のスタイルファイル (.npy) が見つかりません。スキップします。")
                            continue
                    
                    # モデル情報を追加
                    models.append({
                        "id": len(models),
                        "name": model_name,
                        "model_file": model_file,
                        "config_file": config_file,
                        "style_file": style_file
                    })
                    
                    logger.info(f"モデル '{model_name}' を検出しました:")
                    logger.info(f"  モデルファイル: {os.path.basename(model_file)}")
                    logger.info(f"  設定ファイル: {os.path.basename(config_file)}")
                    logger.info(f"  スタイルファイル: {os.path.basename(style_file)}")
            
            # 方法2: 再帰的なファイル検索（apiserver.pyのget_tts_model関数と同様）
            if not models:
                logger.info("サブディレクトリベースの検索で見つからなかったため、再帰的な検索を試みます...")
                
                # モデルファイルを再帰的に検索
                pth_files = glob.glob(os.path.join(self.model_dir, "**/*.pth"), recursive=True)
                safetensors_files = glob.glob(os.path.join(self.model_dir, "**/*.safetensors"), recursive=True)
                model_files = sorted(pth_files + safetensors_files)
                
                for i, model_file in enumerate(model_files):
                    model_dir = os.path.dirname(model_file)
                    model_name = os.path.splitext(os.path.basename(model_file))[0]
                    base_name = os.path.splitext(model_file)[0]
                    
                    # 設定ファイルを検索
                    config_file = None
                    config_candidates = [
                        f"{base_name}.json",
                        os.path.join(model_dir, "config.json")
                    ]
                    for candidate in config_candidates:
                        if os.path.exists(candidate):
                            config_file = candidate
                            break
                    
                    if not config_file:
                        logger.warning(f"モデル '{model_name}' の設定ファイルが見つかりません。スキップします。")
                        continue
                    
                    # スタイルファイルを検索
                    style_file = None
                    style_candidates = [
                        f"{base_name}_style.npy",
                        os.path.join(model_dir, "style_vectors.npy"),
                        os.path.join(model_dir, f"{model_name}_style.npy")
                    ]
                    for candidate in style_candidates:
                        if os.path.exists(candidate):
                            style_file = candidate
                            break
                    
                    if not style_file:
                        logger.warning(f"モデル '{model_name}' のスタイルファイルが見つかりません。スキップします。")
                        continue
                    
                    # モデル情報を追加
                    models.append({
                        "id": len(models),
                        "name": model_name,
                        "model_file": model_file,
                        "config_file": config_file,
                        "style_file": style_file
                    })
                    
                    logger.info(f"モデル '{model_name}' を検出しました (再帰的検索):")
                    logger.info(f"  モデルファイル: {model_file}")
                    logger.info(f"  設定ファイル: {config_file}")
                    logger.info(f"  スタイルファイル: {style_file}")
            
            if models:
                default_model_id = 0
                logger.info(f"{len(models)}個のモデルが見つかりました")
            else:
                logger.warning("有効なモデルが見つかりませんでした。必要なファイルが揃っているか確認してください。")
                logger.info(f"モデルは '{self.model_dir}' ディレクトリに配置してください。")
                logger.info("必要なファイル: モデルファイル(.pthまたは.safetensors), 設定ファイル(.json), スタイルファイル(.npy)")
                
            if self.verbose:
                for model in models:
                    logger.info(f"モデル {model['id']}: {model['name']}")
                
        except Exception as e:
            logger.error(f"モデルスキャンエラー: {str(e)}")
        
        return {
            "models": models,
            "default_model_id": default_model_id
        }
    
    def get_model(self, model_id: int) -> DirectTTS:
        """
        指定されたIDのモデルを取得する
        
        Args:
            model_id: モデルID
            
        Returns:
            DirectTTS: TTSモデルインスタンス
            
        Raises:
            ValueError: 指定されたIDのモデルが見つからない場合
        """
        # モデルが見つからない場合のチェック
        if not self.model_info["models"]:
            logger.error("モデルが見つかりません。model_assetsディレクトリにモデルファイルが存在するか確認してください。")
            raise ValueError("モデルが見つかりません。model_assetsディレクトリにモデルファイルが存在するか確認してください。")
            
        # モデルIDの範囲をチェック
        if model_id < 0 or model_id >= len(self.model_info["models"]):
            # 範囲外の場合はデフォルトモデルを使用
            model_id = self.model_info["default_model_id"]
            logger.warning(f"指定されたモデルIDが範囲外です。デフォルトモデル（ID: {model_id}）を使用します。")
        
        # モデルがキャッシュにあるか確認
        if model_id in self.models:
            return self.models[model_id]
        
        # モデル情報を取得
        model_data = self.model_info["models"][model_id]
        
        # モデルを読み込む
        try:
            logger.info(f"モデル '{model_data['name']}' を読み込んでいます...")
            tts = DirectTTS(
                model_file=model_data["model_file"],
                config_file=model_data["config_file"],
                style_file=model_data["style_file"],
                use_gpu=self.use_gpu,
                verbose=self.verbose
            )
            
            # キャッシュに保存
            self.models[model_id] = tts
            
            return tts
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {str(e)}")
            raise ValueError(f"モデル '{model_data['name']}' の読み込みに失敗しました: {str(e)}")
    
    def get_available_models(self) -> Dict:
        """
        利用可能なモデルの一覧を取得する
        
        Returns:
            Dict: モデル情報の辞書
        """
        return self.model_info 

class VoiceChat:
    """
    Gemini APIとStyle-BERT-VITS2を直接組み込んだチャットボットクラス
    
    APIサーバーを経由せず、より高速な応答を実現します。
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
                    "parts": [f"こんにちは。あなたは以下の設定に従って会話するAIアシスタントです。この設定を厳守してください：\n\n【共通設定】\n{CHARACTER_COMMON_SETTINGS.strip()}\n\n【個別設定】\n{character_prompt}"]
                },
                {
                    "role": "model",
                    "parts": ["かしこまりました。設定に従って会話いたします。どのようにお手伝いできますか？"]
                }
            ]
        )
        
        # モデルマネージャーの初期化
        self.model_manager = ModelManager()
        
        # 利用可能なモデルを取得
        self.available_models = self.model_manager.get_available_models()
        
        # 音声設定
        self.speaker_id = DEFAULT_SPEAKER_ID
        self.model_id = DEFAULT_MODEL_ID
        self.volume = DEFAULT_VOLUME  # 音量設定
        
        print(f"利用可能な音声モデル:")
        if not self.available_models.get("models"):
            print("  警告: モデルが見つかりません。")
            print(f"  各モデルは '{MODEL_DIR}/モデル名/' のようなサブディレクトリに配置してください。")
            print("  必要なファイル:")
            print("    - モデルファイル: モデル名.pth または モデル名.safetensors")
            print("    - 設定ファイル: config.json")
            print("    - スタイルファイル: style_vectors.npy")
        else:
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
                
            # キャラクタープロンプトと共通設定を組み合わせる
            combined_prompt = f"【共通設定】\n{CHARACTER_COMMON_SETTINGS.strip()}\n\n【個別設定】\n{character_prompt}"
            return combined_prompt
            
        except Exception as e:
            print(f"エラー: キャラクタープロンプトの読み込みに失敗しました: {str(e)}")
            # デフォルトのプロンプト
            default_prompt = "あなたは親切で丁寧な日本語AIアシスタントです。"
            return f"【共通設定】\n{CHARACTER_COMMON_SETTINGS.strip()}\n\n【個別設定】\n{character_prompt}"
    
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
        テキストを音声に変換する（直接SBV2を使用）
        
        Args:
            text: 音声に変換するテキスト
            
        Returns:
            Optional[bytes]: WAV形式の音声データ、エラー時はNone
        """
        try:
            # モデルが存在するか確認
            if not self.available_models.get("models"):
                print("エラー: 利用可能なモデルがありません。")
                print(f"各モデルは '{MODEL_DIR}/モデル名/' のようなサブディレクトリに配置してください。")
                print("必要なファイル:")
                print("  - モデルファイル: モデル名.pth または モデル名.safetensors")
                print("  - 設定ファイル: config.json")
                print("  - スタイルファイル: style_vectors.npy")
                return None
                
            # モデルIDの範囲をチェック
            if self.model_id < 0 or self.model_id >= len(self.available_models["models"]):
                print(f"警告: モデルID {self.model_id} は範囲外です。デフォルトモデルを使用します。")
                self.model_id = self.available_models["default_model_id"]
            
            # モデルを取得
            tts_model = self.model_manager.get_model(self.model_id)
            
            # TTSパラメータ
            params = {
                "language": Languages.JP,
                "sdp_ratio": DEFAULT_SDP_RATIO,
                "noise": DEFAULT_NOISE,
                "noise_w": DEFAULT_NOISEW,
                "length": DEFAULT_LENGTH,
                "line_split": DEFAULT_LINE_SPLIT,
                "split_interval": DEFAULT_SPLIT_INTERVAL,
                "assist_text": None,
                "assist_text_weight": DEFAULT_ASSIST_TEXT_WEIGHT,
                "style_weight": DEFAULT_STYLE_WEIGHT,
                "reference_audio_path": None
            }
            
            # 音声生成
            audio_data = await tts_model.tts(
                text=text,
                speaker_id=self.speaker_id,
                style=DEFAULT_STYLE,
                **params
            )
            
            return audio_data
            
        except Exception as e:
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
            print(f"音声再生エラー: {str(e)}") 

    async def process_command(self, command: str) -> bool:
        """
        コマンドを処理する
        
        Args:
            command: ユーザーが入力したコマンド
            
        Returns:
            bool: チャットを続行するかどうか
        """
        cmd_parts = command.split()
        cmd = cmd_parts[0].lower()
        
        if cmd == "/exit" or cmd == "/quit":
            print("チャットを終了します。")
            return False
            
        elif cmd == "/help":
            print("\n利用可能なコマンド:")
            print("  /exit, /quit - チャットを終了する")
            print("  /help - このヘルプメッセージを表示する")
            print("  /speaker <ID> - 話者IDを変更する")
            print("  /model <ID> - 使用するモデルIDを変更する")
            print("  /volume <0.0-1.0> - 音量を変更する")
            print("  /character <名前> - キャラクター設定を変更する")
            print("  /models - 利用可能なモデル一覧を表示する")
            print("  /characters - 利用可能なキャラクター一覧を表示する")
            print("  /clear - チャット履歴をクリアする")
            
        elif cmd == "/speaker" and len(cmd_parts) > 1:
            try:
                speaker_id = int(cmd_parts[1])
                self.speaker_id = speaker_id
                print(f"話者IDを {speaker_id} に変更しました。")
            except ValueError:
                print("エラー: 話者IDは整数で指定してください。")
                
        elif cmd == "/model" and len(cmd_parts) > 1:
            try:
                model_id = int(cmd_parts[1])
                if 0 <= model_id < len(self.available_models["models"]):
                    self.model_id = model_id
                    print(f"モデルIDを {model_id} に変更しました。")
                else:
                    print(f"エラー: モデルID {model_id} は範囲外です。")
                    print(f"有効なモデルID: 0-{len(self.available_models['models'])-1}")
            except ValueError:
                print("エラー: モデルIDは整数で指定してください。")
                
        elif cmd == "/volume" and len(cmd_parts) > 1:
            try:
                volume = float(cmd_parts[1])
                if 0.0 <= volume <= 1.0:
                    self.volume = volume
                    print(f"音量を {volume:.1f} に変更しました。")
                else:
                    print("エラー: 音量は0.0から1.0の範囲で指定してください。")
            except ValueError:
                print("エラー: 音量は数値で指定してください。")
                
        elif cmd == "/character" and len(cmd_parts) > 1:
            character_name = cmd_parts[1]
            prompt_path = os.path.join(CHARACTER_PROMPTS_DIR, f"{character_name}.txt")
            
            if os.path.exists(prompt_path):
                self.character_name = character_name
                character_prompt = self._load_character_prompt(character_name)
                
                # チャット履歴を初期化
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
            else:
                print(f"エラー: キャラクター '{character_name}' が見つかりません。")
                print("利用可能なキャラクター:")
                for file in os.listdir(CHARACTER_PROMPTS_DIR):
                    if file.endswith(".txt"):
                        print(f"  - {os.path.splitext(file)[0]}")
                
        elif cmd == "/models":
            print("\n利用可能なモデル:")
            for model in self.available_models["models"]:
                print(f"  ID: {model['id']} - 名前: {model['name']}")
            print(f"現在選択中のモデルID: {self.model_id}")
            
        elif cmd == "/characters":
            print("\n利用可能なキャラクター:")
            for file in os.listdir(CHARACTER_PROMPTS_DIR):
                if file.endswith(".txt"):
                    character_name = os.path.splitext(file)[0]
                    if character_name == self.character_name:
                        print(f"  - {character_name} (選択中)")
                    else:
                        print(f"  - {character_name}")
                        
        elif cmd == "/clear":
            # キャラクター設定を保持したまま履歴をクリア
            character_prompt = self._load_character_prompt(self.character_name)
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
            print("チャット履歴をクリアしました。")
            
        else:
            print(f"未知のコマンド: {command}")
            print("利用可能なコマンドを確認するには /help と入力してください。")
        
        return True

    async def chat_loop(self) -> None:
        """
        チャットのメインループ
        """
        print("=" * 50)
        print("Voice Chat - Gemini + Style-BERT-VITS2 (直接組み込み版)")
        print("=" * 50)
        print("チャットを開始します。終了するには /exit と入力してください。")
        print("コマンド一覧を表示するには /help と入力してください。")
        print("=" * 50)
        
        # モデルが見つからない場合は警告を表示
        if not self.available_models.get("models"):
            print("\n警告: 音声モデルが見つからないため、テキスト応答のみで動作します。")
            print("音声機能を使用するには、モデルファイルを配置してください。")
        
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
            
            # 音声合成と再生（モデルがある場合のみ）
            if self.available_models.get("models"):
                print("音声を生成中...")
                audio_data = await self.text_to_speech(response_text)
                if audio_data:
                    await self.play_audio(audio_data)
            else:
                print("(音声モデルが見つからないため、音声再生をスキップします)")


async def main():
    """
    メイン関数
    """
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Voice Chat - Gemini + Style-BERT-VITS2 (直接組み込み版)")
    parser.add_argument("--character", "-c", type=str, default=DEFAULT_CHARACTER,
                        help=f"使用するキャラクター設定の名前（デフォルト: {DEFAULT_CHARACTER}）")
    args = parser.parse_args()
    
    # BERTモデルの初期化
    try:
        logger.info(f"BERTモデルを初期化しています: {BERT_MODEL_NAME}")
        bert_models.load_model(Languages.JP, BERT_MODEL_NAME)
        bert_models.load_tokenizer(Languages.JP, BERT_MODEL_NAME)
        logger.info("BERTモデルの初期化が完了しました")
    except Exception as e:
        logger.error(f"BERTモデルの初期化に失敗しました: {str(e)}")
        logger.warning("BERTモデルなしで続行します。音声合成が正常に動作しない可能性があります。")
    
    # 利用可能なキャラクター一覧を表示
    print("利用可能なキャラクター設定:")
    for file in os.listdir(CHARACTER_PROMPTS_DIR):
        if file.endswith(".txt"):
            character_name = os.path.splitext(file)[0]
            print(f"  - {character_name}")
    
    # キャラクター設定を指定してチャットボットを初期化
    chatbot = VoiceChat(character_name=args.character)
    await chatbot.chat_loop()


if __name__ == "__main__":
    asyncio.run(main()) 