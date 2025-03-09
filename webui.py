"""
Verbalis - Web UI

このモジュールはGradioを使用したVerbalisのWebインターフェースを提供します。
"""

import os
import json
import asyncio
import gradio as gr
import logging
import argparse
import sys
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

# 設定ファイルの読み込み
load_dotenv("configuration/.env")

# 自作モジュールのインポート
from modules.voice_chat import DirectVoiceChat
from modules.model_manager import ModelManager
from modules.utils import load_bert_models
import configuration.appconfig as config

# Style-BERT-VITS2のインポート
from style_bert_vits2.constants import Languages

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API Keyの設定
api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    logger.error("GEMINI_API_KEYが設定されていません。configuration/.envファイルを確認してください。")
    sys.exit(1)

# Gemini APIの設定
genai.configure(api_key=api_key)
logger.info("Gemini APIの設定が完了しました。")

# スタイル用のCSS
STYLE_CSS = """
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}

/* チャット入力エリアのスタイル */
.chat-input-container {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-top: 10px;
}
.chat-input-container .message-box {
    flex-grow: 1;
}
.chat-input-container .send-button {
    align-self: flex-end;
    height: 40px;
}

/* データフレームのスタイル */
.dataframe-container table {
    table-layout: fixed;
    width: 100%;
}

.dataframe-container td {
    word-wrap: break-word;
    overflow-wrap: break-word;
    white-space: pre-wrap !important;
    max-width: 100%;
    vertical-align: top;
    padding: 8px;
}

.dataframe-container th {
    text-align: left;
    padding: 8px;
}

/* スクロールバーのスタイル */
.dataframe-container ::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

.dataframe-container ::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 5px;
}

.dataframe-container ::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 5px;
}

.dataframe-container ::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* ボタンのスタイル */
.icon-button {
    min-width: 36px;
    height: 36px;
    padding: 0 !important;
    display: flex;
    align-items: center;
    justify-content: center;
}
"""

class VerbalisWebUI:
    """
    VerbalisのWebインターフェースクラス
    """
    
    def __init__(self):
        """
        VerbalisWebUIクラスの初期化
        """
        # モデルマネージャーの初期化
        self.model_manager = ModelManager(
            model_dir=config.MODEL_DIR,
            use_gpu=config.USE_GPU,
            verbose=config.VERBOSE,
            max_cache_size=config.MAX_CACHE_SIZE
        )
        
        # 利用可能なモデルの取得
        self.available_models = self.model_manager.get_available_models()
        
        # デフォルトモデルID
        self.default_model_id = self.available_models.get('default_model_id', 0)
        
        # 現在のsafetensor ID
        self.current_safetensor_id = 0
        
        # キャラクター設定の読み込み
        self.characters = self._load_characters()
        
        # 現在のチャットインスタンス
        self.voice_chat = None
        
        # 現在の音声データ
        self.current_audio = None
        
        # チャット履歴
        self.chat_history = []
        
        # 音声生成履歴
        self.voice_history = []
        
        # モデルIDとスタイルのマッピング
        self.model_styles = self._get_model_styles()
        
        # BERTモデルの読み込み
        load_bert_models(Languages.JP, config.BERT_MODEL_NAME)
    
    def _load_characters(self) -> Dict[str, str]:
        """
        キャラクター設定を読み込む
        
        Returns:
            キャラクター名とファイルパスの辞書
        """
        characters = {}
        character_files = Path(config.CHARACTER_PROMPTS_DIR).glob("*.txt")
        
        for file_path in character_files:
            if file_path.name != "common_settings.txt":
                character_name = file_path.stem
                characters[character_name] = str(file_path)
        
        return characters
    
    def _get_model_styles(self) -> Dict[int, List[str]]:
        """
        各モデルで利用可能なスタイルのマッピングを取得する
        
        Returns:
            Dict[int, List[str]]: モデルIDとスタイルリストのマッピング
        """
        model_styles = {}
        for model in self.available_models.get('models', []):
            model_id = model['id']
            try:
                # モデルを取得してスタイル情報を抽出
                tts_model = self.model_manager.get_direct_model(model_id, self.current_safetensor_id)
                if tts_model and hasattr(tts_model, 'style_list'):
                    model_styles[model_id] = tts_model.style_list
                else:
                    # モデルからスタイルリストを取得できない場合はデフォルトを設定
                    model_styles[model_id] = ["Neutral"]
            except Exception as e:
                logger.error(f"モデル {model_id} のスタイル情報取得エラー: {e}")
                model_styles[model_id] = ["Neutral"]
        
        return model_styles
    
    def get_safetensors_for_model(self, model_id: int) -> List[Dict]:
        """
        指定されたモデルIDに対応するsafetensorファイルのリストを取得する
        
        Args:
            model_id: モデルID
            
        Returns:
            List[Dict]: safetensorファイルの情報リスト
        """
        for model in self.available_models.get('models', []):
            if model['id'] == model_id:
                return model.get('safetensors_files', [])
        return []
    
    def update_safetensor_id(self, model_id: int, safetensor_id: int) -> None:
        """
        現在のsafetensor IDを更新する
        
        Args:
            model_id: モデルID
            safetensor_id: 新しいsafetensor ID
        """
        self.current_safetensor_id = safetensor_id
        
        # スタイル情報を更新
        self.model_styles = self._get_model_styles()
    
    def initialize_chat(self, model_id: int, character_name: str) -> None:
        """
        チャットを初期化する
        
        Args:
            model_id: 使用するモデルのID
            character_name: 使用するキャラクター設定の名前
        """
        # 音声チャットクラスの初期化
        self.voice_chat = DirectVoiceChat(
            character_name=character_name,
            character_prompts_dir=config.CHARACTER_PROMPTS_DIR,
            character_common_settings=config.CHARACTER_COMMON_SETTINGS,
            model_manager=self.model_manager
        )
        
        # モデルIDとキャラクター名を設定
        self.voice_chat.model_id = model_id
        self.voice_chat.safetensor_id = self.current_safetensor_id
        
        # 音声履歴の初期化
        self.voice_history = []
        
        logger.info(f"チャットを初期化しました: モデルID={model_id}, キャラクター={character_name}")
    
    async def chat(self, 
                  message: str, 
                  model_id: int, 
                  character_name: str,
                  safetensor_id: int = 0,
                  style: str = None,
                  style_weight: float = None,
                  sdp_ratio: float = None,
                  noise: float = None,
                  noise_w: float = None,
                  length: float = None,
                  line_split: bool = None,
                  split_interval: float = None,
                  assist_text_weight: float = None,
                  volume: float = None,
                  chat_history: List = None,
                  save_audio: bool = False) -> Tuple[List, Optional[str]]:
        """
        チャットメッセージを送信し、応答を取得する
        
        Args:
            message: ユーザーメッセージ
            model_id: 使用するモデルのID
            character_name: 使用するキャラクター設定の名前
            safetensor_id: 使用するsafetensorファイルのID
            style: 音声スタイル
            style_weight: スタイルの重み
            sdp_ratio: SDP比率
            noise: ノイズ
            noise_w: ノイズの重み
            length: 長さ
            line_split: 行分割を行うかどうか
            split_interval: 分割間隔
            assist_text_weight: 補助テキストの重み
            volume: 音量
            chat_history: チャット履歴
            save_audio: 音声を保存するかどうか
            
        Returns:
            Tuple[List, Optional[str]]: チャット履歴と音声ファイルのパス
        """
        # チャット履歴の初期化
        if chat_history is None:
            chat_history = []
        
        # 音声チャットクラスの初期化（必要な場合）
        if self.voice_chat is None or self.voice_chat.model_id != model_id or self.current_safetensor_id != safetensor_id:
            self.current_safetensor_id = safetensor_id
            self.initialize_chat(model_id, character_name)
        
        # ユーザーメッセージをチャット履歴に追加
        chat_history.append({"role": "user", "content": message})
        
        try:
            # チャットボットからの応答を取得
            response = await self.voice_chat.chat(message)
            response_text = response.text
            
            # 応答をチャット履歴に追加
            chat_history.append({"role": "assistant", "content": response_text})
            
            # 音声の生成
            audio_data = await self.voice_chat.text_to_speech(
                text=response_text,
                style=style,
                style_weight=style_weight,
                sdp_ratio=sdp_ratio,
                noise=noise,
                noise_w=noise_w,
                length=length,
                line_split=line_split,
                split_interval=split_interval,
                assist_text_weight=assist_text_weight,
                volume=volume
            )
            
            # 音声の保存
            if audio_data and save_audio:
                self.current_audio = audio_data
                
                # タイムスタンプを取得
                timestamp = datetime.datetime.now()
                date_str = timestamp.strftime("%Y%m%d")
                time_str = timestamp.strftime("%H%M%S")
                
                # モデル名を取得
                model_name = "unknown"
                for model in self.available_models.get('models', []):
                    if model['id'] == model_id:
                        model_name = model['name'].replace(" ", "_")
                        break
                
                # テキストをファイル名用に整形
                safe_text = response_text.strip()
                safe_text = safe_text.replace(" ", "_")
                safe_text = safe_text.replace("/", "_")
                safe_text = safe_text.replace("\\", "_")
                safe_text = safe_text.replace(":", "_")
                safe_text = safe_text.replace("*", "_")
                safe_text = safe_text.replace("?", "_")
                safe_text = safe_text.replace("\"", "_")
                safe_text = safe_text.replace("<", "_")
                safe_text = safe_text.replace(">", "_")
                safe_text = safe_text.replace("|", "_")
                safe_text = safe_text.replace("\n", "_")
                safe_text = safe_text.replace("\r", "_")
                safe_text = safe_text.replace("\t", "_")
                
                # 空の場合はデフォルト名を使用
                if not safe_text:
                    safe_text = "voice"
                
                # ファイル名の長さを制限（最大100文字）
                if len(safe_text) > 100:
                    safe_text = safe_text[:97] + "..."
                
                # 出力ディレクトリを作成
                output_dir = f"outputs/Chat/{date_str}"
                os.makedirs(output_dir, exist_ok=True)
                
                # ファイル名を生成
                filename = f"{safe_text}.wav"
                output_path = os.path.join(output_dir, filename)
                
                # ファイル名が既に存在する場合は連番を付ける
                counter = 1
                base_name = os.path.splitext(filename)[0]
                while os.path.exists(output_path):
                    filename = f"{base_name}_{counter}.wav"
                    output_path = os.path.join(output_dir, filename)
                    counter += 1
                
                # ファイルに保存
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                
                # 履歴データをJSONファイルに保存
                json_filename = os.path.splitext(filename)[0] + ".json"
                json_path = os.path.join(output_dir, json_filename)
                
                # 保存するデータを作成
                metadata = {
                    "text": response_text,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_id": model_id,
                    "model_name": model_name,
                    "safetensor_id": safetensor_id,
                    "character": character_name,
                    "style": style,
                    "style_weight": style_weight,
                    "sdp_ratio": sdp_ratio,
                    "noise": noise,
                    "noise_w": noise_w,
                    "length": length,
                    "line_split": line_split,
                    "split_interval": split_interval,
                    "assist_text_weight": assist_text_weight,
                    "volume": volume,
                    "audio_path": output_path,
                    "user_message": message
                }
                
                # JSONファイルに保存
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                return chat_history, output_path
            
            # 音声データがある場合は一時ファイルに保存して返す
            elif audio_data:
                self.current_audio = audio_data
                
                # 一時ファイルに保存
                temp_dir = "outputs/temp"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, "temp_audio.wav")
                
                with open(temp_path, "wb") as f:
                    f.write(audio_data)
                
                return chat_history, temp_path
            
            # 音声データがない場合
            else:
                return chat_history, None
                
        except Exception as e:
            logger.error(f"チャットエラー: {str(e)}")
            chat_history[-1]["content"] = f"エラーが発生しました: {str(e)}"
            return chat_history, None
    
    def reset_chat(self) -> List:
        """
        チャット履歴をリセットする
        
        Returns:
            List: 空のチャット履歴
        """
        self.voice_chat = None
        return []
    
    async def generate_voice(self, 
                           text: str, 
                           model_id: int, 
                           character_name: str,
                           safetensor_id: int = 0,
                           style: str = None,
                           style_weight: float = None,
                           sdp_ratio: float = None,
                           noise: float = None,
                           noise_w: float = None,
                           length: float = None,
                           line_split: bool = None,
                           split_interval: float = None,
                           assist_text_weight: float = None,
                           volume: float = None,
                           voice_history: List = None) -> Tuple[List, Optional[str]]:
        """
        テキストから音声を生成する
        
        Args:
            text: 音声に変換するテキスト
            model_id: 使用するモデルのID
            character_name: 使用するキャラクター設定の名前
            safetensor_id: 使用するsafetensorファイルのID
            style: 音声スタイル
            style_weight: スタイルの重み
            sdp_ratio: SDP比率
            noise: ノイズ
            noise_w: ノイズの重み
            length: 長さ
            line_split: 行分割を行うかどうか
            split_interval: 分割間隔
            assist_text_weight: 補助テキストの重み
            volume: 音量
            voice_history: 音声履歴
            
        Returns:
            Tuple[List, Optional[str]]: 音声履歴と音声ファイルのパス
        """
        # 音声履歴の初期化
        if voice_history is None:
            voice_history = []
        
        # 音声チャットクラスの初期化（必要な場合）
        if self.voice_chat is None or self.voice_chat.model_id != model_id or self.current_safetensor_id != safetensor_id:
            self.current_safetensor_id = safetensor_id
            self.initialize_chat(model_id, character_name)
        
        try:
            # 音声の生成
            audio_data = await self.voice_chat.text_to_speech(
                text=text,
                style=style,
                style_weight=style_weight,
                sdp_ratio=sdp_ratio,
                noise=noise,
                noise_w=noise_w,
                length=length,
                line_split=line_split,
                split_interval=split_interval,
                assist_text_weight=assist_text_weight,
                volume=volume
            )
            
            # 音声の保存
            if audio_data:
                self.current_audio = audio_data
                
                # タイムスタンプを取得
                timestamp = datetime.datetime.now()
                date_str = timestamp.strftime("%Y%m%d")
                time_str = timestamp.strftime("%H%M%S")
                
                # モデル名を取得
                model_name = "unknown"
                for model in self.available_models.get('models', []):
                    if model['id'] == model_id:
                        model_name = model['name'].replace(" ", "_")
                        break
                
                # キャラクター名とスタイル名を整形
                safe_character_name = character_name.replace(" ", "_")
                safe_style_name = style.replace(" ", "_")
                
                # テキストをファイル名用に整形
                # ファイル名に使えない文字を置換
                safe_text = text.strip()
                safe_text = safe_text.replace(" ", "_")
                safe_text = safe_text.replace("/", "_")
                safe_text = safe_text.replace("\\", "_")
                safe_text = safe_text.replace(":", "_")
                safe_text = safe_text.replace("*", "_")
                safe_text = safe_text.replace("?", "_")
                safe_text = safe_text.replace("\"", "_")
                safe_text = safe_text.replace("<", "_")
                safe_text = safe_text.replace(">", "_")
                safe_text = safe_text.replace("|", "_")
                safe_text = safe_text.replace("\n", "_")
                safe_text = safe_text.replace("\r", "_")
                safe_text = safe_text.replace("\t", "_")
                
                # 空の場合はデフォルト名を使用
                if not safe_text:
                    safe_text = "voice"
                
                # ファイル名の長さを制限（最大100文字）
                if len(safe_text) > 100:
                    safe_text = safe_text[:97] + "..."
                
                # 出力ディレクトリを作成
                output_dir = f"outputs/VoiceGen/{date_str}"
                os.makedirs(output_dir, exist_ok=True)
                
                # ファイル名を生成（シンプルな形式）
                filename = f"{safe_text}.wav"
                output_path = os.path.join(output_dir, filename)
                
                # ファイル名が既に存在する場合は連番を付ける
                counter = 1
                base_name = os.path.splitext(filename)[0]
                while os.path.exists(output_path):
                    filename = f"{base_name}_{counter}.wav"
                    output_path = os.path.join(output_dir, filename)
                    counter += 1
                
                # ファイルに保存
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                
                # 履歴データをJSONファイルに保存
                json_filename = os.path.splitext(filename)[0] + ".json"
                json_path = os.path.join(output_dir, json_filename)
                
                # 保存するデータを作成
                metadata = {
                    "text": text,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_id": model_id,
                    "model_name": model_name,
                    "safetensor_id": safetensor_id,
                    "character": character_name,
                    "style": style,
                    "style_weight": style_weight,
                    "sdp_ratio": sdp_ratio,
                    "noise": noise,
                    "noise_w": noise_w,
                    "length": length,
                    "line_split": line_split,
                    "split_interval": split_interval,
                    "assist_text_weight": assist_text_weight,
                    "volume": volume,
                    "audio_path": output_path
                }
                
                # JSONファイルに保存
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                # 履歴に追加
                self.voice_history.append(metadata)
                
                # 履歴データをDataframeに変換
                history_data = []
                for item in self.voice_history:
                    history_data.append({
                        "text": item.get("text", ""),
                        "timestamp": item.get("timestamp", ""),
                        "model": item.get("model_name", ""),
                        "character": item.get("character", ""),
                        "style": item.get("style", "")
                    })
                
                return history_data, output_path
            
            # 音声データがない場合
            else:
                # エラーメッセージを履歴に追加
                error_data = {
                    "text": text,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model_name,
                    "character": character_name,
                    "style": style,
                    "error": "音声生成に失敗しました"
                }
                self.voice_history.append(error_data)
                
                # 履歴データをDataframeに変換
                history_data = []
                for item in self.voice_history:
                    history_data.append({
                        "text": item.get("text", ""),
                        "timestamp": item.get("timestamp", ""),
                        "model": item.get("model_name", "") if "model_name" in item else item.get("model", ""),
                        "character": item.get("character", ""),
                        "style": item.get("style", "")
                    })
                
                return history_data, None
                
        except Exception as e:
            logger.error(f"音声生成エラー: {str(e)}")
            
            # エラーメッセージを履歴に追加
            error_data = {
                "text": text,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model": model_name if 'model_name' in locals() else "unknown",
                "character": character_name,
                "style": style,
                "error": f"エラー: {str(e)}"
            }
            self.voice_history.append(error_data)
            
            # 履歴データをDataframeに変換
            history_data = []
            for item in self.voice_history:
                history_data.append({
                    "text": item.get("text", ""),
                    "timestamp": item.get("timestamp", ""),
                    "model": item.get("model_name", "") if "model_name" in item else item.get("model", ""),
                    "character": item.get("character", ""),
                    "style": item.get("style", "")
                })
            
            return history_data, None
            
    def reset_voice_history(self) -> List:
        """
        音声生成履歴をリセットする
        
        Returns:
            空の音声生成履歴
        """
        self.voice_history = []
        return []
        
    def get_output_directories(self) -> List[str]:
        """
        outputsディレクトリ内の年月日ディレクトリのリストを取得する
        
        Returns:
            年月日ディレクトリのリスト
        """
        try:
            # VoiceGenディレクトリのパス
            voice_gen_dir = os.path.join("outputs", "VoiceGen")
            if not os.path.exists(voice_gen_dir):
                os.makedirs(voice_gen_dir, exist_ok=True)
                return []
                
            # VoiceGenディレクトリ内のサブディレクトリを取得
            dirs = [d for d in os.listdir(voice_gen_dir) if os.path.isdir(os.path.join(voice_gen_dir, d))]
            
            # 年月日形式（YYYYMMDD）のディレクトリのみをフィルタリング
            date_dirs = []
            for d in dirs:
                if len(d) == 8 and d.isdigit():
                    try:
                        # 正しい日付形式かチェック
                        year = int(d[:4])
                        month = int(d[4:6])
                        day = int(d[6:8])
                        if 1 <= month <= 12 and 1 <= day <= 31:
                            # 日付を整形して表示用にする
                            formatted_date = f"{year}/{str(month).zfill(2)}/{str(day).zfill(2)}"
                            date_dirs.append((d, formatted_date))
                    except ValueError:
                        continue
            
            # 日付の新しい順にソート
            date_dirs.sort(reverse=True)
            return date_dirs
        except Exception as e:
            logger.error(f"出力ディレクトリの取得中にエラーが発生しました: {e}")
            return []
    
    def load_voice_history_from_directory(self, date_dir: str) -> List[Dict]:
        """
        指定された年月日ディレクトリから音声生成履歴を読み込む
        
        Args:
            date_dir: 年月日ディレクトリ名（YYYYMMDD形式）
            
        Returns:
            音声生成履歴のリスト
        """
        history_list = []
        try:
            dir_path = os.path.join("outputs", "VoiceGen", date_dir)
            if not os.path.exists(dir_path):
                return []
                
            # ディレクトリ内のJSONファイルを取得
            json_files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
            
            for json_file in json_files:
                try:
                    # JSONファイルからメタデータを読み込む
                    json_path = os.path.join(dir_path, json_file)
                    with open(json_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    # 対応する音声ファイルのパスを確認
                    audio_path = metadata.get("audio_path")
                    if not os.path.exists(audio_path):
                        # パスが絶対パスで保存されている場合、相対パスに変換
                        wav_file = os.path.splitext(json_file)[0] + ".wav"
                        audio_path = os.path.join(dir_path, wav_file)
                        if not os.path.exists(audio_path):
                            logger.warning(f"音声ファイルが見つかりません: {audio_path}")
                            continue
                    
                    # 履歴エントリを作成
                    entry = {
                        "text": metadata.get("text", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "model": metadata.get("model_name", ""),
                        "character": metadata.get("character", ""),
                        "style": metadata.get("style", ""),
                        "audio_path": audio_path,
                        "json_path": json_path
                    }
                    
                    history_list.append(entry)
                except Exception as e:
                    logger.warning(f"JSONファイル {json_file} の解析中にエラーが発生しました: {e}")
                    continue
            
            # JSONファイルが見つからない場合は、従来の方法でWAVファイルから情報を抽出
            if not history_list:
                wav_files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]
                
                for wav_file in wav_files:
                    try:
                        # ファイル名からテキストと日付を抽出
                        # 形式: {テキスト}_{年月日}.wav
                        parts = wav_file.split("_")
                        if len(parts) >= 2:
                            # 最後の部分は日付
                            date_str = parts[-1].split(".")[0]  # .wavを除去
                            
                            # 残りの部分はテキスト
                            text_parts = parts[:-1]
                            text = "_".join(text_parts)
                            
                            # 音声ファイルのパス
                            audio_path = os.path.join(dir_path, wav_file)
                            
                            # 履歴エントリを作成
                            entry = {
                                "text": text,
                                "timestamp": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00",
                                "model": "unknown",
                                "character": "unknown",
                                "style": "unknown",
                                "audio_path": audio_path
                            }
                            
                            history_list.append(entry)
                    except Exception as e:
                        logger.warning(f"ファイル {wav_file} の解析中にエラーが発生しました: {e}")
                        continue
            
            # タイムスタンプでソート
            history_list.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # インスタンスの履歴を更新
            self.voice_history = history_list
            
            return history_list
        except Exception as e:
            logger.error(f"履歴の読み込み中にエラーが発生しました: {e}")
            return []

def create_ui() -> gr.Blocks:
    """
    GradioのUIを作成する
    
    Returns:
        Gradio Blocksインスタンス
    """
    # WebUIインスタンスの作成
    webui = VerbalisWebUI()
    
    # モデル選択肢の作成
    model_choices = {f"{model['name']} ({model['id']})": model['id'] for model in webui.available_models.get('models', [])}
    if not model_choices:
        model_choices = {"デフォルトモデル": 0}
    
    # キャラクター選択肢の作成
    character_choices = list(webui.characters.keys())
    if not character_choices:
        character_choices = ["デフォルト"]
    
    # デフォルトモデルのスタイル選択肢
    default_model_id = webui.available_models.get('default_model_id', 0)
    default_styles = webui.model_styles.get(default_model_id, ["Neutral"])
    
    # デフォルトモデルのsafetensorファイル選択肢
    default_safetensors = []
    for model in webui.available_models.get('models', []):
        if model['id'] == default_model_id:
            default_safetensors = [(sf['name'], sf['id']) for sf in model.get('safetensors_files', [])]
            break
    
    with gr.Blocks(title="Verbalis", css=STYLE_CSS) as demo:
        gr.Markdown("# Verbalis")
        
        # タブ切り替えシステムの追加
        with gr.Tabs() as tabs:
            # Chatタブ
            with gr.TabItem("Chat"):
                with gr.Row():
                    # 左側のチャットエリア
                    with gr.Column(scale=3):
                        # チャットボットコンポーネント
                        chatbot = gr.Chatbot(
                            label="Chat",
                            height=683,
                            resizable=True,
                            autoscroll=True,
                            type="messages",
                            show_copy_button=False,
                            show_share_button=False,
                            render_markdown=True,
                            show_label=True
                        )             

                        # チャット入力エリア（テキストボックスと送信ボタン）
                        with gr.Row(equal_height=True, elem_classes="chat-input-container"):
                            msg = gr.Textbox(
                                placeholder="ここにメッセージを入力してください...",
                                lines=1,
                                show_label=False,
                                container=False,
                                elem_classes="message-box",
                                scale=4
                            )
                            send_btn = gr.Button("送信", variant="primary", elem_classes="send-button")
                        
                        reset_btn = gr.Button("チャット履歴をリセット")
                        
                        audio_player = gr.Audio(
                            label="音声プレイヤー",
                            type="filepath",
                            interactive=False,
                            elem_id="audio_player",
                            elem_classes="audio-player",
                            autoplay=True
                        )
                    
                    # 右側の設定エリア
                    with gr.Column(scale=1):
                        with gr.Group():
                            model_dropdown = gr.Dropdown(
                                label="モデル選択",
                                choices=list(model_choices.keys()),
                                value=list(model_choices.keys())[0] if model_choices else None
                            )
                            
                            # safetensorファイル選択ドロップダウンを追加
                            safetensor_dropdown = gr.Dropdown(
                                label="safetensorファイル選択",
                                choices=[sf[0] for sf in default_safetensors],
                                value=default_safetensors[0][0] if default_safetensors else None,
                                visible=True if default_safetensors else False
                            )
                            
                            character_dropdown = gr.Dropdown(
                                label="キャラクター選択",
                                choices=character_choices,
                                value=config.DEFAULT_CHARACTER if config.DEFAULT_CHARACTER in character_choices else (character_choices[0] if character_choices else None)
                            )

                            style_dropdown = gr.Dropdown(
                                label="スタイル",
                                choices=default_styles,
                                value=default_styles[0] if default_styles else config.DEFAULT_STYLE
                            )
                            
                            style_weight_slider = gr.Slider(
                                label="スタイルの重み",
                                minimum=0.0,
                                maximum=2.0,
                                value=config.DEFAULT_STYLE_WEIGHT,
                                step=0.1
                            )

                            volume_slider = gr.Slider(
                                label="音量",
                                minimum=0.0,
                                maximum=1.0,
                                value=config.DEFAULT_VOLUME,
                                step=0.1
                            )
                        with gr.Group():
                            sdp_ratio_slider = gr.Slider(
                                label="SDP比率",
                                minimum=0.0,
                                maximum=1.0,
                                value=config.DEFAULT_SDP_RATIO,
                                step=0.1
                            )
                            
                            noise_slider = gr.Slider(
                                label="ノイズ",
                                minimum=0.0,
                                maximum=1.0,
                                value=config.DEFAULT_NOISE,
                                step=0.1
                            )
                            
                            noise_w_slider = gr.Slider(
                                label="ノイズの重み",
                                minimum=0.0,
                                maximum=1.0,
                                value=config.DEFAULT_NOISEW,
                                step=0.1
                            )
                            
                            length_slider = gr.Slider(
                                label="長さ",
                                minimum=0.1,
                                maximum=2.0,
                                value=config.DEFAULT_LENGTH,
                                step=0.1
                            )

                            line_split_checkbox = gr.Checkbox(
                                label="自動分割",
                                value=config.DEFAULT_LINE_SPLIT
                            )
                            
                            split_interval_slider = gr.Slider(
                                label="分割間隔",
                                minimum=0.1,
                                maximum=5.0,
                                value=config.DEFAULT_SPLIT_INTERVAL,
                                step=0.1
                            )
                            
                            assist_text_weight_slider = gr.Slider(
                                label="補助テキストの重み",
                                minimum=0.0,
                                maximum=1.0,
                                value=config.DEFAULT_ASSIST_TEXT_WEIGHT,
                                step=0.1
                            )
                            
                            save_audio_checkbox = gr.Checkbox(
                                label="音声を保存する",
                                value=False
                            )
            
            # VoiceGenタブ
            with gr.TabItem("VoiceGen"):
                with gr.Row():
                    # 左側の音声生成エリア
                    with gr.Column(scale=3):
                        voice_text_input = gr.Textbox(
                            label="テキスト入力",
                            placeholder="ここに音声に変換するテキストを入力してください...",
                            lines=5
                        )
                        
                        generate_btn = gr.Button("音声生成", variant="primary")
                        
                        voice_audio_player = gr.Audio(
                            label="音声プレイヤー",
                            type="filepath",
                            interactive=False,
                            elem_id="voice_audio_player",
                            elem_classes="audio-player",
                            autoplay=True
                        )
                        
                        reset_voice_history_btn = gr.Button("音声履歴をリセット")
                    
                    # 右側の設定エリア
                    with gr.Column(scale=1):
                        with gr.Group():
                            voice_model_dropdown = gr.Dropdown(
                                label="モデル選択",
                                choices=list(model_choices.keys()),
                                value=list(model_choices.keys())[0] if model_choices else None
                            )
                            
                            # safetensorファイル選択ドロップダウンを追加
                            voice_safetensor_dropdown = gr.Dropdown(
                                label="safetensorファイル選択",
                                choices=[sf[0] for sf in default_safetensors],
                                value=default_safetensors[0][0] if default_safetensors else None,
                                visible=True if default_safetensors else False
                            )
                            
                            voice_character_dropdown = gr.Dropdown(
                                label="キャラクター選択",
                                choices=character_choices,
                                value=config.DEFAULT_CHARACTER if config.DEFAULT_CHARACTER in character_choices else (character_choices[0] if character_choices else None)
                            )
                            
                            voice_style_dropdown = gr.Dropdown(
                                label="スタイル",
                                choices=default_styles,
                                value=default_styles[0] if default_styles else config.DEFAULT_STYLE
                            )
                            
                            voice_style_weight_slider = gr.Slider(
                                label="スタイルの重み",
                                minimum=0.0,
                                maximum=2.0,
                                value=config.DEFAULT_STYLE_WEIGHT,
                                step=0.1
                            )
                            
                            voice_volume_slider = gr.Slider(
                                label="音量",
                                minimum=0.0,
                                maximum=1.0,
                                value=config.DEFAULT_VOLUME,
                                step=0.1
                            )
                        
                        with gr.Group():
                            voice_sdp_ratio_slider = gr.Slider(
                                label="SDP比率",
                                minimum=0.0,
                                maximum=1.0,
                                value=config.DEFAULT_SDP_RATIO,
                                step=0.1
                            )
                            
                            voice_noise_slider = gr.Slider(
                                label="ノイズ",
                                minimum=0.0,
                                maximum=1.0,
                                value=config.DEFAULT_NOISE,
                                step=0.1
                            )
                            
                            voice_noise_w_slider = gr.Slider(
                                label="ノイズの重み",
                                minimum=0.0,
                                maximum=1.0,
                                value=config.DEFAULT_NOISEW,
                                step=0.1
                            )
                            
                            voice_length_slider = gr.Slider(
                                label="長さ",
                                minimum=0.1,
                                maximum=2.0,
                                value=config.DEFAULT_LENGTH,
                                step=0.1
                            )
                            
                            voice_line_split_checkbox = gr.Checkbox(
                                label="自動分割",
                                value=config.DEFAULT_LINE_SPLIT
                            )
                            
                            voice_split_interval_slider = gr.Slider(
                                label="分割間隔",
                                minimum=0.1,
                                maximum=5.0,
                                value=config.DEFAULT_SPLIT_INTERVAL,
                                step=0.1
                            )
                            
                            voice_assist_text_weight_slider = gr.Slider(
                                label="補助テキストの重み",
                                minimum=0.0,
                                maximum=1.0,
                                value=config.DEFAULT_ASSIST_TEXT_WEIGHT,
                                step=0.1
                            )
                
                # 履歴表示エリア
                with gr.Row():
                    with gr.Column():
                        # 履歴ディレクトリ選択
                        date_dirs = webui.get_output_directories()
                        date_dir_dropdown = gr.Dropdown(
                            label="履歴ディレクトリ",
                            choices=date_dirs,
                            value=date_dirs[0] if date_dirs else None,
                            type="index"
                        )
                        
                        show_history_btn = gr.Button("履歴を表示")
                
                # 更新メッセージ表示用
                refresh_message = gr.Markdown("", visible=True)
                
                # 空の初期データ
                empty_history_data = []
                
                voice_history_display = gr.Dataframe(
                    headers=["テキスト", "生成日時", "モデル", "キャラクター", "スタイル"],
                    datatype=["str", "str", "str", "str", "str"],
                    col_count=(5, "fixed"),
                    row_count=(10, "dynamic"),
                    interactive=False,
                    elem_id="voice_history_display",
                    elem_classes="dataframe-container",
                    label="行をクリックすることで、音声を再生できます。",
                    wrap=True,
                    column_widths=["40%", "15%", "15%", "15%", "15%"],
                    value=empty_history_data
                )
        
        # モデル選択時にsafetensorファイルとスタイル選択肢を更新する関数
        def update_safetensor_choices(model_dropdown_value):
            if not model_dropdown_value:
                return gr.update(choices=[], value=None, visible=False), gr.update(choices=["Neutral"], value="Neutral")
            
            model_id = model_choices[model_dropdown_value]
            
            # safetensorファイルのリストを取得
            safetensors = webui.get_safetensors_for_model(model_id)
            safetensor_choices = [sf['name'] for sf in safetensors]
            
            # スタイルのリストを取得
            styles = webui.model_styles.get(model_id, ["Neutral"])
            
            # safetensorファイルがある場合は表示、ない場合は非表示
            if safetensor_choices:
                return gr.update(choices=safetensor_choices, value=safetensor_choices[0], visible=True), gr.update(choices=styles, value=styles[0] if styles else "Neutral")
            else:
                return gr.update(choices=[], value=None, visible=False), gr.update(choices=styles, value=styles[0] if styles else "Neutral")
        
        # safetensor選択時にスタイル選択肢を更新する関数
        def update_style_on_safetensor_change(model_dropdown_value, safetensor_name):
            if not model_dropdown_value or not safetensor_name:
                return gr.update(choices=["Neutral"], value="Neutral")
            
            model_id = model_choices[model_dropdown_value]
            
            # safetensorのIDを取得
            safetensor_id = 0
            safetensors = webui.get_safetensors_for_model(model_id)
            for sf in safetensors:
                if sf['name'] == safetensor_name:
                    safetensor_id = sf['id']
                    break
            
            # safetensor_idを更新
            webui.update_safetensor_id(model_id, safetensor_id)
            
            # 更新されたスタイルのリストを取得
            styles = webui.model_styles.get(model_id, ["Neutral"])
            
            return gr.update(choices=styles, value=styles[0] if styles else "Neutral")
        
        # モデル選択変更時のイベント
        model_dropdown.change(
            fn=update_safetensor_choices,
            inputs=[model_dropdown],
            outputs=[safetensor_dropdown, style_dropdown]
        )
        
        # safetensor選択変更時のイベント
        safetensor_dropdown.change(
            fn=update_style_on_safetensor_change,
            inputs=[model_dropdown, safetensor_dropdown],
            outputs=[style_dropdown]
        )
        
        # VoiceGenタブのモデル選択変更時のイベント
        voice_model_dropdown.change(
            fn=update_safetensor_choices,
            inputs=[voice_model_dropdown],
            outputs=[voice_safetensor_dropdown, voice_style_dropdown]
        )
        
        # VoiceGenタブのsafetensor選択変更時のイベント
        voice_safetensor_dropdown.change(
            fn=update_style_on_safetensor_change,
            inputs=[voice_model_dropdown, voice_safetensor_dropdown],
            outputs=[voice_style_dropdown]
        )
        
        # イベントハンドラの設定
        async def on_submit(message, chat_history, model_dropdown, safetensor_dropdown, character_dropdown, style, style_weight, 
                           sdp_ratio, noise, noise_w, length, line_split, split_interval, assist_text_weight, volume, save_audio):
            if not message:
                return chat_history, None, ""
            
            model_id = model_choices[model_dropdown]
            
            # safetensor_idを取得
            safetensor_id = 0
            if safetensor_dropdown:
                safetensors = webui.get_safetensors_for_model(model_id)
                for sf in safetensors:
                    if sf['name'] == safetensor_dropdown:
                        safetensor_id = sf['id']
                        break
            
            result = await webui.chat(
                message=message,
                model_id=model_id,
                character_name=character_dropdown,
                safetensor_id=safetensor_id,
                style=style,
                style_weight=style_weight,
                sdp_ratio=sdp_ratio,
                noise=noise,
                noise_w=noise_w,
                length=length,
                line_split=line_split,
                split_interval=split_interval,
                assist_text_weight=assist_text_weight,
                volume=volume,
                chat_history=chat_history,
                save_audio=save_audio
            )
            
            # テキストボックスをクリア
            return result[0], result[1], ""
        
        # テキストボックスのサブミットイベント（Enterキーで送信）
        msg.submit(
            fn=on_submit,
            inputs=[
                msg, chatbot, model_dropdown, safetensor_dropdown, character_dropdown,
                style_dropdown, style_weight_slider, sdp_ratio_slider,
                noise_slider, noise_w_slider, length_slider,
                line_split_checkbox, split_interval_slider,
                assist_text_weight_slider, volume_slider, save_audio_checkbox
            ],
            outputs=[chatbot, audio_player, msg]
        )
        
        # 送信ボタンのクリックイベント
        send_btn.click(
            fn=on_submit,
            inputs=[
                msg, chatbot, model_dropdown, safetensor_dropdown, character_dropdown,
                style_dropdown, style_weight_slider, sdp_ratio_slider,
                noise_slider, noise_w_slider, length_slider,
                line_split_checkbox, split_interval_slider,
                assist_text_weight_slider, volume_slider, save_audio_checkbox
            ],
            outputs=[chatbot, audio_player, msg]
        )
        
        # リセットボタンのクリックイベント
        reset_btn.click(
            fn=webui.reset_chat,
            inputs=[],
            outputs=[chatbot]
        )
        
        # VoiceGenタブのイベントハンドラ
        async def on_generate_voice(text, voice_history, model_dropdown, safetensor_dropdown, character_dropdown, style, style_weight, 
                                   sdp_ratio, noise, noise_w, length, line_split, split_interval, assist_text_weight, volume):
            if not text:
                return voice_history, None
            
            model_id = model_choices[model_dropdown]
            
            # safetensor_idを取得
            safetensor_id = 0
            if safetensor_dropdown:
                safetensors = webui.get_safetensors_for_model(model_id)
                for sf in safetensors:
                    if sf['name'] == safetensor_dropdown:
                        safetensor_id = sf['id']
                        break
            
            result = await webui.generate_voice(
                text=text,
                model_id=model_id,
                character_name=character_dropdown,
                safetensor_id=safetensor_id,
                style=style,
                style_weight=style_weight,
                sdp_ratio=sdp_ratio,
                noise=noise,
                noise_w=noise_w,
                length=length,
                line_split=line_split,
                split_interval=split_interval,
                assist_text_weight=assist_text_weight,
                volume=volume,
                voice_history=voice_history
            )
            
            # 履歴をDataframeに変換
            df_data = []
            for item in result[0]:
                # エラーがある場合はエラーメッセージを表示
                if "error" in item:
                    df_data.append([
                        item.get("text", ""), 
                        item.get("timestamp", ""),
                        item.get("model", ""),
                        item.get("character", ""), 
                        item.get("style", "")
                    ])
                else:
                    df_data.append([
                        item.get("text", ""), 
                        item.get("timestamp", ""),
                        item.get("model", ""),
                        item.get("character", ""), 
                        item.get("style", "")
                    ])
            
            return df_data, result[1]
        
        # 音声生成ボタンのクリックイベント
        generate_btn.click(
            fn=on_generate_voice,
            inputs=[
                voice_text_input, voice_history_display, voice_model_dropdown, voice_safetensor_dropdown, voice_character_dropdown,
                voice_style_dropdown, voice_style_weight_slider, voice_sdp_ratio_slider,
                voice_noise_slider, voice_noise_w_slider, voice_length_slider,
                voice_line_split_checkbox, voice_split_interval_slider,
                voice_assist_text_weight_slider, voice_volume_slider
            ],
            outputs=[voice_history_display, voice_audio_player]
        )
        
        # 音声履歴リセットボタンのクリックイベント
        reset_voice_history_btn.click(
            fn=webui.reset_voice_history,
            inputs=[],
            outputs=[voice_history_display]
        )
        
        # 履歴から音声を再生する関数
        def play_from_history(evt: gr.SelectData, voice_history):
            row_idx = evt.index[0]
            # 履歴データから対応する音声ファイルを探す
            try:
                # 現在の履歴データを取得
                history_list = []
                for i in range(len(voice_history)):
                    history_list.append({
                        "text": voice_history.iloc[i, 0] if len(voice_history.columns) > 0 else "",
                        "timestamp": voice_history.iloc[i, 1] if len(voice_history.columns) > 1 else "",
                        "model": voice_history.iloc[i, 2] if len(voice_history.columns) > 2 else "",
                        "character": voice_history.iloc[i, 3] if len(voice_history.columns) > 3 else "",
                        "style": voice_history.iloc[i, 4] if len(voice_history.columns) > 4 else ""
                    })
                
                # 対応する音声ファイルを探す
                for item in webui.voice_history:
                    if (item.get("text") == history_list[row_idx].get("text") and 
                        item.get("timestamp") == history_list[row_idx].get("timestamp")):
                        return item.get("audio_path")
            except Exception as e:
                logger.error(f"履歴からの再生中にエラーが発生しました: {e}")
            return None
        
        # 履歴からの再生イベント
        voice_history_display.select(
            fn=play_from_history,
            inputs=[voice_history_display],
            outputs=[voice_audio_player]
        )
        
        # 年月日ディレクトリの選択が変更されたときの処理
        def on_date_dir_change(date_dir_idx):
            if date_dir_idx is None:
                return []
            
            date_dirs = webui.get_output_directories()
            if not date_dirs or date_dir_idx >= len(date_dirs):
                return []
            
            # 選択された年月日ディレクトリから履歴を読み込む
            date_dir = date_dirs[date_dir_idx][0]
            history_list = webui.load_voice_history_from_directory(date_dir)
            
            # DataFrameに表示するデータを作成
            df_data = []
            for item in history_list:
                df_data.append([
                    item.get("text", ""), 
                    item.get("timestamp", ""),
                    item.get("model", ""),
                    item.get("character", ""), 
                    item.get("style", "")
                ])
            
            return df_data
        
        # 年月日ディレクトリの選択が変更されたときのイベント
        show_history_btn.click(
            fn=on_date_dir_change,
            inputs=[date_dir_dropdown],
            outputs=[voice_history_display]
        )
    
    return demo

def parse_args():
    """
    コマンドライン引数を解析する
    
    Returns:
        解析された引数
    """
    parser = argparse.ArgumentParser(description="Verbalis WebUI")
    parser.add_argument("--host", type=str, default=config.HOST, help="ホストアドレス")
    parser.add_argument("--port", type=int, default=config.PORT, help="ポート番号")
    parser.add_argument("--share", action="store_true", help="Gradio共有リンクを生成する")
    parser.add_argument("--gpu", action="store_true", help="GPUを使用する（設定ファイルの値を上書き）")
    args = parser.parse_args()
    
    # GPUの設定を上書き
    if args.gpu:
        config.USE_GPU = True
    
    return args

if __name__ == "__main__":
    # コマンドライン引数の解析
    args = parse_args()
    
    # BERTモデルの読み込み
    try:
        logger.info(f"BERTモデル {config.BERT_MODEL_NAME} を読み込んでいます...")
        load_bert_models(Languages.JP, config.BERT_MODEL_NAME)
        logger.info("BERTモデルの読み込みが完了しました。")
    except Exception as e:
        logger.error(f"BERTモデルの読み込みに失敗しました: {e}")
        logger.error("WebUIは起動しますが、音声合成が正常に動作しない可能性があります。")
    
    # UIの作成と起動
    logger.info("Verbalis WebUIを起動しています...")
    demo = create_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share) 