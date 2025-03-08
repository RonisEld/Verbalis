"""
音声チャットモジュール

このモジュールはGemini APIとStyle-BERT-VITS2を組み合わせた音声チャットの基本機能を提供します。
"""

import os
import sys
import json
import asyncio
import logging
import simpleaudio as sa
import io
import numpy as np
import wave
import google.generativeai as genai
from typing import Dict, List, Optional, Any, Union, Tuple
import requests

# ロギングの設定
logger = logging.getLogger(__name__)

class BaseVoiceChat:
    """
    音声チャットの基本クラス
    
    このクラスはGemini APIを使用したテキスト応答生成と
    音声再生の基本機能を提供します。
    """
    
    def __init__(self, character_name: str, character_prompts_dir: str, character_common_settings: str):
        """
        BaseVoiceChatクラスの初期化
        
        Args:
            character_name: 使用するキャラクター設定の名前
            character_prompts_dir: キャラクタープロンプトファイルのディレクトリ
            character_common_settings: 共通キャラクター設定
        """
        self.character_name = character_name
        self.character_prompts_dir = character_prompts_dir
        self.character_common_settings = character_common_settings
        
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
        
        # 音声設定のデフォルト値
        self.speaker_id = 0
        self.model_id = 0
        self.volume = 1.0
    
    def _load_character_prompt(self, character_name: str) -> str:
        """
        キャラクタープロンプトファイルを読み込む
        
        Args:
            character_name: キャラクター設定の名前
            
        Returns:
            str: キャラクタープロンプトの内容（共通設定を含む）
        """
        prompt_path = os.path.join(self.character_prompts_dir, f"{character_name}.txt")
        
        # ファイルが存在しない場合はデフォルトを使用
        if not os.path.exists(prompt_path):
            logger.warning(f"キャラクター '{character_name}' が見つかりません。デフォルトを使用します。")
            return self._load_default_prompt()
            
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                character_prompt = f.read().strip()
            # 共通設定とキャラクタープロンプトを組み合わせる
            combined_prompt = f"【共通設定】\n{self.character_common_settings.strip()}\n\n【個別設定】\n{character_prompt}"
            return combined_prompt
            
        except Exception as e:
            logger.error(f"キャラクタープロンプトの読み込みに失敗しました: {str(e)}")
            return self._load_default_prompt()
    
    def _load_default_prompt(self) -> str:
        """
        デフォルトのキャラクタープロンプトを返す
        
        Returns:
            str: デフォルトのキャラクタープロンプト
        """
        # デフォルトのプロンプト
        default_prompt = "あなたは親切で丁寧な日本語AIアシスタントです。"
        return f"【共通設定】\n{self.character_common_settings.strip()}\n\n【個別設定】\n{default_prompt}"
    
    async def generate_response(self, user_input: str) -> str:
        """
        ユーザー入力に対する応答を生成する
        
        Args:
            user_input: ユーザーの入力テキスト
            
        Returns:
            str: Gemini APIからの応答テキスト
        """
        try:
            # 応答を生成
            response = await asyncio.to_thread(
                self.chat_session.send_message,
                user_input
            )
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini APIからの応答生成に失敗しました: {str(e)}")
            return "申し訳ありません。応答の生成中にエラーが発生しました。"
    
    async def play_audio(self, audio_data: bytes) -> None:
        """
        音声データを再生する
        
        Args:
            audio_data: WAV形式の音声データ（バイト列）
        """
        try:
            # 音量調整
            if self.volume != 1.0:
                audio_data = self._adjust_volume(audio_data, self.volume)
            
            # WAVデータを読み込む
            wave_read = wave.open(io.BytesIO(audio_data), 'rb')
            audio = wave_read.readframes(wave_read.getnframes())
            
            # 音声を再生
            play_obj = sa.play_buffer(
                audio,
                num_channels=wave_read.getnchannels(),
                bytes_per_sample=wave_read.getsampwidth(),
                sample_rate=wave_read.getframerate()
            )
            
            # 再生が終わるまで待機
            play_obj.wait_done()
            
        except Exception as e:
            logger.error(f"音声再生エラー: {str(e)}")
    
    def _adjust_volume(self, audio_data: bytes, volume: float) -> bytes:
        """
        音声データの音量を調整する
        
        Args:
            audio_data: WAV形式の音声データ（バイト列）
            volume: 音量倍率（1.0が原音量）
            
        Returns:
            bytes: 音量調整後の音声データ
        """
        try:
            # WAVデータを読み込む
            with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                # WAVパラメータを取得
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                
                # 音声データを読み込む
                frames = wf.readframes(n_frames)
            
            # バイトデータをnumpy配列に変換
            if sampwidth == 2:  # 16-bit
                dtype = np.int16
                max_value = 32767
            elif sampwidth == 1:  # 8-bit
                dtype = np.uint8
                max_value = 255
            elif sampwidth == 4:  # 32-bit
                dtype = np.int32
                max_value = 2147483647
            else:
                raise ValueError(f"サポートされていないサンプル幅: {sampwidth}")
                
            # バイトデータをnumpy配列に変換
            audio_array = np.frombuffer(frames, dtype=dtype)
            
            # 音量を調整
            audio_array = audio_array * volume
            
            # クリッピングを防ぐ
            audio_array = np.clip(audio_array, -max_value, max_value).astype(dtype)
            
            # numpy配列をバイトデータに戻す
            adjusted_frames = audio_array.tobytes()
            
            # 新しいWAVファイルを作成
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(n_channels)
                wf.setsampwidth(sampwidth)
                wf.setframerate(framerate)
                wf.writeframes(adjusted_frames)
                
            # バイトデータを返す
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            logger.error(f"音量調整エラー: {str(e)}")
            return audio_data  # エラーが発生した場合は元の音声データを返す

class DirectVoiceChat(BaseVoiceChat):
    """
    Style-BERT-VITS2を直接使用した音声チャットクラス
    
    このクラスはStyle-BERT-VITS2モデルを直接使用して音声合成を行います。
    """
    
    def __init__(self, character_name: str, character_prompts_dir: str, character_common_settings: str, model_manager):
        """
        DirectVoiceChatクラスの初期化
        
        Args:
            character_name: 使用するキャラクター設定の名前
            character_prompts_dir: キャラクタープロンプトファイルのディレクトリ
            character_common_settings: 共通キャラクター設定
            model_manager: モデル管理クラスのインスタンス
        """
        super().__init__(character_name, character_prompts_dir, character_common_settings)
        self.model_manager = model_manager
        
        # デフォルトのTTS設定
        self.style = "Neutral"
        self.style_weight = 0.7
        self.sdp_ratio = 0.2
        self.noise = 0.6
        self.noise_w = 0.8
        self.length = 1.0
        self.line_split = True
        self.split_interval = 0.5
        self.assist_text_weight = 0.7
        self.volume = 1.0
    
    async def text_to_speech(self, text: str, style: str = None, style_weight: float = None, 
                           sdp_ratio: float = None, noise: float = None, noise_w: float = None, 
                           length: float = None, line_split: bool = None, split_interval: float = None,
                           assist_text_weight: float = None, volume: float = None) -> Optional[bytes]:
        """
        テキストを音声に変換する
        
        Args:
            text: 音声に変換するテキスト
            style: スタイル名
            style_weight: スタイルの重み
            sdp_ratio: SDP比率
            noise: ノイズ量
            noise_w: ノイズ幅
            length: 音声の長さ
            line_split: 自動分割するかどうか
            split_interval: 分割間隔
            assist_text_weight: 補助テキストの重み
            volume: 音量
            
        Returns:
            Optional[bytes]: WAV形式の音声データ、エラー時はNone
        """
        try:
            # パラメータが指定されていない場合はデフォルト値を使用
            style = style or self.style
            style_weight = style_weight or self.style_weight
            sdp_ratio = sdp_ratio or self.sdp_ratio
            noise = noise or self.noise
            noise_w = noise_w or self.noise_w
            length = length or self.length
            line_split = line_split if line_split is not None else self.line_split
            split_interval = split_interval or self.split_interval
            assist_text_weight = assist_text_weight or self.assist_text_weight
            volume = volume if volume is not None else self.volume
            
            # モデルを取得
            tts_model = self.model_manager.get_direct_model(self.model_id)
            if tts_model is None:
                logger.error(f"モデルの取得に失敗しました (ID: {self.model_id})")
                return None
            
            # テキストから音声を生成
            audio_data = await tts_model.tts(
                text=text,
                speaker_id=self.speaker_id,
                style=style,
                style_weight=style_weight,
                sdp_ratio=sdp_ratio,
                noise=noise,
                noise_w=noise_w,
                length=length,
                line_split=line_split,
                split_interval=split_interval,
                assist_text_weight=assist_text_weight
            )
            
            # 音量調整
            if volume != 1.0 and audio_data:
                audio_data = self._adjust_volume(audio_data, volume)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"音声合成エラー: {str(e)}")
            return None

class APIVoiceChat(BaseVoiceChat):
    """
    Style-BERT-VITS2 APIを使用した音声チャットクラス
    
    このクラスはStyle-BERT-VITS2 APIサーバーを使用して音声合成を行います。
    """
    
    def __init__(self, character_name: str, character_prompts_dir: str, character_common_settings: str, api_host: str, api_port: int):
        """
        APIVoiceChatクラスの初期化
        
        Args:
            character_name: 使用するキャラクター設定の名前
            character_prompts_dir: キャラクタープロンプトファイルのディレクトリ
            character_common_settings: 共通キャラクター設定
            api_host: APIサーバーのホスト
            api_port: APIサーバーのポート
        """
        super().__init__(character_name, character_prompts_dir, character_common_settings)
        self.api_host = api_host
        self.api_port = api_port
        
        # 利用可能なモデルを取得
        self.available_models = self._get_available_models()
        
        # デフォルトのTTS設定
        self.style = "Neutral"
        self.style_weight = 0.7
        self.sdp_ratio = 0.2
        self.noise = 0.6
        self.noise_w = 0.8
        self.length = 1.0
        self.line_split = True
        self.split_interval = 0.5
        self.assist_text_weight = 0.7
        self.volume = 1.0
    
    def _get_available_models(self) -> Dict:
        """
        利用可能な音声モデルの一覧を取得する
        
        Returns:
            Dict: モデル情報の辞書
        """
        try:
            # APIサーバーに接続を試みる
            max_port = self.api_port + 10  # 最大10ポート試す
            current_port = self.api_port
            
            while current_port <= max_port:
                try:
                    api_url = f"http://{self.api_host}:{current_port}/models"
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
            raise requests.RequestException(f"すべてのポート（{self.api_port}〜{max_port}）への接続に失敗しました。")
            
        except requests.RequestException as e:
            logger.error(f"音声モデルの取得に失敗しました。Style-BERT-VITS2 APIサーバーが起動しているか確認してください。")
            logger.error(f"詳細: {str(e)}")
            return {"models": [], "default_model_id": 0}
    
    async def text_to_speech(self, text: str, style: str = None, style_weight: float = None, 
                           sdp_ratio: float = None, noise: float = None, noise_w: float = None, 
                           length: float = None, line_split: bool = None, split_interval: float = None,
                           assist_text_weight: float = None, volume: float = None) -> Optional[bytes]:
        """
        テキストを音声に変換する
        
        Args:
            text: 音声に変換するテキスト
            style: スタイル名
            style_weight: スタイルの重み
            sdp_ratio: SDP比率
            noise: ノイズ量
            noise_w: ノイズ幅
            length: 音声の長さ
            line_split: 自動分割するかどうか
            split_interval: 分割間隔
            assist_text_weight: 補助テキストの重み
            volume: 音量
            
        Returns:
            Optional[bytes]: WAV形式の音声データ、エラー時はNone
        """
        try:
            # パラメータが指定されていない場合はデフォルト値を使用
            style = style or self.style
            style_weight = style_weight or self.style_weight
            sdp_ratio = sdp_ratio or self.sdp_ratio
            noise = noise or self.noise
            noise_w = noise_w or self.noise_w
            length = length or self.length
            line_split = line_split if line_split is not None else self.line_split
            split_interval = split_interval or self.split_interval
            assist_text_weight = assist_text_weight or self.assist_text_weight
            volume = volume if volume is not None else self.volume
            
            # APIリクエストのパラメータを設定
            params = {
                "text": text,
                "speaker_id": self.speaker_id,
                "model_id": self.model_id,
                "style": style,
                "style_weight": style_weight,
                "sdp_ratio": sdp_ratio,
                "noise": noise,
                "noisew": noise_w,
                "length": length,
                "auto_split": line_split,
                "split_interval": split_interval,
                "assist_text_weight": assist_text_weight
            }
            
            # APIリクエストを送信
            api_url = f"http://{self.api_host}:{self.api_port}/voice"
            response = await asyncio.to_thread(requests.get, api_url, params=params)
            response.raise_for_status()
            
            # 音量調整
            if volume != 1.0:
                audio_data = self._adjust_volume(response.content, volume)
                return audio_data
            
            return response.content
            
        except Exception as e:
            logger.error(f"音声合成エラー: {str(e)}")
            return None 