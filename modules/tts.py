"""
テキスト音声合成（TTS）モジュール

このモジュールはStyle-BERT-VITS2を使用したテキスト音声合成クラスを提供します。
APIサーバー用とクライアント直接利用用の両方の実装を含みます。
"""

import io
import wave
import hashlib
import logging
import asyncio
from time import time
from typing import Dict, List, Optional

# Style-BERT-VITS2のインポート
from style_bert_vits2.tts_model import TTSModel

# ロギングの設定
logger = logging.getLogger(__name__)

class BaseTTS:
    """
    Style-BERT-VITS2を使用したテキスト音声合成の基本クラス
    
    このクラスはStyle-BERT-VITS2モデルをラップし、テキストから音声を生成する機能と
    生成した音声をキャッシュする機能を提供します。
    """
    
    def __init__(self, model_file: str, config_file: str, style_file: str, use_gpu: bool = False, verbose: bool = False, max_cache_size: int = 100):
        """
        BaseTTSクラスの初期化
        
        Args:
            model_file: モデルファイルのパス
            config_file: 設定ファイルのパス
            style_file: スタイルベクトルファイルのパス
            use_gpu: GPUを使用するかどうか
            verbose: 詳細なログを出力するかどうか
            max_cache_size: キャッシュに保存する最大アイテム数
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
        self.max_cache_size = max_cache_size

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

    @property
    def style_list(self) -> List[str]:
        """
        モデルが持つスタイルリストを取得する
        
        Returns:
            List[str]: スタイルのリスト
        """
        try:
            if hasattr(self.tts_model, 'style_list') and self.tts_model.style_list:
                return self.tts_model.style_list
            elif hasattr(self.tts_model, 'style_vectors') and self.tts_model.style_vectors:
                return list(self.tts_model.style_vectors.keys())
            else:
                return ["Neutral"]
        except Exception as e:
            logger.error(f"スタイルリスト取得エラー: {e}")
            return ["Neutral"]

class StyleBertVits2TTS(BaseTTS):
    """
    APIサーバー用のStyle-BERT-VITS2テキスト音声合成クラス
    
    このクラスはAPIサーバーで使用するための追加機能を提供します。
    """
    
    def __init__(self, model_file: str, config_file: str, style_file: str, use_gpu: bool = False, verbose: bool = False, max_cache_size: int = 100):
        """
        StyleBertVits2TTSクラスの初期化
        
        Args:
            model_file: モデルファイルのパス
            config_file: 設定ファイルのパス
            style_file: スタイルベクトルファイルのパス
            use_gpu: GPUを使用するかどうか
            verbose: 詳細なログを出力するかどうか
            max_cache_size: キャッシュに保存する最大アイテム数
        """
        super().__init__(model_file, config_file, style_file, use_gpu, verbose, max_cache_size)

    async def tts(self, text: str, speaker_id: int, style: str, optimize_short_sentences: bool = True, **kwargs) -> bytes:
        """
        テキストから音声を生成する（APIサーバー用の拡張機能付き）
        
        Args:
            text: 音声に変換するテキスト
            speaker_id: 話者ID
            style: スタイル名
            optimize_short_sentences: 短い文章の処理を最適化するかどうか
            **kwargs: その他のパラメータ
            
        Returns:
            WAV形式の音声データ（バイト列）
        """
        # 短いテキストの場合は処理を高速化
        if optimize_short_sentences and len(text) <= 10:
            kwargs['line_split'] = False  # 短いテキストは分割しない
            
        return await super().tts(text, speaker_id, style, **kwargs)

class DirectTTS(BaseTTS):
    """
    クライアント直接利用用のStyle-BERT-VITS2テキスト音声合成クラス
    
    このクラスはクライアントアプリケーションで直接使用するための追加機能を提供します。
    """
    
    def __init__(self, model_file: str, config_file: str, style_file: str, use_gpu: bool = False, verbose: bool = False, max_cache_size: int = 100):
        """
        DirectTTSクラスの初期化
        
        Args:
            model_file: モデルファイルのパス
            config_file: 設定ファイルのパス
            style_file: スタイルベクトルファイルのパス
            use_gpu: GPUを使用するかどうか
            verbose: 詳細なログを出力するかどうか
            max_cache_size: キャッシュに保存する最大アイテム数
        """
        super().__init__(model_file, config_file, style_file, use_gpu, verbose, max_cache_size) 