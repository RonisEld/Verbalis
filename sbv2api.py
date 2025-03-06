"""
Style-BERT-VITS2 API Server

このモジュールはStyle-BERT-VITS2を使用した音声合成APIサーバーを提供します。
FastAPIを使用してHTTPエンドポイントを公開し、テキストから音声を生成します。
"""

import asyncio
import io
import wave
import hashlib
import logging
import os
import glob
from time import time
from typing import Optional, Dict, List
from fastapi import FastAPI, Query, Response, HTTPException, Path
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
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
    Languages,
)

# 設定ファイルからの読み込み
from config import (
    MODEL_DIR,
    USE_GPU,
    VERBOSE,
    HOST,
    PORT,
    BERT_MODEL_NAME
)

# ロギングの設定
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 利用可能なモデルを自動検出
def get_available_models():
    models = []
    model_dirs = glob.glob(os.path.join(MODEL_DIR, "*"))
    for model_dir in model_dirs:
        if os.path.isdir(model_dir):
            model_name = os.path.basename(model_dir)
            # モデルファイルが存在するか確認
            model_file = os.path.join(model_dir, f"{model_name}.safetensors")
            config_file = os.path.join(model_dir, "config.json")
            style_file = os.path.join(model_dir, "style_vectors.npy")
            
            if os.path.exists(config_file) and os.path.exists(style_file):
                # safetensorsファイルが存在しない場合、他の拡張子を確認
                if not os.path.exists(model_file):
                    model_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
                    if model_files:
                        model_file = model_files[0]  # 最初に見つかったモデルファイルを使用
                    else:
                        continue  # 有効なモデルファイルがない場合はスキップ
                
                models.append({
                    "id": len(models),
                    "name": model_name,
                    "path": model_dir,
                    "model_file": model_file,
                    "config_file": config_file,
                    "style_file": style_file
                })
    return models

# 利用可能なモデルのリスト
AVAILABLE_MODELS = get_available_models()

# デフォルトモデル（最初に見つかったモデル）
DEFAULT_MODEL_ID = 0 if AVAILABLE_MODELS else None

# 利用可能なモデルがない場合はエラーを表示
if not AVAILABLE_MODELS:
    logger.error("利用可能なモデルが見つかりません。model_assetsディレクトリにモデルを配置してください。")
    raise RuntimeError("No available models found")

# BERTモデルの読み込み
bert_models.load_model(Languages.JP, BERT_MODEL_NAME)
bert_models.load_tokenizer(Languages.JP, BERT_MODEL_NAME)


class StyleBertVits2TTS:
    """
    Style-BERT-VITS2を使用したテキスト音声合成クラス
    
    このクラスはStyle-BERT-VITS2モデルをラップし、テキストから音声を生成する機能と
    生成した音声をキャッシュする機能を提供します。
    """
    
    def __init__(self, model_file: str, config_file: str, style_file: str, use_gpu: bool = False, verbose: bool = False):
        """
        StyleBertVits2TTSクラスの初期化
        
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
        self.cache = {}  # キャッシュ用の辞書

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
        cache_key = self.generate_cache_key(text, speaker_id, style)

        # キャッシュをチェック
        if cache_key in self.cache:
            if self.verbose:
                logger.info(f"キャッシュヒット: {speaker_id}/{style}: {text}")
            return self.cache[cache_key]

        # 別スレッドでTTSを実行
        try:
            start_time = time()
            rate, audio = await asyncio.to_thread(self.tts_model.infer, text=text, speaker_id=speaker_id, style=style, **kwargs)
            if self.verbose:
                logger.info(f"音声生成完了（{time() - start_time}秒）: {text}")

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
            self.cache[cache_key] = audio_data
            return audio_data

        except Exception as e:
            logger.error(f"WAV変換エラー: {str(e)}")
            raise ex


# モデルインスタンスを保持する辞書
tts_models: Dict[int, StyleBertVits2TTS] = {}

# ライフスパンイベント
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    """
    サーバーのライフスパンイベント
    """
    # 起動時の処理
    model_count = len(AVAILABLE_MODELS)
    
    print("\n" + "=" * 50)
    print("Style-BERT-VITS2 API サーバーが起動しました！")
    print("-" * 50)
    logger.info(f"利用可能なモデル数: {model_count}")
    logger.info(f"APIドキュメント: http://localhost:{PORT}/docs")
    print("=" * 50 + "\n")

    yield

    # 終了時の処理
    logger.info("サーバーをシャットダウンしています...")

# FastAPIアプリケーションの作成
app = FastAPI(
    title="Style-BERT-VITS2 API",
    description="テキストから音声を生成するAPIサーバー",
    version="1.0.0",
    lifespan=lifespan
)


def get_tts_model(model_id: int) -> StyleBertVits2TTS:
    """
    指定されたモデルIDに対応するTTSモデルを取得する
    
    Args:
        model_id: モデルID
        
    Returns:
        TTSモデルインスタンス
        
    Raises:
        HTTPException: 指定されたモデルIDが存在しない場合
    """
    # モデルIDの存在チェック
    if model_id < 0 or model_id >= len(AVAILABLE_MODELS):
        raise HTTPException(status_code=404, detail=f"Model ID {model_id} not found")
    
    # モデルがまだロードされていない場合はロード
    if model_id not in tts_models:
        model_info = AVAILABLE_MODELS[model_id]
        logger.info(f"モデルをロード中: {model_info['name']} (ID: {model_id})")
        tts_models[model_id] = StyleBertVits2TTS(
            model_info["model_file"],
            model_info["config_file"],
            model_info["style_file"],
            USE_GPU,
            VERBOSE
        )
    
    return tts_models[model_id]


@app.get("/models")
async def list_models():
    """
    利用可能なモデルの一覧を取得するエンドポイント
    
    Returns:
        利用可能なモデルの一覧
    """
    return {
        "models": [
            {
                "id": model["id"],
                "name": model["name"]
            } for model in AVAILABLE_MODELS
        ],
        "default_model_id": DEFAULT_MODEL_ID
    }


@app.get("/voice")
async def get_voice(
    text: str = Query(..., alias="text", description="音声に変換するテキスト"),
    speaker_id: int = Query(0, alias="speaker_id", description="話者ID"),
    sdp_ratio: float = Query(DEFAULT_SDP_RATIO, alias="sdp_ratio", description="SDP比率"),
    noise: float = Query(DEFAULT_NOISE, alias="noise", description="ノイズ量"),
    noise_w: float = Query(DEFAULT_NOISEW, alias="noisew", description="ノイズ幅"),
    length: float = Query(DEFAULT_LENGTH, alias="length", description="音声の長さ"),
    language: Languages = Query(Languages.JP, alias="language", description="言語"),
    line_split: bool = Query(DEFAULT_LINE_SPLIT, alias="auto_split", description="自動分割するかどうか"),
    split_interval: float = Query(DEFAULT_SPLIT_INTERVAL, alias="split_interval", description="分割間隔"),
    assist_text: Optional[str] = Query(None, alias="assist_text", description="補助テキスト"),
    assist_text_weight: float = Query(DEFAULT_ASSIST_TEXT_WEIGHT, alias="assist_text_weight", description="補助テキストの重み"),
    style: str = Query(DEFAULT_STYLE, alias="style", description="スタイル名"),
    style_weight: float = Query(DEFAULT_STYLE_WEIGHT, alias="style_weight", description="スタイルの重み"),
    reference_audio_path: Optional[str] = Query(None, alias="reference_audio_path", description="参照音声ファイルのパス"),
    model_id: int = Query(DEFAULT_MODEL_ID, alias="model_id", description="使用するモデルのID")
):
    """
    テキストから音声を生成するエンドポイント
    
    Args:
        text: 音声に変換するテキスト
        speaker_id: 話者ID
        sdp_ratio: SDP比率
        noise: ノイズ量
        noise_w: ノイズ幅
        length: 音声の長さ
        language: 言語
        line_split: 自動分割するかどうか
        split_interval: 分割間隔
        assist_text: 補助テキスト
        assist_text_weight: 補助テキストの重み
        style: スタイル名
        style_weight: スタイルの重み
        reference_audio_path: 参照音声ファイルのパス
        model_id: 使用するモデルのID
        
    Returns:
        WAV形式の音声データ
        
    Raises:
        HTTPException: 音声生成中にエラーが発生した場合
    """
    try:
        # 指定されたモデルIDのTTSモデルを取得
        sbv = get_tts_model(model_id)
        
        audio_data = await sbv.tts(
            text=text,
            speaker_id=speaker_id,
            style=style,
            language=language,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noise_w,
            length=length,
            line_split=line_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            style_weight=style_weight,
            reference_audio_path=reference_audio_path,
        )
        return Response(content=audio_data, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as ex:
        logger.error(f"音声生成エンドポイントエラー: {str(ex)}")
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/voice/{model_id}")
async def get_voice_by_model_id(
    model_id: int = Path(..., description="使用するモデルのID"),
    text: str = Query(..., alias="text", description="音声に変換するテキスト"),
    speaker_id: int = Query(0, alias="speaker_id", description="話者ID"),
    sdp_ratio: float = Query(DEFAULT_SDP_RATIO, alias="sdp_ratio", description="SDP比率"),
    noise: float = Query(DEFAULT_NOISE, alias="noise", description="ノイズ量"),
    noise_w: float = Query(DEFAULT_NOISEW, alias="noisew", description="ノイズ幅"),
    length: float = Query(DEFAULT_LENGTH, alias="length", description="音声の長さ"),
    language: Languages = Query(Languages.JP, alias="language", description="言語"),
    line_split: bool = Query(DEFAULT_LINE_SPLIT, alias="auto_split", description="自動分割するかどうか"),
    split_interval: float = Query(DEFAULT_SPLIT_INTERVAL, alias="split_interval", description="分割間隔"),
    assist_text: Optional[str] = Query(None, alias="assist_text", description="補助テキスト"),
    assist_text_weight: float = Query(DEFAULT_ASSIST_TEXT_WEIGHT, alias="assist_text_weight", description="補助テキストの重み"),
    style: str = Query(DEFAULT_STYLE, alias="style", description="スタイル名"),
    style_weight: float = Query(DEFAULT_STYLE_WEIGHT, alias="style_weight", description="スタイルの重み"),
    reference_audio_path: Optional[str] = Query(None, alias="reference_audio_path", description="参照音声ファイルのパス")
):
    """
    指定したモデルIDを使用してテキストから音声を生成するエンドポイント
    
    Args:
        model_id: 使用するモデルのID
        text: 音声に変換するテキスト
        speaker_id: 話者ID
        sdp_ratio: SDP比率
        noise: ノイズ量
        noise_w: ノイズ幅
        length: 音声の長さ
        language: 言語
        line_split: 自動分割するかどうか
        split_interval: 分割間隔
        assist_text: 補助テキスト
        assist_text_weight: 補助テキストの重み
        style: スタイル名
        style_weight: スタイルの重み
        reference_audio_path: 参照音声ファイルのパス
        
    Returns:
        WAV形式の音声データ
        
    Raises:
        HTTPException: 音声生成中にエラーが発生した場合
    """
    try:
        # 指定されたモデルIDのTTSモデルを取得
        sbv = get_tts_model(model_id)
        
        audio_data = await sbv.tts(
            text=text,
            speaker_id=speaker_id,
            style=style,
            language=language,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noise_w,
            length=length,
            line_split=line_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            style_weight=style_weight,
            reference_audio_path=reference_audio_path,
        )
        return Response(content=audio_data, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as ex:
        logger.error(f"音声生成エンドポイントエラー: {str(ex)}")
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/")
async def root():
    """
    ルートエンドポイント
    
    Returns:
        APIの基本情報
    """
    return {
        "message": "Style-BERT-VITS2 API Server",
        "docs_url": "/docs",
        "endpoints": {
            "models": "/models",
            "voice": "/voice",
            "voice_by_model_id": "/voice/{model_id}"
        },
        "available_models": len(AVAILABLE_MODELS)
    }


if __name__ == "__main__":
    import uvicorn
    import sys
    
    # サーバー起動前のメッセージ
    print("\n" + "=" * 50)
    print("Style-BERT-VITS2 API サーバーを起動しています...")
    print(f"ホスト: {HOST}, ポート: {PORT}")
    print("=" * 50)
    
    try:
        # サーバー起動
        uvicorn.run("sbv2api:app", host=HOST, port=PORT)
    except OSError as e:
        if "address already in use" in str(e).lower() or "通常、各ソケット アドレス" in str(e):
            print("\n" + "=" * 50)
            print(f"エラー: ポート {PORT} はすでに使用されています。")
            print("別のポートを使用するか、現在実行中のサーバーを停止してください。")
            print("=" * 50 + "\n")
            sys.exit(1)
        else:
            raise