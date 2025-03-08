"""
Style-BERT-VITS2 API Server

このモジュールはStyle-BERT-VITS2を使用した音声合成APIサーバーを提供します。
FastAPIを使用してHTTPエンドポイントを公開し、テキストから音声を生成します。
"""

import asyncio
import logging
from typing import Optional, Dict
from fastapi import FastAPI, Query, Response, HTTPException, Path
from contextlib import asynccontextmanager
from style_bert_vits2.constants import Languages
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
from modules.utils import setup_logging, load_bert_models

# 設定ファイルからの読み込み
from configuration.appconfig import (
    MODEL_DIR,
    USE_GPU,
    VERBOSE,
    HOST,
    PORT,
    BERT_MODEL_NAME,
    MAX_CACHE_SIZE,
    OPTIMIZE_SHORT_SENTENCES,
)

# ロギングの設定
logger = logging.getLogger(__name__)
setup_logging()

# モデル管理クラスのインスタンス化
model_manager = ModelManager(MODEL_DIR, USE_GPU, VERBOSE, MAX_CACHE_SIZE)

# デフォルトモデルID
DEFAULT_MODEL_ID = model_manager.default_model_id

# 利用可能なモデルがない場合はエラーを表示
if DEFAULT_MODEL_ID is None:
    logger.error("利用可能なモデルが見つかりません。model_assetsディレクトリにモデルを配置してください。")
    raise RuntimeError("No available models found")

# BERTモデルの読み込み
load_bert_models(Languages.JP, BERT_MODEL_NAME)

# ライフスパンイベント
@asynccontextmanager
async def lifespan(app):
    """
    アプリケーションのライフスパンイベント
    
    Args:
        app: FastAPIアプリケーション
    """
    # 起動時の処理
    logger.info(f"APIサーバーを起動しています...")
    logger.info(f"ホスト: {HOST}")
    logger.info(f"ポート: {PORT}")
    logger.info(f"GPU使用: {USE_GPU}")
    logger.info(f"詳細ログ: {VERBOSE}")
    logger.info(f"モデルディレクトリ: {MODEL_DIR}")
    
    # 利用可能なモデルを表示
    models = model_manager.get_available_models()["models"]
    logger.info(f"利用可能なモデル数: {len(models)}")
    for model in models:
        logger.info(f"  ID: {model['id']} - 名前: {model['name']}")
    
    yield
    
    # 終了時の処理
    logger.info("APIサーバーを終了しています...")

# FastAPIアプリケーションの作成
app = FastAPI(
    title="Style-BERT-VITS2 API",
    description="Style-BERT-VITS2を使用した音声合成API",
    version="1.0.0",
    lifespan=lifespan
)

def get_tts_model(model_id: int):
    """
    指定されたIDのTTSモデルを取得する
    
    Args:
        model_id: モデルID
        
    Returns:
        StyleBertVits2TTS: TTSモデルインスタンス
        
    Raises:
        HTTPException: モデルが見つからない場合
    """
    try:
        model = model_manager.get_api_model(model_id)
        if model is None:
            raise HTTPException(status_code=404, detail=f"モデルID {model_id} が見つかりません")
        return model
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
async def list_models():
    """
    利用可能なモデルの一覧を取得する
    
    Returns:
        Dict: モデル情報の辞書
    """
    return model_manager.get_available_models()

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
    テキストを音声に変換する
    
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
        Response: WAV形式の音声データ
    """
    # モデルを取得
    tts_model = get_tts_model(model_id)
    
    try:
        # テキストから音声を生成
        audio_data = await tts_model.tts(
            text=text,
            speaker_id=speaker_id,
            style=style,
            style_weight=style_weight,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noise_w,
            length=length,
            language=language,
            line_split=line_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            reference_audio_path=reference_audio_path,
            optimize_short_sentences=OPTIMIZE_SHORT_SENTENCES
        )
        
        # 音声データを返す
        return Response(
            content=audio_data,
            media_type="audio/wav"
        )
        
    except Exception as e:
        logger.error(f"音声生成エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"音声生成エラー: {str(e)}")

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
    指定されたモデルIDを使用してテキストを音声に変換する
    
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
        Response: WAV形式の音声データ
    """
    # モデルを取得
    tts_model = get_tts_model(model_id)
    
    try:
        # テキストから音声を生成
        audio_data = await tts_model.tts(
            text=text,
            speaker_id=speaker_id,
            style=style,
            style_weight=style_weight,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noise_w,
            length=length,
            language=language,
            line_split=line_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            reference_audio_path=reference_audio_path,
            optimize_short_sentences=OPTIMIZE_SHORT_SENTENCES
        )
        
        # 音声データを返す
        return Response(
            content=audio_data,
            media_type="audio/wav"
        )
        
    except Exception as e:
        logger.error(f"音声生成エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"音声生成エラー: {str(e)}")

@app.get("/")
async def root():
    """
    ルートエンドポイント
    
    Returns:
        Dict: APIの情報
    """
    return {
        "name": "Style-BERT-VITS2 API",
        "version": "1.0.0",
        "description": "Style-BERT-VITS2を使用した音声合成API",
        "endpoints": {
            "/": "このエンドポイント（APIの情報）",
            "/models": "利用可能なモデルの一覧",
            "/voice": "テキストを音声に変換",
            "/voice/{model_id}": "指定されたモデルIDを使用してテキストを音声に変換"
        }
    }

# メイン関数
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)