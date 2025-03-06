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
import sys
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
        self.cache = {}
        self.cache_keys = []  # キャッシュキーの順序を保持
        
        # 設定ファイルから最大キャッシュサイズを読み込む
        try:
            self.max_cache_size = MAX_CACHE_SIZE
        except (ImportError, AttributeError):
            # 設定が見つからない場合はデフォルト値を使用
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
        try:
            if OPTIMIZE_SHORT_SENTENCES and len(text) <= 10:
                kwargs['line_split'] = False  # 短いテキストは分割しない
        except (ImportError, AttributeError):
            # 設定が見つからない場合はデフォルトの動作
            if len(text) <= 10:
                kwargs['line_split'] = False
        
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

        except Exception as e:
            logger.error(f"WAV変換エラー: {str(e)}")
            raise e  # 正しい例外を再発生


# モデルインスタンスを保持する辞書
tts_models: Dict[int, StyleBertVits2TTS] = {}

# ライフスパンイベント
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    """
    アプリケーションのライフスパンイベント
    
    Args:
        app: FastAPIアプリケーション
    """
    # 設定ファイルからの読み込み
    from configuration.appconfig import (
        MODEL_DIR,
        USE_GPU,
        VERBOSE,
    )
    
    # モデルディレクトリの確認
    if not os.path.exists(MODEL_DIR):
        logger.error(f"モデルディレクトリが見つかりません: {MODEL_DIR}")
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info(f"モデルディレクトリを作成しました: {MODEL_DIR}")
    
    # BERTモデルは既に自動的に初期化されているため、
    # 明示的な初期化は不要
    logger.info("BERTモデルは既に初期化されています")
    
    # モデルファイルの検索（サブディレクトリも含む）
    model_files = []
    
    # .pthファイルを検索
    pth_files = glob.glob(os.path.join(MODEL_DIR, "**/*.pth"), recursive=True)
    model_files.extend(pth_files)
    
    # .safetensorsファイルを検索
    safetensors_files = glob.glob(os.path.join(MODEL_DIR, "**/*.safetensors"), recursive=True)
    model_files.extend(safetensors_files)
    
    if not model_files:
        logger.warning(f"モデルファイルが見つかりません: {MODEL_DIR}")
        logger.info("モデルファイルなしで起動を続行します。後でモデルを追加してください。")
    else:
        logger.info(f"モデルファイルが見つかりました: {len(model_files)}個")
    
    logger.info("APIサーバーを起動しました")
    yield
    
    # 終了処理
    logger.info("APIサーバーを終了しています...")
    # BERTモデルの解放は不要
    logger.info("APIサーバーを終了しました")

# FastAPIアプリケーションの作成
app = FastAPI(
    title="Style-BERT-VITS2 API",
    description="テキストから音声を生成するAPIサーバー",
    version="1.0.0",
    lifespan=lifespan
)


def get_tts_model(model_id: int) -> StyleBertVits2TTS:
    """
    指定されたIDのTTSモデルを取得する
    
    Args:
        model_id: モデルID
        
    Returns:
        StyleBertVits2TTS: TTSモデルインスタンス
        
    Raises:
        HTTPException: モデルが見つからない場合
    """
    # モデルがすでにロードされている場合はそれを返す
    if model_id in tts_models:
        return tts_models[model_id]
    
    # モデルファイルの検索（サブディレクトリも含む）
    model_files = []
    
    # .pthファイルを検索
    pth_files = glob.glob(os.path.join(MODEL_DIR, "**/*.pth"), recursive=True)
    model_files.extend(pth_files)
    
    # .safetensorsファイルを検索
    safetensors_files = glob.glob(os.path.join(MODEL_DIR, "**/*.safetensors"), recursive=True)
    model_files.extend(safetensors_files)
    
    # 利用可能なモデルIDをチェック
    if not model_files:
        raise HTTPException(
            status_code=404,
            detail=f"モデルが見つかりません。model_assetsディレクトリにモデルファイルを配置してください。"
        )
    
    # 指定されたIDが範囲外の場合
    if model_id < 0 or model_id >= len(model_files):
        available_models = list(range(len(model_files)))
        raise HTTPException(
            status_code=404,
            detail=f"モデルID {model_id} が見つかりません。利用可能なモデルID: {available_models}"
        )
    
    # モデルをロード
    try:
        model_file = model_files[model_id]
        model_dir = os.path.dirname(model_file)
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
            raise HTTPException(
                status_code=500,
                detail=f"設定ファイルが見つかりません: {model_file}"
            )
        
        # スタイルファイルを検索
        style_file = None
        style_candidates = [
            f"{base_name}.style.json",
            os.path.join(model_dir, "style_vectors.npy"),
            os.path.join(model_dir, "style.json")
        ]
        for candidate in style_candidates:
            if os.path.exists(candidate):
                style_file = candidate
                break
        
        if not style_file:
            raise HTTPException(
                status_code=500,
                detail=f"スタイルファイルが見つかりません: {model_file}"
            )
        
        # モデルの初期化
        logger.info(f"モデル {model_id} をロード中: {os.path.basename(model_file)}")
        tts_model = StyleBertVits2TTS(model_file, config_file, style_file, USE_GPU, VERBOSE)
        tts_models[model_id] = tts_model
        
        return tts_model
        
    except Exception as e:
        logger.error(f"モデル {model_id} のロードに失敗しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"モデルのロードに失敗しました: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """
    利用可能なモデルの一覧を取得する
    
    Returns:
        Dict: モデル情報の辞書
    """
    # モデルファイルの検索（サブディレクトリも含む）
    model_files = []
    
    # .pthファイルを検索
    pth_files = glob.glob(os.path.join(MODEL_DIR, "**/*.pth"), recursive=True)
    model_files.extend(pth_files)
    
    # .safetensorsファイルを検索
    safetensors_files = glob.glob(os.path.join(MODEL_DIR, "**/*.safetensors"), recursive=True)
    model_files.extend(safetensors_files)
    
    # モデル情報を生成
    models = []
    for i, model_file in enumerate(model_files):
        model_name = os.path.basename(model_file)
        model_info = {
            "id": i,
            "name": model_name,
            "description": f"音声合成モデル {i}"
        }
        models.append(model_info)
    
    # デフォルトのモデルIDを設定
    default_model_id = 0 if models else None
    
    return {
        "models": models,
        "default_model_id": default_model_id
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
    # 設定ファイルからの読み込み
    from configuration.appconfig import HOST, PORT
    
    print("\n" + "=" * 50)
    print("Style-BERT-VITS2 API サーバーを起動しています...")
    print(f"ホスト: {HOST}, ポート: {PORT}")
    print("=" * 50)
    
    # サーバーの起動
    import uvicorn
    import socket
    
    # 使用するポート
    current_port = PORT
    max_port = PORT + 10  # 最大10ポート試す
    
    while current_port <= max_port:
        try:
            # ポートが使用可能かチェック
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((HOST, current_port))
                # ポートが使用可能
                break
        except OSError:
            # ポートが使用中
            print(f"ポート {current_port} は既に使用されています。次のポートを試します。")
            current_port += 1
    
    if current_port > max_port:
        print(f"エラー: 利用可能なポートが見つかりませんでした（{PORT}〜{max_port}）")
        sys.exit(1)
    
    # 実際のポートが設定と異なる場合は通知
    if current_port != PORT:
        print(f"ポート {PORT} は使用中のため、ポート {current_port} で起動します。")
    
    try:
        uvicorn.run(
            "apiserver:app",
            host=HOST,
            port=current_port,
            log_level="info"
        )
    except Exception as e:
        print(f"サーバーの起動に失敗しました: {str(e)}")
        sys.exit(1)