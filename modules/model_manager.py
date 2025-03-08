"""
モデル管理モジュール

このモジュールはStyle-BERT-VITS2モデルの検索、読み込み、管理を行うクラスを提供します。
"""

import os
import glob
import logging
from typing import Dict, List, Optional

# 自作モジュールのインポート
from modules.tts import BaseTTS, StyleBertVits2TTS, DirectTTS

# ロギングの設定
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Style-BERT-VITS2モデルの管理クラス
    
    このクラスはモデルファイルの検索、読み込み、管理を行います。
    """
    
    def __init__(self, model_dir: str, use_gpu: bool = False, verbose: bool = False, max_cache_size: int = 100):
        """
        ModelManagerクラスの初期化
        
        Args:
            model_dir: モデルディレクトリのパス
            use_gpu: GPUを使用するかどうか
            verbose: 詳細なログを出力するかどうか
            max_cache_size: TTSモデルのキャッシュサイズ
        """
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.max_cache_size = max_cache_size
        self.models = {}  # モデルのキャッシュ
        
        # モデル情報をスキャン
        self.model_info = self._scan_models()
        
        # デフォルトモデルID
        self.default_model_id = 0 if self.model_info else None
        
        # 利用可能なモデルがない場合はエラーを表示
        if not self.model_info:
            logger.error("利用可能なモデルが見つかりません。model_assetsディレクトリにモデルを配置してください。")
        
    def _scan_models(self) -> List[Dict]:
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
            List[Dict]: 利用可能なモデルのリスト
        """
        models = []
        model_dirs = glob.glob(os.path.join(self.model_dir, "*"))
        
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
                        # safetensorsファイルを探す
                        safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
                        if safetensors_files:
                            model_file = safetensors_files[0]  # 最初に見つかったモデルファイルを使用
                        else:
                            # pthファイルを探す
                            pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
                            if pth_files:
                                model_file = pth_files[0]
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
                    
                    if self.verbose:
                        logger.info(f"モデルを検出: {model_name} (ID: {len(models)-1})")
        
        return models
    
    def get_model(self, model_id: int, model_class=None) -> Optional[BaseTTS]:
        """
        指定されたIDのモデルを取得する
        
        Args:
            model_id: モデルID
            model_class: 使用するモデルクラス（デフォルトはDirectTTS）
            
        Returns:
            Optional[BaseTTS]: TTSモデルインスタンス、モデルが見つからない場合はNone
            
        Raises:
            ValueError: 無効なモデルIDが指定された場合
        """
        if model_class is None:
            model_class = DirectTTS
            
        # モデルIDの検証
        if model_id < 0 or model_id >= len(self.model_info):
            raise ValueError(f"無効なモデルID: {model_id}")
        
        # キャッシュをチェック
        cache_key = f"{model_id}_{model_class.__name__}"
        if cache_key in self.models:
            return self.models[cache_key]
        
        # モデル情報を取得
        model_info = self.model_info[model_id]
        
        # モデルを初期化
        try:
            model = model_class(
                model_file=model_info["model_file"],
                config_file=model_info["config_file"],
                style_file=model_info["style_file"],
                use_gpu=self.use_gpu,
                verbose=self.verbose,
                max_cache_size=self.max_cache_size
            )
            
            # キャッシュに保存
            self.models[cache_key] = model
            
            return model
            
        except Exception as e:
            logger.error(f"モデルの初期化エラー (ID: {model_id}): {str(e)}")
            return None
    
    def get_api_model(self, model_id: int) -> Optional[StyleBertVits2TTS]:
        """
        APIサーバー用のモデルを取得する
        
        Args:
            model_id: モデルID
            
        Returns:
            Optional[StyleBertVits2TTS]: TTSモデルインスタンス
        """
        return self.get_model(model_id, StyleBertVits2TTS)
    
    def get_direct_model(self, model_id: int) -> Optional[DirectTTS]:
        """
        クライアント直接利用用のモデルを取得する
        
        Args:
            model_id: モデルID
            
        Returns:
            Optional[DirectTTS]: TTSモデルインスタンス
        """
        return self.get_model(model_id, DirectTTS)
    
    def get_available_models(self) -> Dict:
        """
        利用可能なモデルの情報を取得する
        
        Returns:
            Dict: モデル情報の辞書
        """
        return {
            "models": self.model_info,
            "default_model_id": self.default_model_id
        } 