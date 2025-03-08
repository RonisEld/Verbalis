"""
ユーティリティモジュール

このモジュールは共通のユーティリティ関数を提供します。
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models

# ロギングの設定
logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """
    ロギングを設定する
    
    Args:
        level: ロギングレベル
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_bert_models(language: Languages, model_name: str):
    """
    BERTモデルを読み込む
    
    Args:
        language: 言語
        model_name: モデル名
    """
    try:
        bert_models.load_model(language, model_name)
        bert_models.load_tokenizer(language, model_name)
        logger.info(f"BERTモデルを読み込みました: {model_name}")
    except Exception as e:
        logger.error(f"BERTモデルの読み込みに失敗しました: {str(e)}")
        raise

def check_api_key(api_key: str, key_name: str, help_message: str) -> bool:
    """
    APIキーをチェックする
    
    Args:
        api_key: APIキー
        key_name: キーの名前
        help_message: ヘルプメッセージ
        
    Returns:
        bool: APIキーが有効かどうか
    """
    if not api_key or api_key == f"your_{key_name}_here":
        logger.error(f"エラー: {key_name}が設定されていません。")
        logger.error(help_message)
        return False
        
    # テスト用のダミーAPIキーの場合は警告を表示
    if api_key == f"dummy_{key_name}_for_testing":
        logger.warning(f"警告: テスト用のダミー{key_name}を使用しています。実際のAPIキーに置き換えてください。")
        
    return True

def get_env_path():
    """
    .envファイルのパスを取得する
    
    Returns:
        str: .envファイルのパス
    """
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configuration", ".env")

def get_character_prompt_path(character_name: str, character_prompts_dir: str) -> str:
    """
    キャラクタープロンプトファイルのパスを取得する
    
    Args:
        character_name: キャラクター名
        character_prompts_dir: キャラクタープロンプトディレクトリ
        
    Returns:
        str: キャラクタープロンプトファイルのパス
    """
    return os.path.join(character_prompts_dir, f"{character_name}.txt") 