# Verbalis - Voice Chat Assistant

このプロジェクトは、Google Gemini 2.0 Flash APIを使用したチャットボットと、Style-BERT-VITS2による音声合成を組み合わせたコマンドラインアプリケーションです。テキストでチャットし、AIの応答を音声で聞くことができます。

## 機能

- Google Gemini 2.0 Flash APIを使用したテキストチャット
- Style-BERT-VITS2による高品質な音声合成
- 複数の音声モデルと話者の切り替え
- コマンドラインインターフェース

## 前提条件

- Python 3.8以上
- Style-BERT-VITS2 APIサーバー
- Google Gemini API Key

## インストール

1. 依存関係をインストールします
```bash
pip install -r requirements.txt
```

2. GPUを使用する場合は、環境のCUDAバージョンに合わせてPyTorchを再インストールします
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   # CUDA 11.8の場合  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4の場合  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.6の場合  
```

3. 自分のGPUの対応CUDAバージョンを確認する方法。
```bash
nvidia-smi
```
を実行して、右上のCUDAバージョンを確認します。
```bash
nvcc -V
```
を実行して、インストールされているCUDA toolkitのバージョンを確認します。

CUDAは後方互換性があるため、GPUが対応しているCUDAバージョンとCUDA toolkitのバージョンが一致していない場合は、対応している最新のToolkitをインストールしてください。

その上で12.6より新しい場合は12.6を。12.4より新しい場合は12.4を。11.8より新しい場合は11.8を指定するようにしてください。

4. `.env.example`ファイルを`.env`にコピーし、必要な設定を行います：

```bash
cp .env.example .env
```

5. `.env`ファイルを編集して、Google Gemini APIキーを設定します：

```
GEMINI_API_KEY=your_api_key_here
DEFAULT_SPEAKER_ID=0
DEFAULT_MODEL_ID=0
```

## モデルの設定

1. モデルファイルを `model_assets/[モデル名]` ディレクトリに配置します
   - 例: `model_assets/MergedVoice_Rnn_Asm__Mzk(0_0.3_0.5_0)/`
   - モデルファイル: `[モデル名].safetensors`
   - 設定ファイル: `config.json`
   - スタイルファイル: `style_vectors.npy`

   複数のモデルを使用する場合は、それぞれのモデルを別々のディレクトリに配置します。
   例:
   ```
   model_assets/
   ├── Model1/
   │   ├── Model1.safetensors
   │   ├── config.json
   │   └── style_vectors.npy
   ├── Model2/
   │   ├── Model2.safetensors
   │   ├── config.json
   │   └── style_vectors.npy
   ```

2. 必要に応じて `config.py` ファイルの設定を変更します
   - `USE_GPU` を環境に合わせて設定してください

## 使い方

1. まず、Style-BERT-VITS2 APIサーバーを起動します：

```bash
python sbv2api.py
```

または直接uvicornを使用する場合:
```bash
uvicorn sbv2api:app
```

2. 別のターミナルで、Voice Chatアプリケーションを起動します：

```bash
python voice_chat.py
```

3. プロンプトが表示されたら、チャットを開始できます。

## チャットコマンド

チャット中に以下のコマンドを使用できます：

- `/model <id>` - 使用する音声モデルIDを変更
- `/speaker <id>` - 使用する話者IDを変更
- `/models` - 利用可能な音声モデル一覧を表示
- `/help` - ヘルプを表示
- `/exit` - チャットを終了

## API エンドポイント

### GET /models

利用可能なモデルの一覧を取得します。

**レスポンス例:**
```json
{
  "models": [
    {
      "id": 0,
      "name": "Model1"
    },
    {
      "id": 1,
      "name": "Model2"
    }
  ],
  "default_model_id": 0
}
```

### GET /voice

テキストから音声を生成します。

**パラメータ:**
- `text`: 音声に変換するテキスト（必須）
- `speaker_id`: 話者ID（デフォルト: 0）
- `style`: スタイル名（デフォルト: "Neutral"）
- `model_id`: 使用するモデルのID（デフォルト: 0）
- その他多数のパラメータが利用可能

**レスポンス:**
- WAV形式の音声データ

### GET /voice/{model_id}

指定したモデルIDを使用してテキストから音声を生成します。

**パラメータ:**
- `model_id`: 使用するモデルのID（パスパラメータ）
- `text`: 音声に変換するテキスト（必須）
- `speaker_id`: 話者ID（デフォルト: 0）
- `style`: スタイル名（デフォルト: "Neutral"）
- その他多数のパラメータが利用可能

**レスポンス:**
- WAV形式の音声データ

## 設定

`config.py` ファイルを編集して設定を変更できます:

```python
# モデルディレクトリ設定
MODEL_DIR = "model_assets"

# 実行設定
USE_GPU = True
VERBOSE = True

# サーバー設定
HOST = "0.0.0.0"
PORT = 8000

# BERTモデル設定
BERT_MODEL_NAME = "ku-nlp/deberta-v2-large-japanese-char-wwm"
```

## 注意事項

- Style-BERT-VITS2 APIサーバーが起動していない場合、音声合成機能は使用できません。
- 長いテキストの場合、音声合成に時間がかかる場合があります。
- Google Gemini APIの利用には、APIキーと適切な権限が必要です。

## 将来の展望

- GUIインターフェースの追加
- 音声認識機能の統合
- より多くの音声設定オプションの追加

## ライセンス

このプロジェクトは、MITライセンスの下で公開されています。 