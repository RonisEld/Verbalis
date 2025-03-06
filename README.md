# Verbalis - Voice Chat Assistant

このプロジェクトは、Google Gemini 2.0 Flash APIを使用したチャットボットと、Style-BERT-VITS2による音声合成を組み合わせたコマンドラインアプリケーションです。テキストでチャットし、AIの応答を音声で聞くことができます。

## 機能

- Google Gemini 2.0 Flash APIを使用したテキストチャット
- Style-BERT-VITS2による高品質な音声合成
- 複数の音声モデルと話者の切り替え
- コマンドラインインターフェース
- 複数のキャラクター設定による会話スタイルの切り替え
- 直接組み込み式のチャットボット実装（APIサーバー不要）

## 前提条件

- Python 3.8以上
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
cp configuration/.env.example configuration/.env
```

5. `.env`ファイルを編集して、Google Gemini APIキーを設定します：

```bash
GEMINI_API_KEY=your_api_key_here
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

2. 必要に応じて `configuration/appconfig.py` ファイルの設定を変更します
   - `USE_GPU` を環境に合わせて設定してください

## 使い方

### 直接組み込み式チャットボット（推奨）

直接組み込み式のチャットボットは、APIサーバーを経由せずに動作するため、より高速な応答が可能です。

```bash
python chatbot.py
```

または、特定のキャラクター設定を指定して起動することもできます：

```bash
python chatbot.py --character friendly
```

### APIサーバー経由のチャットボット

1. まず、Style-BERT-VITS2 APIサーバーを起動します：

```bash
python apiserver.py
```

または直接uvicornを使用する場合:
```bash
uvicorn apiserver:app
```

2. 別のターミナルで、Voice Chatアプリケーションを起動します：

```bash
python chatclient.py
```

または、特定のキャラクター設定を指定して起動することもできます：

```bash
python chatclient.py --character friendly
```

## キャラクター設定

Verbalisは複数のキャラクター設定をサポートしています。キャラクター設定は `character_prompts` ディレクトリに配置されています。

### 利用可能なキャラクター

- `friendly` - カジュアルで親しみやすい口調のアシスタント
- `formal` - 敬語を使用する丁寧な口調のアシスタント
- `default` - シンプルな標準的なアシスタント

### 独自のキャラクター設定の追加

独自のキャラクター設定を追加するには、`character_prompts` ディレクトリに新しいテキストファイルを作成します。
ファイル名がキャラクター名になります（例: `mycharacter.txt`）。

## チャットコマンド

チャット中に以下のコマンドを使用できます：

- `/model <id>` - 使用する音声モデルIDを変更
- `/speaker <id>` - 使用する話者IDを変更
- `/models` - 利用可能な音声モデル一覧を表示
- `/character <name>` - キャラクター設定を変更
- `/characters` - 利用可能なキャラクター一覧を表示
- `/clear` - チャット履歴をクリア（キャラクター設定は保持）
- `/volume <0.0-1.0>` - 音声の音量を設定
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

`configuration/appconfig.py` ファイルを編集して設定を変更できます:

- `MODEL_DIR`: モデルファイルが配置されているディレクトリ
- `CHARACTER_PROMPTS_DIR`: キャラクタープロンプトファイルが配置されているディレクトリ
- `DEFAULT_CHARACTER`: デフォルトのキャラクター設定
- `USE_GPU`: GPUを使用するかどうか
- その他の音声合成パラメータ

## 実装の違い

### 直接組み込み式チャットボット (chatbot.py)

- APIサーバーを経由せず、直接Style-BERT-VITS2モデルを呼び出します
- より高速な応答が可能
- 単一のプロセスで動作するため、リソース効率が良い
- オフライン環境でも使用可能

### APIサーバー経由のチャットボット (apiserver.py + chatclient.py)

- APIサーバーを経由して音声合成を行います
- 複数のクライアントから同じAPIサーバーにアクセス可能
- サーバーとクライアントを分離できるため、柔軟な構成が可能
- 将来的なGUIアプリケーションの開発に適している

## 注意事項

- 長いテキストの場合、音声合成に時間がかかる場合があります。
- Google Gemini APIの利用には、APIキーと適切な権限が必要です。

## 将来の展望

- GUIインターフェースの追加
- 音声認識機能の統合
- より多くの音声設定オプションの追加
- 複数のLLMモデルのサポート

## ライセンス

このプロジェクトは、MITライセンスの下で公開されています。 