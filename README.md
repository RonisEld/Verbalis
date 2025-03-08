# Verbalis - Voice Chat Assistant


Verbalisは、[Style-BERT-VITS2](https://github.com/litagin02/Style-Bert-VITS2)による高品質な音声合成を活用した様々な機能を含むUIを目指します。

## ✨ 機能

- [Style-BERT-VITS2](https://github.com/litagin02/Style-Bert-VITS2)による高品質な音声合成
- 用意した音声モデルの声と、自然に会話が可能なチャットボット"Chat"タブ
- 音声合成のワークフローを扱いやすくするためのエディター"VoiceGen"タブ
- [SBV2](https://github.com/litagin02/Style-Bert-VITS2)に標準搭載されている"スタイル"機能や各種パラメーターの変更機能
- ChatBotは具体的なプロフィールやユーザーの呼称など、自由に決めることが可能

## 📋 前提条件

- Python 3.8以上
- Google Gemini API Key

## 🔧 インストール

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

3. 自分のGPUの対応CUDAバージョンを確認する方法：
```bash
nvidia-smi
```
右上のCUDAバージョンを確認します。
```bash
nvcc -V
```
インストールされているCUDA toolkitのバージョンを確認します。

CUDAは後方互換性があるため、GPUが対応しているCUDAバージョンとCUDA toolkitのバージョンが一致していない場合は、対応している最新のToolkitをインストールしてください。

4. `.env.example`ファイルを`.env`にコピーします：
```bash
cp configuration/.env.example configuration/.env
```

5. `configuration/.env`ファイルを編集して、Google Gemini APIキーを設定します：
```bash
GEMINI_API_KEY=your_api_key_here
```

6. `appconfig.py.example`ファイルを`appconfig.py`にコピーします：
```bash
cp configuration/appconfig.py.example configuration/appconfig.py
```

7. `appconfig.py`ファイル内のユーザー名やその他の設定を環境に合わせて編集します：
```c
# キャラクターの共通設定
CHARACTER_COMMON_SETTINGS = """
ユーザーの名前は「あなたの名前」です。  # ここを変更
君・さん・ちゃん・呼び捨てなどの形式は、キャラクターの性格やシチュエーションによって判断して下さい。
絵文字の使用は禁止します。
難しい漢字を含む言葉については、ひらがなやカタカナを使用します。
"""

# 実行設定
USE_GPU = True  # GPUを使用しない場合はFalseに変更

# 音声パラメーター
DEFAULT_STYLE: 音声の感情スタイル（Neutral、Happy、Sad、Angry、Surprised）
DEFAULT_STYLE_WEIGHT: スタイルの適用強度
DEFAULT_SDP_RATIO: SDP（Speaker Disentanglement Pretraining）の比率
DEFAULT_NOISE: 音声のノイズ量
DEFAULT_NOISEW: ノイズの適用強度
DEFAULT_LENGTH: 音声の速度
DEFAULT_LINE_SPLIT: テキストを自動で分割するかどうか
DEFAULT_SPLIT_INTERVAL: 分割点で何秒間隔をあけるか
DEFAULT_ASSIST_TEXT_WEIGHT: アシストテキストの重み デフォルト推奨
DEFAULT_VOLUME: 音量のデフォルト値（0.0〜1.0）
```

## 🎤 モデルの配置

- モデルファイルを `model_assets/[モデル名]` ディレクトリに配置します
   - 例: `model_assets/[モデル名]/`
   - モデルファイル: `[モデル名].safetensors`
   - 設定ファイル: `config.json`
   - スタイルファイル: `style_vectors.npy`

- 複数のモデルを使用する場合は、それぞれのモデルを別々のディレクトリに配置します：
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

- モデルを所持していない場合
   -  [Style-BERT-VITS2](https://github.com/litagin02/Style-Bert-VITS2)で学習を行う。
   -  サンプルモデルや有料モデルを用意する。

## 🚀 使い方

VerbalisはGradioを使用したWebインターフェイスを採用しています。

### 起動方法

```bash
python webui.py
```

起動オプション:
- `--host`: ホストアドレス（デフォルト: appconfig.pyのHOST）
- `--port`: ポート番号（デフォルト: appconfig.pyのPORT）
- `--share`: Gradio共有リンクを生成する
- `--gpu`: GPUを使用する（appconfig.pyの設定を上書き）

### 機能

- Chat
   - Gemini APIを活用した、高品質なチャットボット
   - SBV2によるチャット音声の自動生成と自動再生
   - キャラクタープロンプトによる、チャットボットのカスタマイズ
   - 音声ファイルの自動保存(On/Off可) `(outputs/Chat/{日付}/)`

- VoiceGem
   - SBV2による高速かつ高品質な音声合成
   - 履歴機能により、ワンクリックで過去の音声を再生
   - セッションを跨いだ履歴の読み込み
   - 音声ファイルの自動保存 `(outputs/VoiceGen/{日付}/)`

- 共通要素
   - 保存した音声のメタデータを同名のjsonファイルとして自動生成
   - UI上でのモデル切り替え
   - 音声パラメーターのカスタマイズ

### UI要素

1. **チャットボット**: チャットの履歴を表示します。
2. **設定パネル**: モデル選択、キャラクター選択、音声パラメーターなどの設定を行います。
3. **テキストボックス**: メッセージを入力します。
4. **送信ボタン**: メッセージを送信します。
5. **チャット履歴をリセット**: チャット履歴をクリアします。
6. **音声プレイヤー**: 生成された音声を再生します。 

## 🎭 キャラクター設定

Verbalisは複数のキャラクター設定をサポートしています。キャラクター設定は `character_prompts` ディレクトリに配置されています。

#### 利用可能なキャラクター

- `default` - シンプルな標準的なアシスタント
- `friendly` - カジュアルで親しみやすい口調のアシスタント
- `formal` - 敬語を使用する丁寧な口調のアシスタント

#### 独自のキャラクター設定の追加

独自のキャラクター設定を追加するには、`character_prompts` ディレクトリに新しいテキストファイルを作成します。
ファイル名がキャラクター名になります（例: `mycharacter.txt`）。

### 💬 チャットコマンド

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


### ⚠️ 注意事項

- 長いテキストの場合、音声合成に時間がかかる場合があります
- Google Gemini APIの利用には、APIキーと適切な権限が必要です

### 📝 ライセンス
このリポジトリは、Style-Bert-VITS2に準拠し、GNU Affero General Public License v3.0 を採用します。
詳しくはLICENSEを確認してください。


