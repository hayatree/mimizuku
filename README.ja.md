<h1 align="center">mimizuku</h1>

<p align="center">Rust と CUDA で書いた、タイムスタンプ付きの <a href="https://github.com/openai/whisper">Whisper</a> バッチ文字起こし</p>

<p align="center"><a href="#クイックスタート"><b>クイックスタート</b></a> · <a href="#使い方">使い方</a> · <a href="#仕組み">仕組み</a> · <a href="#検証">検証</a> · <a href="#制限">制限</a></p>

<p align="center">
  <a href="https://github.com/hayatree/mimizuku/actions/workflows/ci.yml"><img src="https://github.com/hayatree/mimizuku/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#ライセンス"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue" alt="License: MIT OR Apache-2.0"></a>
</p>

<p align="center"><sub><a href="README.md">English</a> · <b>日本語</b></sub></p>

Hugging Face のチェックポイントを読み、複数のクリップをまとめて貪欲デコードし、`transformers` と同じ方法でアライメントヘッドから文字や単語片ごとの開始・終了時刻を求めます。

- **`transformers` と一致**（fp16）：確認したすべての基準クリップで、トークン列とトークンのタイムスタンプが同じです（「[検証](#検証)」を参照）。
- **バッチ処理でもタイムスタンプ付き**：複数のクリップをまとめてデコードし、どのクリップにもトークンのタイムスタンプを付けます。
- **小さい**：約3,300行、依存は1つ（[cudarc](https://github.com/coreylowman/cudarc)）。WAV、リサンプリング、log-mel 特徴量、safetensors、JSON、トークナイザは自前で実装しています。

現状は 0.1 です。30秒までのクリップと貪欲デコードに限ります（「[制限](#制限)」を参照）。

## クイックスタート

Rust、CUDA ツールキット、NVIDIA GPU（詳しくは「[ビルド](#ビルド)」）と、モデルのダウンロード用に [uv](https://docs.astral.sh/uv/) があれば、次の手順で動きます。

    git clone https://github.com/hayatree/mimizuku && cd mimizuku
    cargo build --release --locked

    uvx --from huggingface_hub hf download openai/whisper-large-v3-turbo \
        model.safetensors config.json generation_config.json vocab.json \
        --local-dir models/whisper-large-v3-turbo

    ffmpeg -i speech.mp3 speech.wav     # 30秒までの WAV なら何でも可
    ./target/release/mimizuku --model models/whisper-large-v3-turbo speech.wav

書き起こしとトークンごとの時刻が JSON で出力されます。形式とオプションは「[使い方](#使い方)」を見てください。

## ビルド

必要なもの：Linux、Ampere 世代以降の NVIDIA GPU（sm_80 以上）、cudarc 0.19 が対応する CUDA ツールキット（`nvcc` を使用し、cuBLAS は実行時に読み込み）、Rust 1.88 以上。

    cargo build --release --locked      # 実行ファイルは target/release/mimizuku

| 変数 | 既定値 | 内容 |
|---|---|---|
| `NVCC`, `CUDA_HOME` | `PATH` 上の `nvcc`、なければ `/usr/local/cuda` | カーネルをコンパイルする CUDA コンパイラ。cudarc は `CUDARC_CUDA_VERSION` がなければ `PATH` 上の `nvcc --version` を使います |
| `CUDA_ARCH` | `compute_80` | PTX のアーキテクチャ。実行する GPU 向けにドライバが JIT コンパイルします |
| `CUDA_SASS` | なし | 指定した GPU 向けの機械語も埋め込み、JIT を省きます（例：`86,89,121`）。コンテナで便利です |
| `CUDARC_CUDA_VERSION` | `nvcc --version` から判定 | バインディングの CUDA API の版（例：`12080`） |

NVIDIA DGX Spark（GB10、sm_121、CUDA 13.0、aarch64）で開発・動作確認しています。CI では x86-64 と CUDA 12.8 の環境（GPU なし）でビルドと CPU のテストを行っています。他の GPU での報告を歓迎します。

## 使い方

モデルのディレクトリには、`model.safetensors`、`config.json`、`generation_config.json`、`vocab.json` を含む Hugging Face のスナップショットを指定します（例：`openai/whisper-large-v3`、`openai/whisper-large-v3-turbo`）。

    mimizuku --model ~/models/whisper-large-v3 --language ja clip1.wav clip2.wav > out.json
    mimizuku --model ~/models/whisper-large-v3-turbo --dir clips/ --out out.json

| オプション | 内容 |
|---|---|
| `--model DIR` | モデルのディレクトリ（必須） |
| `--dir DIR [--prefix STR]` | DIR 内のすべての `.wav`（名前順）。STR で始まる名前だけに絞れます |
| `--language CODE` | `en`、`ja` など。`auto`（既定）はクリップごとに判定します |
| `--task NAME` | `transcribe`（既定）または `translate`（英語への翻訳） |
| `--out FILE` | JSON を標準出力ではなく FILE に書きます |
| `--batch N` | まとめてデコードするクリップ数（既定 32。large-v3 で32なら GPU メモリ約14.5 GB） |
| `--enc-group N` | エンコーダに一度に通すクリップ数（既定 8） |
| `--quiet` | 標準エラーに進み具合や時間を出しません |

英語専用モデル（`*.en`）では、`--language` は `en` か `auto`、`--task` は `transcribe` だけを受け付けます。

入力：WAV（16/24/32ビット PCM または32ビット float）、チャンネル数は問いません（混ぜてモノラルに）、1〜768 kHz、30秒まで。

出力は入力の順で、時刻の単位は秒です。

```json
{"clips/a.wav": {"language": "ja", "text": "2つ目。",
  "words": [["2", 0.0, 0.2], ["つ", 0.2, 0.36], ["目", 0.36, 0.46], ["。", 0.46, 1.32]]}}
```

ここでの「単語」は、完全な UTF-8 文字になる最短のトークン列です。日本語や中国語ならたいてい1文字、英語なら単語かその一部になります。開始は最初のトークンの時刻、終了は次のトークンの開始時刻で、時刻は Whisper のフレーム間隔（20ミリ秒）刻みです。

## 仕組み

| 段階 | ファイル | 内容 |
|---|---|---|
| 特徴量 | `audio.rs` | WAV → 16 kHz（多相の Kaiser 窓 sinc）→ `WhisperFeatureExtractor` と同じ log-mel（DFT の積算は f64）を全コアで計算 |
| エンコーダ | `model.rs`、`kernels/whisper.cu` | クリップをグループごとに処理。cuBLAS の行列積、flash attention のカーネル（`mma.sync`、`cp.async`）、残差加算と次の layer norm の融合。最後に全デコーダ層の cross-attention のキーとバリューを作ります |
| デコーダ | `model.rs`、`kernels/whisper.cu` | バッチ全体を1ステップ1つの CUDA グラフで実行（位置は GPU メモリから読むので同じグラフを毎ステップ再生）。小さいバッチには gemv カーネル。アライメントヘッドの注意を途中で記録 |
| タイムスタンプ | `align.rs` | 正規化、メディアンフィルタ、動的時間伸縮。`transformers` の `_extract_token_timestamps` と同じ手順を CPU で計算 |
| テキスト | `tok.rs`、`main.rs` | バイトレベル BPE を完全な文字の単位に復号し、JSON に出力 |

`transcribe.rs` がバッチごとにこれらの段階を実行し、`main.rs` がコマンドライン、`dev.rs` が下の `--ref` による照合を担当します。

デコードの1ステップで読むメモリの大半は cross-attention のキーとバリューなので、長い連続領域として読めるように並べてあります。1ステップ全体は一度 CUDA グラフに記録し、毎ステップ1回の起動で再生します。

`transformers` と結果をそろえるため、PyTorch が fp16 に丸める箇所（線形層の出力、残差加算、デコーダの注意のスコアと確率）ではカーネルも同じ箇所で丸めます。エンコーダの注意は PyTorch の SDPA と同じく融合して計算するため、エンコーダの出力はビット単位ではなく fp16 の誤差の範囲で一致します。貪欲選択の同点は `argmax` と同じく小さいトークン ID を選びます。

## 検証

| 確認内容 | 結果 |
|---|---|
| 基準クリップ（`scripts/hf_reference.py` で作成し `--ref` で照合）：large-v3（日本語指定と言語自動判定）、large-v3-turbo、small（fp32 のチェックポイント、80 mel、言語自動判定）、tiny.en | トークン列とトークンのタイムスタンプが一致 |
| 日本語ナレーション64クリップを `transformers` と比較 | 63クリップの書き起こしが一致。残る1つは1文字違い（`transformers` 自身もこのクリップでバッチと単独で結果が変わる） |
| 上記クリップの単語の時刻 1,663個 | 98% が一致、最大の差は0.06秒 |

基準クリップでの照合は、[uv](https://docs.astral.sh/uv/) で次のように再現できます（使うパッケージの版はスクリプトの先頭で固定しています）。

    uv run scripts/hf_reference.py --model DIR --language ja clip.wav --out ref.safetensors
    mimizuku --model DIR --language ja --ref ref.safetensors

各段階の差を表示し、トークン列が違えばエラーで終了します。

## 制限

- 30秒（1ウィンドウ）までのクリップだけです。長いファイルは断ります。長尺の文字起こしはありません。
- 貪欲デコードだけです。ビームサーチ、温度によるやり直し、プロンプト、タイムスタンプトークンはありません。
- 入力は WAV だけです（`ffmpeg -i in.mp3 out.wav`）。
- 推論は fp16 で行います（fp32 のチェックポイントは読み込み時に変換）。ヘッド幅64、エンコーダ位置1500のモデルが対象です（OpenAI の Whisper はすべて該当）。
- ライブラリではなくコマンドラインのツールです。GPU のバッファはプロセスの終了時に解放します。
- Linux と NVIDIA GPU のみです。

## セキュリティ

ネットワークには接続しません。読むのはモデルのディレクトリと WAV ファイルで、書くのは JSON だけです。モデルファイル、WAV、JSON は検証付きの読み込み処理を通します（オフセット、テンソルの形、トークン ID、入れ子の深さをファイルとモデルの設定に照らして確認）。

## ライセンス

[Apache License 2.0](LICENSE-APACHE) と [MIT](LICENSE-MIT) のどちらかを選べます。モデルの重みは含みません（Hugging Face 上の OpenAI の Whisper は MIT または Apache-2.0）。

mimizuku は OpenAI の Whisper、Hugging Face `transformers`（特徴量抽出とトークンのタイムスタンプの手順を踏襲）、cudarc の成果の上に作っています。`transformers` と Whisper に由来する部分は [NOTICE](NOTICE) にまとめています。
