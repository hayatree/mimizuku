<h1 align="center">mimizuku</h1>

<p align="center">Batched <a href="https://github.com/openai/whisper">Whisper</a> transcription with token timestamps, in Rust and CUDA.</p>

<p align="center"><a href="#quick-start"><b>Quick start</b></a> · <a href="#usage">Usage</a> · <a href="#how-it-works">How it works</a> · <a href="#verification">Verification</a> · <a href="#limitations">Limitations</a></p>

<p align="center">
  <a href="https://github.com/hayatree/mimizuku/actions/workflows/ci.yml"><img src="https://github.com/hayatree/mimizuku/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue" alt="License: MIT OR Apache-2.0"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.ja.md">日本語</a></sub></p>

It reads Hugging Face checkpoints, decodes a batch of clips greedily, and gives every character or word piece a
start and end time from the alignment heads, as `transformers` does.

- **Matches `transformers`** (fp16): identical tokens and token timestamps on every reference clip checked (see
  [Verification](#verification)).
- **Batched, with timestamps**: a whole batch of clips is decoded together, and every clip gets token timestamps.
- **Small**: about 3,300 lines and one dependency ([cudarc](https://github.com/coreylowman/cudarc)). WAV,
  resampling, log-mel features, safetensors, JSON and the tokenizer are implemented in-tree.

Status: 0.1. Clips of up to 30 s and greedy decoding; see [Limitations](#limitations).

## Quick start

You need Rust, a CUDA toolkit and an NVIDIA GPU (see [Build](#build)), plus [uv](https://docs.astral.sh/uv/) to download the model:

    git clone https://github.com/hayatree/mimizuku && cd mimizuku
    cargo build --release --locked

    uvx --from huggingface_hub hf download openai/whisper-large-v3-turbo \
        model.safetensors config.json generation_config.json vocab.json \
        --local-dir models/whisper-large-v3-turbo

    ffmpeg -i speech.mp3 speech.wav     # any WAV of up to 30 s
    ./target/release/mimizuku --model models/whisper-large-v3-turbo speech.wav

The transcript and per-token times are printed as JSON; see [Usage](#usage) for the format and options.

## Build

Requirements: Linux, an NVIDIA GPU of the Ampere generation or newer (sm_80+), a CUDA toolkit supported by cudarc
0.19 (`nvcc`; cuBLAS is loaded at run time), and Rust 1.88+.

    cargo build --release --locked      # the binary is target/release/mimizuku

| variable | default | |
|---|---|---|
| `NVCC`, `CUDA_HOME` | `nvcc` on `PATH`, then `/usr/local/cuda` | CUDA compiler for the kernels; cudarc still runs `nvcc --version` from `PATH` unless `CUDARC_CUDA_VERSION` is set |
| `CUDA_ARCH` | `compute_80` | PTX architecture; the driver JIT-compiles it for the GPU |
| `CUDA_SASS` | none | also embed machine code for these GPUs, e.g. `86,89,121`, to skip the JIT (useful in containers) |
| `CUDARC_CUDA_VERSION` | from `nvcc --version` | CUDA API version of the bindings, e.g. `12080` |

Developed and tested on an NVIDIA DGX Spark (GB10, sm_121, CUDA 13.0, aarch64). CI builds and runs the CPU tests on
x86-64 with CUDA 12.8, without a GPU. Reports from other GPUs are welcome.

## Usage

The model directory is a Hugging Face snapshot with `model.safetensors`, `config.json`, `generation_config.json` and
`vocab.json`, e.g. `openai/whisper-large-v3` or `openai/whisper-large-v3-turbo`.

    mimizuku --model ~/models/whisper-large-v3 --language ja clip1.wav clip2.wav > out.json
    mimizuku --model ~/models/whisper-large-v3-turbo --dir clips/ --out out.json

| option | |
|---|---|
| `--model DIR` | model directory (required) |
| `--dir DIR [--prefix STR]` | every `.wav` in DIR (sorted), optionally only names starting with STR |
| `--language CODE` | `en`, `ja`, ...; `auto` (default) detects it per clip |
| `--task NAME` | `transcribe` (default) or `translate` (into English) |
| `--out FILE` | write the JSON there instead of standard output |
| `--batch N` | clips decoded together (default 32; large-v3 at 32 needs about 14.5 GB of GPU memory) |
| `--enc-group N` | clips per encoder pass (default 8) |
| `--quiet` | no progress or timing on standard error |

English-only models (`*.en`) take only `--language en` or `auto` and `--task transcribe`.

Input: WAV, 16/24/32-bit PCM or 32-bit float, any number of channels (mixed down), 1–768 kHz, up to 30 s.

Output, in input order; times in seconds:

```json
{"clips/a.wav": {"language": "ja", "text": "2つ目。",
  "words": [["2", 0.0, 0.2], ["つ", 0.2, 0.36], ["目", 0.36, 0.46], ["。", 0.46, 1.32]]}}
```

A "word" is the shortest run of tokens that forms whole UTF-8 characters: usually a character in Japanese or Chinese,
a word or part of one in English. It starts at its first token's time and ends where the next token starts; times
fall on Whisper's 20 ms frame grid.

## How it works

| step | files | |
|---|---|---|
| features | `audio.rs` | WAV → 16 kHz (polyphase Kaiser-windowed sinc) → `WhisperFeatureExtractor`'s log-mel (DFT accumulated in f64), on all cores |
| encoder | `model.rs`, `kernels/whisper.cu` | groups of clips; cuBLAS GEMMs, a flash-attention kernel (`mma.sync`, `cp.async`), residual add fused with the next layer norm; then every decoder layer's cross-attention keys and values |
| decoder | `model.rs`, `kernels/whisper.cu` | one CUDA graph per step for the batch (the position is read from device memory, so the same graph replays every step); gemv kernels for small batches; alignment-head attention recorded on the way |
| timestamps | `align.rs` | normalisation, median filter and dynamic time warping, as `transformers`' `_extract_token_timestamps`, on the CPU |
| text | `tok.rs`, `main.rs` | byte-level BPE decoding into whole-character pieces; JSON output |

`transcribe.rs` runs these stages for a batch, `main.rs` is the command line, and `dev.rs` holds the `--ref` check
below.

Most of the memory a decoding step reads is the cross-attention keys and values, so they are laid out to be read as
long sequential runs. The whole step is captured once as a CUDA graph and replayed with one launch per step.

To follow `transformers`, the kernels round to fp16 where PyTorch does (linear outputs, residual adds, the decoder's
attention scores and probabilities). Encoder attention is fused like PyTorch's SDPA, so the encoder output agrees to
fp16 noise rather than bit for bit. Greedy ties go to the lower token id, as `argmax` does.

## Verification

| check | result |
|---|---|
| reference clips (`scripts/hf_reference.py`, then `--ref`): large-v3 with Japanese and with language detection, large-v3-turbo, small (fp32 checkpoint, 80 mel bins, language detection), tiny.en | identical token sequences and token timestamps |
| 64 Japanese narration clips against `transformers` | 63 transcripts identical; one differs by a character, as `transformers` itself does between batched and single runs of that clip |
| 1,663 word timings of those clips | 98% identical, largest difference 0.06 s |

To repeat the reference check with [uv](https://docs.astral.sh/uv/) (the script pins the versions it was checked with):

    uv run scripts/hf_reference.py --model DIR --language ja clip.wav --out ref.safetensors
    mimizuku --model DIR --language ja --ref ref.safetensors

It prints the difference at every stage and exits with an error if the token sequence differs.

## Limitations

- Clips of up to 30 s (one window); longer files are rejected. No long-form transcription.
- Greedy decoding only: no beam search, temperature fallback, prompts or timestamp tokens.
- WAV input only (`ffmpeg -i in.mp3 out.wav`).
- Inference in fp16 (fp32 checkpoints are converted on load); models need 64-wide heads and 1500 encoder positions
  (every OpenAI Whisper size has them).
- A command-line tool, not a library: GPU buffers are freed when the process exits.
- Linux and NVIDIA GPUs only.

## Security

The engine makes no network access: it reads the model directory and the WAV files, and writes JSON. Model files,
WAV and JSON go through validating readers (offsets, tensor shapes, token ids and nesting depth are checked against
the file and the model configuration).

## License

[Apache License 2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at your option. Model weights are not included (OpenAI's
Whisper checkpoints on Hugging Face are MIT or Apache-2.0).

mimizuku builds on OpenAI's Whisper, Hugging Face `transformers` (whose feature extraction and token timestamps it
follows) and cudarc; [NOTICE](NOTICE) lists the parts derived from `transformers` and Whisper.
