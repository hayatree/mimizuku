# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "torch==2.10.0",
#     "transformers==5.3.0",
#     "librosa==0.11.0",
#     "safetensors",
#     "numpy",
# ]
# ///
"""Reference values for `mimizuku --ref`, from Hugging Face transformers on the same clip.

    uv run scripts/hf_reference.py --model DIR --language ja clip.wav --out ref.safetensors
    mimizuku --model DIR --language ja --ref ref.safetensors

Saves the 16 kHz audio librosa produced, the input features, every encoder layer's input, the encoder output,
the greedy sequence and its token timestamps (fp16 model, return_token_timestamps=True), so the engine can be
checked stage by stage. Runs on a CUDA GPU.
"""
import argparse

import librosa
import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--model", required=True, help="model directory (Hugging Face format)")
ap.add_argument("--language", default=None, help="language code such as ja; omit to let the model detect it")
ap.add_argument("--task", default="transcribe")
ap.add_argument("--out", required=True)
a = ap.parse_args()

proc = WhisperProcessor.from_pretrained(a.model)
model = WhisperForConditionalGeneration.from_pretrained(a.model, dtype=torch.float16, use_safetensors=True).to("cuda").eval()
y, _ = librosa.load(a.wav, sr=16000, mono=True)
x = proc(y, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
feat = x.input_features.to("cuda", torch.float16)
multilingual = model.generation_config.is_multilingual
lang = {"language": a.language, "task": a.task} if multilingual else {}
with torch.no_grad():
    eo = model.model.encoder(feat, output_hidden_states=True)
    g = model.generate(feat, attention_mask=x.attention_mask.to("cuda"), return_token_timestamps=True, return_dict_in_generate=True, **lang)
g = g if isinstance(g, dict) else vars(g)
seq = g["sequences"][0]
print("sequence", seq.tolist())
print("text", proc.tokenizer.decode(seq, skip_special_tokens=True))
save_file({"audio16k": y.astype(np.float32),
           "features": x.input_features[0].float().numpy(),
           "num_frames": np.array([int(x.attention_mask.sum())], dtype=np.int32),
           "encoder": eo.last_hidden_state[0].float().cpu().numpy(),
           # input of each encoder layer (index 0: after the convolutions and positions), fp16 as computed
           "enc_hidden": torch.stack([h[0] for h in eo.hidden_states[:-1]]).cpu().numpy(),
           "sequence": seq.cpu().numpy().astype(np.int32),
           "token_timestamps": g["token_timestamps"][0].float().cpu().numpy()}, a.out, metadata={"wav": a.wav})
print("saved", a.out)
