import os
import torch
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)

# Device selection
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Download model files from HuggingFace
repo_id = "ai4bharat/IndicF5"
vocab_path = hf_hub_download(repo_id, filename="checkpoints/vocab.txt")
ckpt_path = hf_hub_download(repo_id, filename="model.safetensors")

# Load vocoder
print("Loading vocoder...")
vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

# Load model
print("Loading model...")
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    mel_spec_type="vocos",
    vocab_file=vocab_path,
    device=device,
)

# Load checkpoint weights (filter to ema_model keys and strip prefix)
state_dict = load_file(ckpt_path, device=device)
state_dict = {
    k.replace("ema_model._orig_mod.", ""): v
    for k, v in state_dict.items()
    if k.startswith("ema_model.")
}
model.load_state_dict(state_dict)
model.eval()

# Optimize for speed (optional)
if device == "cuda":
    model = model.half()  # Use float16 on CUDA for ~2x speedup
    vocoder = vocoder.half()

# Prepare inputs
ref_audio_path = "prompts/PAN_F_HAPPY_00001.wav"
ref_text = "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
gen_text = "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए."

# Preprocess reference audio
print("Preprocessing reference audio...")
ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text, device=device)

# Generate speech
# nfe_step: diffusion steps (default 32). Lower = faster but lower quality. Try 16 for 2x speed.
print("Generating speech...")
audio, sample_rate, _ = infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model,
    vocoder,
    mel_spec_type="vocos",
    nfe_step=32,  # Reduce from 32 to 16 for faster inference
    device=device,
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
os.makedirs("samples", exist_ok=True)
sf.write("samples/namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
print("Audio saved to samples/namaste.wav")
