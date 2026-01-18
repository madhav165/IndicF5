import os
import torch
import torchaudio
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    chunk_text,
    convert_char_to_pinyin,
    target_sample_rate,
    hop_length,
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


def stream_tts(
    gen_text: str,
    ref_audio_path: str,
    ref_text: str,
    model,
    vocoder,
    device: str,
    speed: float = 1.0,
    target_rms: float = 0.1,
    max_chars: int = None,
):
    """
    Streaming TTS generator that yields audio chunks as they are generated.

    Args:
        max_chars: Maximum characters per chunk. Lower = faster streaming, smaller chunks.
                   Default None uses auto-calculation. Try 50-100 for faster streaming.

    Yields:
        tuple: (audio_chunk as np.ndarray, sample_rate)
    """
    # Preprocess reference audio
    ref_audio_file, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text, device=device)

    # Load and prepare reference audio
    audio, sr = torchaudio.load(ref_audio_file)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    # Use provided max_chars or calculate from reference audio
    if max_chars is None:
        max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    ref_audio_len = audio.shape[-1] // hop_length

    # Generate and yield each chunk
    for i, chunk_text_content in enumerate(gen_text_batches):
        print(f"Generating chunk {i + 1}/{len(gen_text_batches)}...")

        # Prepare the text
        text_list = [ref_text + chunk_text_content]
        final_text_list = convert_char_to_pinyin(text_list)

        # Calculate duration
        ref_text_len = len(ref_text.encode("utf-8"))
        gen_text_len = len(chunk_text_content.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        # Inference
        with torch.inference_mode():
            generated, _ = model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1,
            )

            generated = generated.to(torch.float32)
            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = generated.permute(0, 2, 1)
            generated_wave = vocoder.decode(generated_mel_spec)

            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # Convert to numpy
            audio_chunk = generated_wave.squeeze().cpu().numpy()

            yield audio_chunk, target_sample_rate


# Example usage
if __name__ == "__main__":
    import sounddevice as sd

    ref_audio_path = "prompts/PAN_F_HAPPY_00001.wav"
    ref_text = "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
    gen_text = """నిశ్శబ్దం మాట్లాడుతుంది.
గాలి చెవిలో రహస్యాలు చెబుతుంది.
చెట్టు ఆకుల మధ్య
సూర్యుడు తన చిరునవ్వు దాచుకుంటాడు.
నడిచే అడుగుల్లో కాలం
తన జాడలు వదిలి వెళ్తుంది.
మనసు మాత్రం
ఒక్క క్షణాన్ని నిత్యంగా చేసుకుంటుంది."""

    os.makedirs("samples", exist_ok=True)

    # Streaming mode - play and save chunks as they come
    # Use max_chars=50 for smaller, faster chunks (adjust as needed)
    all_chunks = []
    for i, (audio_chunk, sample_rate) in enumerate(stream_tts(
        gen_text, ref_audio_path, ref_text, model, vocoder, device, max_chars=50
    )):
        # Play audio chunk immediately
        sd.play(audio_chunk, sample_rate)

        # Save chunk while it's playing
        chunk_path = f"samples/chunk_{i}.wav"
        sf.write(chunk_path, audio_chunk.astype(np.float32), samplerate=sample_rate)
        print(f"  Playing & saved {chunk_path}")
        all_chunks.append(audio_chunk)

        # Wait for playback to finish before next chunk
        sd.wait()

    # Also save combined audio
    combined_audio = np.concatenate(all_chunks)
    sf.write("samples/combined.wav", combined_audio.astype(np.float32), samplerate=24000)
    print("\nAll chunks saved. Combined audio: samples/combined.wav")
