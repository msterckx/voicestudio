import os
from pathlib import Path

import runpod
import soundfile as sf

VOLUME_ROOT = Path("/runpod-volume")
DATA_ROOT = VOLUME_ROOT / "voice-studio"

SAMPLES_DIR = DATA_ROOT / "samples"
OUTPUT_DIR = DATA_ROOT / "output"

os.environ["HF_HOME"] = str(DATA_ROOT / "cache" / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(
    DATA_ROOT / "cache" / "huggingface" / "hub"
)

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Important: import after HF environment variables are set.
from modules.core_components.ai_models.tts_manager import get_tts_manager

# Singleton. Models are lazy-loaded and then retained while the worker is warm.
tts_manager = get_tts_manager()


def handler(job):
    job_input = job["input"]

    text = job_input["text"]
    sample_name = job_input["sample"]
    sample_text = job_input["sample_text"]

    language = job_input.get("language", "Auto")
    model_size = job_input.get("model_size", "1.7B")
    seed = int(job_input.get("seed", -1))

    sample_path = SAMPLES_DIR / sample_name

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Reference sample not found: {sample_path}"
        )

    if not sample_text.strip():
        raise ValueError("sample_text is required for Qwen voice cloning")

    job_id = job.get("id", "manual")

    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = job_output_dir / "output.wav"

    print(f"Generating voice for job {job_id}")
    print(f"Sample: {sample_path}")
    print(f"Model: Qwen3 {model_size}")

    audio_data, sample_rate = tts_manager.generate_voice_clone_dispatch(
        text=text,
        engine="qwen",
        model_size=model_size,
        sample_wav_path=str(sample_path),
        sample_name=sample_name,
        sample_ref_text=sample_text,
        language=language,
        seed=seed,
        qwen_params={
            "do_sample": True,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "max_new_tokens": 2048,
        },

        # The standard Qwen path normally uses a precomputed prompt.
        # Faster-Qwen3-TTS can use ref_audio/ref_text directly.
        prompt_items=None,
        user_config={},
        progress_callback=None,
    )

    sf.write(
        str(output_path),
        audio_data,
        sample_rate,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Generation completed but output file does not exist: {output_path}"
        )

    file_size = output_path.stat().st_size

    if file_size == 0:
        raise RuntimeError("Generated WAV is empty")

    output_key = output_path.relative_to(VOLUME_ROOT).as_posix()

    print(f"Generated: {output_path}")
    print(f"Size: {file_size} bytes")

    return {
        "sample": sample_name,
        "output_key": output_key,
        "sample_rate": sample_rate,
        "size_bytes": file_size,
    }


if __name__ == "__main__":
    if os.environ.get("LOCAL_TEST") == "1":
        result = handler({
            "id": "local-test",
            "input": {
                "text": "This is a local test.",
                "sample": "celeste_48k_stereo.wav"
            }
        })

        print(result)
    else:
        runpod.serverless.start({
            "handler": handler
        })