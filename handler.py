import os
from pathlib import Path

import runpod
import soundfile as sf

# ---------------------------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------------------------

VOLUME_ROOT = Path("/runpod-volume")
DATA_ROOT = VOLUME_ROOT / "voice-studio"

SAMPLES_DIR = DATA_ROOT / "samples"
OUTPUT_DIR = DATA_ROOT / "output"

HF_HOME = DATA_ROOT / "cache" / "huggingface"
HF_HUB_CACHE = HF_HOME / "hub"

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HUB_CACHE)

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Voice Clone Studio imports
# ---------------------------------------------------------------------------

from modules.core_components.ai_models.tts_manager import get_tts_manager
from modules.core_components.tools import get_or_create_voice_prompt_standalone

# Keep one manager alive for the lifetime of the worker.
tts_manager = get_tts_manager()


def handler(job):
    job_input = job.get("input", {})

    text = job_input.get("text")
    sample_name = job_input.get("sample")
    sample_text = job_input.get("sample_text")

    language = job_input.get("language", "Auto")
    model_size = job_input.get("model_size", "1.7B")
    seed = int(job_input.get("seed", -1))

    if not text:
        raise ValueError('Missing required field "text"')

    if not sample_name:
        raise ValueError('Missing required field "sample"')

    if not sample_text:
        raise ValueError(
            'Missing required field "sample_text": '
            "provide the exact transcript of the reference audio."
        )

    sample_path = SAMPLES_DIR / sample_name

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Reference sample not found: {sample_path}"
        )

    job_id = str(job.get("id", "manual"))

    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = job_output_dir / "output.wav"

    print(f"Generating voice for job {job_id}", flush=True)
    print(f"Sample: {sample_path}", flush=True)
    print(f"Model: Qwen3 {model_size}", flush=True)

    # -----------------------------------------------------------------------
    # Load Qwen Base model
    # -----------------------------------------------------------------------

    model = tts_manager.get_qwen3_base(model_size)

    # -----------------------------------------------------------------------
    # Build or reuse the voice-clone prompt
    # -----------------------------------------------------------------------

    print("Preparing voice clone prompt...", flush=True)

    prompt_items, was_cached = get_or_create_voice_prompt_standalone(
        model=model,
        sample_name=sample_name,
        wav_path=str(sample_path),
        ref_text=sample_text,
        model_size=model_size,
        progress_callback=None,
    )

    print(
        f"Voice clone prompt ready; cached={was_cached}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Generate cloned speech
    # -----------------------------------------------------------------------

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

        prompt_items=prompt_items,
        user_config={},
        progress_callback=None,
    )

    # -----------------------------------------------------------------------
    # Write WAV
    # -----------------------------------------------------------------------

    print(
        f"Writing audio to {output_path}",
        flush=True,
    )

    sf.write(
        str(output_path),
        audio_data,
        sample_rate,
    )

    # -----------------------------------------------------------------------
    # Validate the output
    # -----------------------------------------------------------------------

    if not output_path.exists():
        raise RuntimeError(
            f"TTS generation completed but output file was not created: "
            f"{output_path}"
        )

    file_size = output_path.stat().st_size

    if file_size == 0:
        raise RuntimeError(
            f"Generated WAV is empty: {output_path}"
        )

    output_key = output_path.relative_to(VOLUME_ROOT).as_posix()

    print(
        f"Generation complete: {output_path} "
        f"({file_size} bytes, {sample_rate} Hz)",
        flush=True,
    )

    return {
        "sample": sample_name,
        "output_key": output_key,
        "sample_rate": sample_rate,
        "size_bytes": file_size,
    }


# ---------------------------------------------------------------------------
# Local debug mode / Serverless mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if os.environ.get("LOCAL_TEST") == "1":

        print("Running handler in LOCAL_TEST mode", flush=True)

        result = handler({
            "id": "local-test",
            "input": {
                "text": "This is a local test of my cloned voice.",
                "sample": "celeste_48k_stereo.wav",

                # IMPORTANT:
                # Replace this with the exact transcript of the
                # celeste_48k_stereo.wav reference recording.
                "sample_text": "At the edge of the northern forest, morning light drifts across ancient trees, revealing a quiet world shaped by time, memory, and the delicate balance between nature and change.",

                "language": "Auto",
                "model_size": "1.7B",
                "seed": -1,
            },
        })

        print("Local test result:", flush=True)
        print(result, flush=True)

    else:

        print("Starting Runpod Serverless worker", flush=True)

        runpod.serverless.start({
            "handler": handler
        })