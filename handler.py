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


def _normalize_generations(job_input):
    """Return a list of per-generation request dicts.

    Supports two input shapes:

    * A ``generations`` list, where each item holds its own
      ``text`` / ``sample`` / ``sample_text`` and optionally overrides
      ``language`` / ``model_size`` / ``seed``.
    * The legacy single-generation shape, where those fields live at the
      top level of ``input``.

    Top-level ``language`` / ``model_size`` / ``seed`` act as defaults that
    each generation inherits unless it provides its own value.
    """

    default_language = job_input.get("language", "Auto")
    default_model_size = job_input.get("model_size", "1.7B")
    default_seed = job_input.get("seed", -1)

    raw_generations = job_input.get("generations")

    if raw_generations is None:
        # Legacy single-generation request.
        raw_generations = [{
            "text": job_input.get("text"),
            "sample": job_input.get("sample"),
            "sample_text": job_input.get("sample_text"),
        }]
    elif not isinstance(raw_generations, list):
        raise ValueError('Field "generations" must be a list')
    elif not raw_generations:
        raise ValueError('Field "generations" must not be empty')

    generations = []

    for gen in raw_generations:
        if not isinstance(gen, dict):
            raise ValueError("Each generation must be an object")

        generations.append({
            "text": gen.get("text"),
            "sample": gen.get("sample"),
            "sample_text": gen.get("sample_text"),
            "language": gen.get("language", default_language),
            "model_size": gen.get("model_size", default_model_size),
            "seed": int(gen.get("seed", default_seed)),
        })

    return generations


def _generate_one(spec, index, job_output_dir):
    """Run a single voice-clone generation and write its WAV to disk."""

    text = spec["text"]
    sample_name = spec["sample"]
    sample_text = spec["sample_text"]
    language = spec["language"]
    model_size = spec["model_size"]
    seed = spec["seed"]

    if not text:
        raise ValueError(f'Generation {index}: missing required field "text"')

    if not sample_name:
        raise ValueError(
            f'Generation {index}: missing required field "sample"'
        )

    if not sample_text:
        raise ValueError(
            f'Generation {index}: missing required field "sample_text": '
            "provide the exact transcript of the reference audio."
        )

    sample_path = SAMPLES_DIR / sample_name

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Generation {index}: reference sample not found: {sample_path}"
        )

    output_path = job_output_dir / f"output_{index}.wav"

    print(f"Generation {index}: sample {sample_path}", flush=True)
    print(f"Generation {index}: model Qwen3 {model_size}", flush=True)

    # -----------------------------------------------------------------------
    # Load Qwen Base model
    # -----------------------------------------------------------------------

    model = tts_manager.get_qwen3_base(model_size)

    # -----------------------------------------------------------------------
    # Build or reuse the voice-clone prompt
    # -----------------------------------------------------------------------

    print(f"Generation {index}: preparing voice clone prompt...", flush=True)

    prompt_items, was_cached = get_or_create_voice_prompt_standalone(
        model=model,
        sample_name=sample_name,
        wav_path=str(sample_path),
        ref_text=sample_text,
        model_size=model_size,
        progress_callback=None,
    )

    print(
        f"Generation {index}: voice clone prompt ready; cached={was_cached}",
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

    print(f"Generation {index}: writing audio to {output_path}", flush=True)

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
            f"Generation {index}: TTS generation completed but output file "
            f"was not created: {output_path}"
        )

    file_size = output_path.stat().st_size

    if file_size == 0:
        raise RuntimeError(
            f"Generation {index}: generated WAV is empty: {output_path}"
        )

    output_key = output_path.relative_to(VOLUME_ROOT).as_posix()

    print(
        f"Generation {index}: complete: {output_path} "
        f"({file_size} bytes, {sample_rate} Hz)",
        flush=True,
    )

    return {
        "sample": sample_name,
        "output_key": output_key,
        "sample_rate": sample_rate,
        "size_bytes": file_size,
    }


def handler(job):
    job_input = job.get("input", {})

    generations = _normalize_generations(job_input)

    job_id = str(job.get("id", "manual"))

    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating voice for job {job_id} "
        f"({len(generations)} generation(s))",
        flush=True,
    )

    results = [
        _generate_one(spec, index, job_output_dir)
        for index, spec in enumerate(generations)
    ]

    # Preserve the original single-object response shape for legacy
    # single-generation requests, while returning a list for batches.
    if job_input.get("generations") is None:
        return results[0]

    return {"generations": results}


# ---------------------------------------------------------------------------
# Local debug mode / Serverless mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if os.environ.get("LOCAL_TEST") == "1":

        print("Running handler in LOCAL_TEST mode", flush=True)

        result = handler({
            "id": "local-test",
            "input": {
                # Shared defaults inherited by every generation below.
                "language": "Auto",
                "model_size": "1.7B",
                "seed": -1,

                # IMPORTANT:
                # Replace "sample_text" with the exact transcript of the
                # celeste_48k_stereo.wav reference recording.
                "generations": [
                    {
                        "text": "This is the first line of my cloned voice.",
                        "sample": "celeste_48k_stereo.wav",
                        "sample_text": "At the edge of the northern forest, morning light drifts across ancient trees, revealing a quiet world shaped by time, memory, and the delicate balance between nature and change.",
                    },
                    {
                        "text": "And this is a second, separate generation.",
                        "sample": "celeste_48k_stereo.wav",
                        "sample_text": "At the edge of the northern forest, morning light drifts across ancient trees, revealing a quiet world shaped by time, memory, and the delicate balance between nature and change.",
                    },
                ],
            },
        })

        print("Local test result:", flush=True)
        print(result, flush=True)

    else:

        print("Starting Runpod Serverless worker", flush=True)

        runpod.serverless.start({
            "handler": handler
        })