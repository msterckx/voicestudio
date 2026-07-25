import os
from pathlib import Path
import runpod


VOLUME_ROOT = Path("/runpod-volume")
DATA_ROOT = VOLUME_ROOT / "voice-studio"

SAMPLES_DIR = DATA_ROOT / "samples"
OUTPUT_DIR = DATA_ROOT / "output"

# Make Hugging Face use the persistent volume.
os.environ["HF_HOME"] = str(
    DATA_ROOT / "cache" / "huggingface"
)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(
    DATA_ROOT / "cache" / "huggingface" / "hub"
)

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def handler(job):
    job_input = job["input"]

    text = job_input["text"]
    sample_name = job_input["sample"]

    sample_path = SAMPLES_DIR / sample_name

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Reference sample not found: {sample_path}"
        )

    job_id = job.get("id", "manual")

    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = job_output_dir / "output.wav"

    #
    # NEXT STEP:
    # Call Voice Clone Studio / Qwen generation here.
    #
    # generate(
    #     text=text,
    #     reference_audio=str(sample_path),
    #     output_path=str(output_path),
    # )
    #

    return {
        "sample": sample_name,
        "output": str(output_path)
    }


runpod.serverless.start({
    "handler": handler
})