#!/usr/bin/env python3
"""
Pre-download a Hugging Face model into the local cache.

Used at Docker build time so the container starts instantly at runtime.
Controlled by environment variables:

    MODEL_ID   — HF model ID (default: mistralai/Voxtral-Mini-4B-Realtime-2602)
    HF_HOME    — cache directory (default: /app/models)

Usage:
    # In Dockerfile:
    RUN python scripts/download_model.py

    # Or manually:
    MODEL_ID=mistralai/Voxtral-Mini-4B-Realtime-2602 python scripts/download_model.py
"""

import os
import sys


def main():
    model_id = os.environ.get("MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602")
    hf_home = os.environ.get("HF_HOME", "/app/models")

    print(f"Pre-downloading model: {model_id}")
    print(f"  Cache directory: {hf_home}")

    os.environ["HF_HOME"] = hf_home

    # Import here so HF_HOME is set before transformers reads it
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    # Download processor (tokenizer + audio feature extractor)
    print("  Downloading processor...")
    AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Download model weights (CPU-only — no GPU needed at build time)
    # AutoModelForSpeechSeq2Seq resolves the right class from config.json,
    # so this works for both VoxtralRealtime (4B) and VoxtralSmall (24B).
    print("  Downloading model weights...")
    AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"  Done. Model cached in {hf_home}")


if __name__ == "__main__":
    main()
