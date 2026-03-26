#!/usr/bin/env python3
"""
Quantize Voxtral-Mini-4B-Realtime to FP8 for NVIDIA Blackwell / Ada+ GPUs.

Uses torchao's float8_weight_only quantization via HF TorchAoConfig.
This path uses PyTorch's native FP8 dtype (float8_e4m3fn) — no CUTLASS
or qutlass dependency, so it works on GB10 (SM120) where NVFP4 is broken.

IMPORTANT: torchao must match your PyTorch version. On GB10 / DGX Spark
with PyTorch 2.11+cu130, install from the PyTorch index:
    pip install torchao --index-url https://download.pytorch.org/whl/cu130
Or run inside the NGC container (Dockerfile.spark) where deps are matched.

Requirements:
    pip install transformers>=4.49.0 accelerate torch torchao

Usage:
    # Quantize and save locally (recommended for GB10)
    python scripts/quantize_fp8.py --output-dir ./models/Voxtral-Mini-4B-FP8

    # Push to Hugging Face Hub
    python scripts/quantize_fp8.py --push-to-hub your-user/Voxtral-Mini-4B-FP8
"""

import argparse
import sys
import torch


def _check_torchao():
    """Check torchao is installed and compatible with the current PyTorch."""
    try:
        import torchao  # noqa: F401
    except ImportError:
        return False, "not_installed"

    # Check if the C++ extensions loaded (they fail silently with a warning)
    try:
        from torchao.quantization import float8_weight_only  # noqa: F401
        return True, "ok"
    except ImportError:
        return False, "import_error"


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Voxtral to FP8 for Blackwell / Ada+ GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # FP8 quantization on Blackwell (GB10):\n"
            "  python scripts/quantize_fp8.py --output-dir ./models/Voxtral-FP8\n\n"
            "  # Push to Hugging Face Hub:\n"
            "  python scripts/quantize_fp8.py --push-to-hub your-user/Voxtral-FP8\n"
        ),
    )
    parser.add_argument(
        "--model-id",
        default="mistralai/Voxtral-Mini-4B-Realtime-2602",
        help="Source model ID on Hugging Face (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory to save the quantized model",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Hugging Face repo ID to push the quantized model",
    )
    args = parser.parse_args()

    if not args.output_dir and not args.push_to_hub:
        parser.error("Provide at least one of --output-dir or --push-to-hub")

    # --- Pre-flight checks ---
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. FP8 quantization requires an NVIDIA GPU.")
        sys.exit(1)

    ok, status = _check_torchao()
    if not ok:
        torch_ver = torch.__version__
        print("=" * 70)
        if status == "not_installed":
            print("ERROR: torchao is not installed.")
        else:
            print(f"ERROR: torchao is installed but incompatible with PyTorch {torch_ver}.")
        print()
        print("torchao must match your PyTorch version exactly. Install with:")
        print()
        if "cu130" in torch_ver:
            print("  # For PyTorch + CUDA 13.0 (GB10 / DGX Spark):")
            print("  pip install torchao --index-url https://download.pytorch.org/whl/cu130")
        else:
            print(f"  pip install torchao  # for PyTorch {torch_ver}")
        print()
        print("Or run this script inside the Docker container where deps are matched:")
        print("  docker compose run --rm voxtral-transcribe-spark \\")
        print("    python scripts/quantize_fp8.py --output-dir /app/models/Voxtral-FP8")
        print()
        print("Alternative: use bitsandbytes q4 quantization which works now:")
        print("  python main.py /data/audio.mp3 --precision q4 --flash-attn --compile")
        print("=" * 70)
        sys.exit(1)

    # --- Configure FP8 quantization via TorchAoConfig ---
    from transformers import (
        VoxtralRealtimeForConditionalGeneration,
        AutoProcessor,
        TorchAoConfig,
    )

    # float8_weight_only quantizes weights to float8_e4m3fn while keeping
    # activations in bf16/fp16. This gives ~2x memory reduction with
    # minimal quality loss and uses PyTorch native FP8 tensor cores.
    quant_config = TorchAoConfig("float8_weight_only")

    print(f"Loading model: {args.model_id}")
    print(f"Quantization: FP8 (torchao float8_weight_only)")
    print(f"  weight dtype:  float8_e4m3fn")
    print(f"  compute dtype: bfloat16")

    # --- Load model with FP8 quantization ---
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    print("Model loaded and quantized successfully.")

    # --- Save locally ---
    if args.output_dir:
        print(f"Saving quantized model to: {args.output_dir}")
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Done. Model saved to {args.output_dir}")

    # --- Push to Hub ---
    if args.push_to_hub:
        print(f"Pushing quantized model to: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        processor.push_to_hub(args.push_to_hub)
        print(f"Done. Model available at: https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
