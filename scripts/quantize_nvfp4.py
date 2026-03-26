#!/usr/bin/env python3
"""
Quantize Voxtral-Mini-4B-Realtime to NVFP4 for NVIDIA Blackwell GPUs.

WARNING — KNOWN ISSUE ON GB10 (SM120/SM121):
    NVFP4 quantization via qutlass is currently BROKEN on the GB10 chip
    (Asus Ascent GX10, DGX Spark). The CUTLASS FP4 GEMM kernels require
    128-256 KiB of shared memory but the GB10 (SM120) only provides 99 KiB.
    This causes "Error Internal" crashes in fusedQuantizeNvAbsMax.

    Use FP8 instead:  python scripts/quantize_fp8.py --output-dir ./models/Voxtral-FP8

    NVFP4 works on B200/B100 (SM100) which have sufficient shared memory.
    This script is kept for those GPUs and for when CUTLASS ships SM120-
    specific tile configs.

    Tracking issues:
    - https://github.com/NVIDIA/cutlass/issues/2800
    - https://github.com/NVIDIA/cutlass/issues/3096

IMPORTANT: Real NVFP4 quantization requires a Blackwell SM100 GPU (B200 / B100)
and the qutlass library. If you're not on Blackwell hardware, use --pseudo
to validate the workflow with Triton-based emulation (no speedup, but saves
a model that can be loaded on Blackwell later).

Requirements:
    pip install transformers>=4.52.0 accelerate torch

    # Real quantization (Blackwell only):
    sudo apt install python3-dev   # Python.h headers needed to compile qutlass
    pip install fp_quant
    git clone https://github.com/IST-DASLab/qutlass.git
    cd qutlass && pip install --no-build-isolation .

Usage:
    # ── On your Blackwell machine (GX10 / DGX Spark) ──
    python scripts/quantize_nvfp4.py --output-dir ./models/Voxtral-Mini-4B-NVFP4

    # ── On any machine (validation / pseudo-quantization) ──
    python scripts/quantize_nvfp4.py --pseudo --output-dir ./models/Voxtral-Mini-4B-NVFP4-pseudo

    # ── Push to Hugging Face Hub ──
    python scripts/quantize_nvfp4.py --push-to-hub your-user/Voxtral-Mini-4B-NVFP4
"""

import argparse
import sys
import torch
from transformers import (
    VoxtralRealtimeForConditionalGeneration,
    AutoProcessor,
    FPQuantConfig,
)


def _check_blackwell() -> bool:
    """Return True if a Blackwell GPU (compute capability >= 10.0) is available."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major >= 10


def _check_qutlass() -> bool:
    """Return True if qutlass is importable."""
    try:
        import qutlass  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Voxtral to NVFP4 for Blackwell",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Real quantization on Blackwell:\n"
            "  python scripts/quantize_nvfp4.py --output-dir ./models/Voxtral-NVFP4\n\n"
            "  # Pseudo-quantization on any hardware:\n"
            "  python scripts/quantize_nvfp4.py --pseudo --output-dir ./models/Voxtral-NVFP4-pseudo\n"
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
    parser.add_argument(
        "--pseudo",
        action="store_true",
        help="Use Triton-based pseudo-quantization (works on any GPU, no speedup)",
    )
    args = parser.parse_args()

    if not args.output_dir and not args.push_to_hub:
        parser.error("Provide at least one of --output-dir or --push-to-hub")

    # --- Pre-flight checks ---
    use_pseudo = args.pseudo

    # Warn about known SM120/SM121 issue (GB10)
    if not use_pseudo and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if props.major >= 12:
            print("=" * 70)
            print(f"WARNING: Detected {props.name} (SM{props.major}{props.minor}).")
            print("NVFP4 via qutlass is KNOWN BROKEN on SM120/SM121 (GB10).")
            print("The CUTLASS FP4 kernels need more shared memory than SM120 provides.")
            print()
            print("Use FP8 instead (works on GB10, no qutlass needed):")
            print("  python scripts/quantize_fp8.py --output-dir ./models/Voxtral-FP8")
            print()
            print("See: https://github.com/NVIDIA/cutlass/issues/2800")
            print("=" * 70)
            sys.exit(1)

    if not use_pseudo:
        has_blackwell = _check_blackwell()
        has_qutlass = _check_qutlass()

        if not has_blackwell and not has_qutlass:
            print("=" * 70)
            print("ERROR: Real NVFP4 quantization requires:")
            print("  1. A Blackwell GPU (compute capability >= 10.0)")
            print("  2. The qutlass library")
            print()
            print("You don't appear to have either. Your options:")
            print()
            print("  a) Run this script on your Blackwell machine (GX10 / DGX Spark)")
            print("     after installing dependencies:")
            print("       sudo apt install python3-dev")
            print("       pip install fp_quant")
            print("       git clone https://github.com/IST-DASLab/qutlass.git")
            print("       cd qutlass && pip install --no-build-isolation .")
            print()
            print("  b) Use --pseudo to create a pseudo-quantized model that")
            print("     emulates NVFP4 behavior (useful for testing the pipeline):")
            print(f"       python scripts/quantize_nvfp4.py --pseudo --output-dir {args.output_dir or './models/Voxtral-NVFP4-pseudo'}")
            print()
            print("  c) Run quantization inside the Docker container on your GX10:")
            print("       docker run --gpus all -v ./models:/app/output voxtral-spark \\")
            print("         python scripts/quantize_nvfp4.py --output-dir /app/output/Voxtral-NVFP4")
            print("=" * 70)
            sys.exit(1)

        if not has_blackwell:
            print(f"WARNING: No Blackwell GPU detected (CUDA available: {torch.cuda.is_available()}).")
            print("         Falling back to --pseudo mode automatically.")
            print("         The saved model will work but won't have real NVFP4 speedups.")
            print()
            use_pseudo = True

        if has_blackwell and not has_qutlass:
            print("ERROR: Blackwell GPU detected but qutlass is not installed.")
            print("       Install it with:")
            print("         sudo apt install python3-dev  # Python.h headers for compilation")
            print("         pip install fp_quant")
            print("         git clone https://github.com/IST-DASLab/qutlass.git")
            print("         cd qutlass && pip install --no-build-isolation .")
            sys.exit(1)

    # --- Step 1: Configure NVFP4 quantization ---
    mode_label = "pseudo" if use_pseudo else "real"

    # Modules to keep in higher precision (bf16).
    # The multi-modal projector bridges the audio encoder and the LM —
    # its layers are small and have tensor shapes incompatible with the
    # NVFP4 CUTLASS GEMM kernels.  The LM head (lm_head) is typically
    # excluded from quantization as well to preserve output quality.
    modules_to_skip = [
        "multi_modal_projector",
        "lm_head",
    ]

    quant_config = FPQuantConfig(
        forward_dtype="nvfp4",
        forward_method="abs_max",
        hadamard_group_size=16,
        pseudoquantization=use_pseudo,
        modules_to_not_convert=modules_to_skip,
    )

    print(f"Loading model: {args.model_id}")
    print(f"Quantization: NVFP4 ({mode_label})")
    print(f"  forward_dtype:       nvfp4")
    print(f"  forward_method:      abs_max")
    print(f"  hadamard_group_size: 16")
    print(f"  pseudoquantization:  {use_pseudo}")
    print(f"  modules_to_not_convert: {modules_to_skip}")

    # --- Step 2: Load model with NVFP4 quantization applied on the fly ---
    # Pre-import fp_quant to give a clear error before transformers does its
    # own opaque ImportError inside validate_environment().
    if not use_pseudo:
        try:
            import fp_quant  # noqa: F401
        except ImportError:
            print("ERROR: fp_quant is not installed. Required for real NVFP4 quantization.")
            print("       pip install fp_quant")
            sys.exit(1)

        if not _check_qutlass():
            print("ERROR: qutlass is not installed. Required for real NVFP4 quantization.")
            print("       git clone https://github.com/IST-DASLab/qutlass.git")
            print("       cd qutlass && pip install --no-build-isolation .")
            sys.exit(1)

    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Also load the processor (tokenizer + audio feature extractor)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"Model loaded and quantized ({mode_label}) successfully.")

    # --- Step 3: Save locally ---
    if args.output_dir:
        print(f"Saving quantized model to: {args.output_dir}")
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Done. Model saved to {args.output_dir}")
        if use_pseudo:
            print()
            print("NOTE: This model was pseudo-quantized. For real NVFP4 performance,")
            print("      re-run this script on your Blackwell hardware without --pseudo.")

    # --- Step 4: Push to Hugging Face Hub ---
    if args.push_to_hub:
        print(f"Pushing quantized model to: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        processor.push_to_hub(args.push_to_hub)
        print(f"Done. Model available at: https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
