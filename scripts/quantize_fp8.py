#!/usr/bin/env python3
"""
Quantize Voxtral-Mini-4B-Realtime to FP8 for NVIDIA Blackwell / Ada+ GPUs.

FP8 (E4M3) uses native PyTorch FP8 tensor-core support — no qutlass or
external CUDA kernels required. This is the recommended quantization path
for the GB10 (Asus Ascent GX10 / DGX Spark) where NVFP4 is currently
broken due to CUTLASS SM120 shared memory constraints.

Requirements:
    pip install transformers>=4.52.0 accelerate torch fp_quant

Usage:
    # Quantize and save locally (recommended for GB10)
    python scripts/quantize_fp8.py --output-dir ./models/Voxtral-Mini-4B-FP8

    # Push to Hugging Face Hub
    python scripts/quantize_fp8.py --push-to-hub your-user/Voxtral-Mini-4B-FP8

    # Pseudo-quantization on any hardware (for pipeline validation)
    python scripts/quantize_fp8.py --pseudo --output-dir ./models/Voxtral-Mini-4B-FP8-pseudo
"""

import argparse
import sys
import torch
from transformers import (
    VoxtralRealtimeForConditionalGeneration,
    AutoProcessor,
    FPQuantConfig,
)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Voxtral to FP8 for Blackwell / Ada+ GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Real FP8 quantization on Blackwell (GB10):\n"
            "  python scripts/quantize_fp8.py --output-dir ./models/Voxtral-FP8\n\n"
            "  # Pseudo-quantization on any hardware:\n"
            "  python scripts/quantize_fp8.py --pseudo --output-dir ./models/Voxtral-FP8-pseudo\n"
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
        help="Use pseudo-quantization (works on any GPU, no speedup)",
    )
    args = parser.parse_args()

    if not args.output_dir and not args.push_to_hub:
        parser.error("Provide at least one of --output-dir or --push-to-hub")

    use_pseudo = args.pseudo

    # --- Pre-flight checks ---
    if not use_pseudo:
        if not torch.cuda.is_available():
            print("ERROR: No CUDA GPU detected. FP8 quantization requires an NVIDIA GPU.")
            print("       Use --pseudo for pipeline validation without a GPU.")
            sys.exit(1)

        try:
            import fp_quant  # noqa: F401
        except ImportError:
            print("ERROR: fp_quant is not installed. Required for FP8 quantization.")
            print("       pip install fp_quant")
            sys.exit(1)

    # --- Configure FP8 quantization ---
    mode_label = "pseudo" if use_pseudo else "real"

    # Keep projector and LM head in higher precision — same rationale as NVFP4:
    # small layers with shapes that may not align to FP8 tile requirements,
    # and they are critical for output quality.
    modules_to_skip = [
        "multi_modal_projector",
        "lm_head",
    ]

    quant_config = FPQuantConfig(
        forward_dtype="fp8",
        forward_method="abs_max",
        pseudoquantization=use_pseudo,
        modules_to_not_convert=modules_to_skip,
    )

    print(f"Loading model: {args.model_id}")
    print(f"Quantization: FP8 ({mode_label})")
    print(f"  forward_dtype:          fp8")
    print(f"  forward_method:         abs_max")
    print(f"  pseudoquantization:     {use_pseudo}")
    print(f"  modules_to_not_convert: {modules_to_skip}")

    # --- Load model with FP8 quantization ---
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"Model loaded and quantized ({mode_label}) successfully.")

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
