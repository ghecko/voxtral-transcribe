import torch
import logging
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.utils import logging as hf_logging
import numpy as np
from typing import Optional, List

from core.platform import (
    detect_platform,
    get_optimal_device_map,
    get_torch_dtype,
    supports_nvfp4,
    platform_summary,
)

# Suppress the "layers not sharded" accelerate warning via standard logging
logging.getLogger("accelerate.big_modeling").setLevel(logging.ERROR)


def _is_prequantized(model_id: str) -> str | None:
    """
    Detect if a model checkpoint is already pre-quantized from its HF config.
    Returns the quant type string ("fp8", "gptq", "awq", etc.) or None.

    Checks:
      1. Model ID heuristics (e.g. "FP8" in name)
      2. HF config.json quantization_config field (if downloadable)
    """
    if not model_id:
        return None

    # Fast heuristic: well-known naming conventions
    model_lower = model_id.lower()
    for tag in ("fp8", "-fp8-", "_fp8_", "fp8-dynamic"):
        if tag in model_lower:
            return "fp8"
    for tag in ("gptq", "awq", "gguf"):
        if tag in model_lower:
            return tag

    # Slow path: peek at config.json from HF Hub
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        qc = getattr(config, "quantization_config", None)
        if qc:
            quant_method = qc.get("quant_method", None) if isinstance(qc, dict) else getattr(qc, "quant_method", None)
            if quant_method:
                return quant_method
    except Exception:
        pass

    return None


def _build_quantization_config(platform: str, precision: str, model_id: str = ""):
    """
    Build the right quantization config for the detected platform and
    requested precision.

    - auto               → detect pre-quantized checkpoints, skip re-quantization
    - blackwell + fp8    → TorchAoConfig fp8 (native FP8 tensor-core path)
    - blackwell + nvfp4  → FPQuantConfig nvfp4 (KNOWN BROKEN on GB10/SM120 — see README)
    - cuda/rocm + q8     → BitsAndBytesConfig 8-bit
    - cuda/rocm + q4     → BitsAndBytesConfig NF4
    - fp16 / bf16        → None (no quantization)
    """
    # --- Auto mode: detect pre-quantized checkpoints ---
    if precision == "auto":
        prequant = _is_prequantized(model_id)
        if prequant:
            print(f"  Auto-detected pre-quantized model ({prequant}) — skipping quantization config")
            return None
        # Not pre-quantized: pick the best available precision for this platform
        if platform == "blackwell":
            print("  Auto mode on Blackwell — using q4 (bitsandbytes NF4)")
            return _build_quantization_config(platform, "q4", model_id)
        elif platform in ("cuda", "rocm"):
            print("  Auto mode — using q4 (bitsandbytes NF4)")
            return _build_quantization_config(platform, "q4", model_id)
        else:
            print("  Auto mode on CPU — using fp16 (no quantization)")
            return None

    if precision == "fp8":
        if platform not in ("blackwell", "cuda"):
            raise RuntimeError(
                f"FP8 quantization requires an NVIDIA GPU with FP8 support (detected: {platform}). "
                "Use --precision fp16, q8, or q4 on this hardware."
            )
        # FP8 uses torchao's TorchAoConfig. Requires torchao matching the
        # PyTorch version — currently broken on PyTorch 2.11+cu130 (GB10).
        try:
            from torchao.quantization import float8_weight_only  # noqa: F401
            from transformers import TorchAoConfig
            return TorchAoConfig("float8_weight_only")
        except (ImportError, Exception) as e:
            raise RuntimeError(
                f"FP8 quantization is not available: {e}\n"
                "torchao is incompatible with this PyTorch version.\n"
                "Use --precision q4 (recommended) or --precision q8 instead."
            ) from e

    if precision == "nvfp4":
        if not supports_nvfp4(platform):
            raise RuntimeError(
                f"NVFP4 quantization requires a Blackwell GPU (detected: {platform}). "
                "Use --precision fp16, fp8, q8, or q4 on this hardware."
            )
        from transformers import FPQuantConfig
        return FPQuantConfig(
            forward_dtype="nvfp4",
            forward_method="abs_max",
            hadamard_group_size=16,
            modules_to_not_convert=["multi_modal_projector", "lm_head"],
        )

    if precision == "q8":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(load_in_8bit=True)

    if precision == "q4":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    return None  # fp16 — no quantization


class VoxtralTranscriber:
    def __init__(
        self,
        model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602",
        device: str = "auto",
        precision: str = "fp16",
        flash_attn: bool = False,
        compile_model: bool = False,
    ):
        # --- Detect platform ---
        if device == "auto":
            self.platform = detect_platform()
        else:
            # Map user-supplied device hint to a platform label
            self.platform = {
                "cuda": detect_platform() if detect_platform() == "blackwell" else "cuda",
                "rocm": "rocm",
                "cpu": "cpu",
            }.get(device, detect_platform())

        self.precision = precision
        self.compile_model = compile_model

        # Suggest FP8 on Blackwell when user asks for fp16
        if self.platform == "blackwell" and precision == "fp16":
            print(f"  Blackwell detected — consider using --precision fp8 for faster inference")

        opts = [f"precision={precision}", f"platform={self.platform}"]
        if flash_attn:
            opts.append("sdpa")
        if compile_model:
            opts.append("torch.compile")

        device_map = get_optimal_device_map(self.platform)
        print(f"Loading Transformers model: {model_id} ({', '.join(opts)}) device_map='{device_map}'...")

        # --- Load processor ---
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # --- Build quantization config ---
        quantization_config = _build_quantization_config(self.platform, precision, model_id)
        torch_dtype = get_torch_dtype(self.platform, precision)

        # --- Load model (suppress noisy HF warnings) ---
        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            load_kwargs = dict(
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config

            # SDPA — PyTorch's native Scaled Dot-Product Attention.
            # Dispatches to: CK flash-attn on ROCm, FA2 on CUDA/Blackwell.
            if flash_attn:
                load_kwargs["attn_implementation"] = "sdpa"

            # Use AutoModelForSpeechSeq2Seq so the correct architecture is
            # resolved from config.json — works for both VoxtralRealtime (4B)
            # and VoxtralForConditionalGeneration (24B / pre-quantized).
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                **load_kwargs,
            )

            # torch.compile — reduce Python overhead and fuse kernels.
            # "reduce-overhead" mode is best for generate() workloads.
            if compile_model:
                print("  Applying torch.compile (first inference will be slower)...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
        finally:
            hf_logging.set_verbosity(prev_verbosity)

        print(f"  Model ready on {platform_summary(self.platform)}")

    # ------------------------------------------------------------------
    # Single-segment transcription
    # ------------------------------------------------------------------
    def transcribe_segment(self, audio: np.ndarray, context: Optional[str] = None) -> str:
        """
        Transcribe a single audio segment.
        Optionally accepts context (last few words from previous segment)
        to maintain continuity across segment boundaries.
        """
        inputs = self._prepare_inputs(audio, context)

        with torch.inference_mode():
            max_new_tokens = self._estimate_max_tokens(audio)
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    # ------------------------------------------------------------------
    # Batched transcription (for independent segments)
    # ------------------------------------------------------------------
    def transcribe_batch(self, audio_segments: List[np.ndarray]) -> List[str]:
        """
        Transcribe multiple independent audio segments in a single forward pass.
        No context-carry — use this for segments from different speakers.
        """
        if not audio_segments:
            return []

        inputs = self.processor(
            audio=audio_segments,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device, dtype=self.model.dtype)

        # Use the longest segment to estimate max tokens for the whole batch
        longest = max(len(a) for a in audio_segments)
        max_new_tokens = max(64, int((longest / 16000) * 10))

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.strip() for t in transcriptions]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_inputs(self, audio: np.ndarray, context: Optional[str] = None):
        """Tokenise audio (+ optional text context) and move to device."""
        kwargs = dict(audio=audio, return_tensors="pt")
        if context:
            kwargs["text"] = " ".join(context.split()[-30:])

        return self.processor(**kwargs).to(self.model.device, dtype=self.model.dtype)

    @staticmethod
    def _estimate_max_tokens(audio: np.ndarray) -> int:
        """Estimate max output tokens from audio length (~10 tokens/sec)."""
        duration_s = len(audio) / 16000
        return max(64, int(duration_s * 10))
