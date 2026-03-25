import torch
import logging
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from transformers.utils import logging as hf_logging
import numpy as np
from typing import Optional

# Suppress the "layers not sharded" accelerate warning via standard logging
logging.getLogger("accelerate.big_modeling").setLevel(logging.ERROR)

class VoxtralTranscriber:
    def __init__(self, model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602", device: str = "cuda",
                 precision: str = "fp16", flash_attn: bool = False, compile_model: bool = False):
        self.device = device
        self.precision = precision
        self.compile_model = compile_model

        opts = [f"precision={precision}"]
        if flash_attn:
            opts.append("sdpa")
        if compile_model:
            opts.append("torch.compile")
        print(f"Loading Transformers model: {model_id} ({', '.join(opts)}) using device_map='auto'...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Build quantization config based on precision
        quantization_config = None
        torch_dtype = "auto"

        if precision == "q8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "q4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Temporarily set HF verbosity to ERROR to suppress the expected
        # "You are using a model of type mixtral..." message during load.
        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            load_kwargs = dict(
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config

            # SDPA (Scaled Dot-Product Attention) — PyTorch's native implementation
            # that automatically dispatches to the best available backend:
            # - ROCm: Composable Kernel (CK) flash attention
            # - CUDA: FlashAttention-2 or memory-efficient attention
            # No external packages needed — it's built into PyTorch 2.x+.
            if flash_attn:
                load_kwargs["attn_implementation"] = "sdpa"

            self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
                model_id,
                **load_kwargs,
            )

            # torch.compile — reduces Python overhead and fuses kernels.
            # Uses "reduce-overhead" mode for best latency on generate() workloads.
            # ROCm uses the Triton backend by default which works well here.
            if compile_model:
                print("  Applying torch.compile (first inference will be slower due to compilation)...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
        finally:
            hf_logging.set_verbosity(prev_verbosity)

    def transcribe_segment(self, audio: np.ndarray, context: Optional[str] = None) -> str:
        """
        Transcription pass for Voxtral Realtime.
        Optionally accepts context (last few words from previous segment)
        to maintain continuity across segment boundaries.
        """
        # Build inputs with optional text context for continuity
        if context:
            # Prepend the last ~30 words as a continuation hint
            context_words = " ".join(context.split()[-30:])
            inputs = self.processor(
                audio=audio,
                text=context_words,
                return_tensors="pt"
            ).to(self.model.device, dtype=self.model.dtype)
        else:
            inputs = self.processor(
                audio=audio,
                return_tensors="pt"
            ).to(self.model.device, dtype=self.model.dtype)

        with torch.no_grad():
            # max_new_tokens avoids the model-agnostic max_length warning.
            # Audio at 16kHz: ~8 tokens/sec.
            audio_duration_s = len(audio) / 16000
            max_new_tokens = max(64, int(audio_duration_s * 10))
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
