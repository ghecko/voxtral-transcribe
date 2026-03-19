import torch
import logging
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
from transformers.utils import logging as hf_logging
import numpy as np

# Suppress the "layers not sharded" accelerate warning via standard logging
logging.getLogger("accelerate.big_modeling").setLevel(logging.ERROR)

class VoxtralTranscriber:
    def __init__(self, model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602", device: str = "cuda"):
        self.device = device
        print(f"Loading Transformers model: {model_id} using device_map='auto'...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Temporarily set HF verbosity to ERROR to suppress the expected
        # "You are using a model of type mixtral..." message during load.
        # This uses HF's own API, which is the only reliable way to suppress
        # messages emitted through their internal logging subsystem.
        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
        finally:
            hf_logging.set_verbosity(prev_verbosity)

    def transcribe_segment(self, audio: np.ndarray) -> str:
        """
        Transcription pass for Voxtral Realtime.
        """
        inputs = self.processor(audio=audio, return_tensors="pt").to(self.model.device, dtype=self.model.dtype)
        
        with torch.no_grad():
            # max_new_tokens avoids the model-agnostic max_length warning.
            # Audio at 16kHz: ~8 tokens/sec. Cap at 448 tokens (~56s of speech).
            audio_duration_s = len(audio) / 16000
            max_new_tokens = max(64, int(audio_duration_s * 10))
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    def contextual_diarization_pass(self, transcript: str) -> str:
        """
        Pass 2: Feed transcription back for speaker tagging.
        """
        prompt = f"Insert speaker tags [SPEAKER_1], [SPEAKER_2], etc. into the following transcript based on conversational flow:\n\n{transcript}"
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
            
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
