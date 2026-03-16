import torch
from transformers import VoxtralRealtimeForConditionalGeneration, AutoProcessor
import numpy as np

class VoxtralTranscriber:
    def __init__(self, model_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602", device: str = "cuda"):
        self.device = device
        print(f"Loading Transformers model: {model_id} using device_map='auto'...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

    def transcribe_segment(self, audio: np.ndarray) -> str:
        """
        Transcription pass for Voxtral Realtime.
        """
        inputs = self.processor(audio=audio, return_tensors="pt").to(self.model.device, dtype=self.model.dtype)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
        
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    def contextual_diarization_pass(self, transcript: str) -> str:
        """
        Pass 2: Feed transcription back for speaker tagging.
        """
        # Simple text-to-text generation if supported by the model
        prompt = f"Insert speaker tags [SPEAKER_1], [SPEAKER_2], etc. into the following transcript based on conversational flow:\n\n{transcript}"
        inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
            
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
