import torch
from silero_vad import load_silero_vad, get_speech_timestamps
import numpy as np
from typing import List, Dict

class VADAnalyzer:
    def __init__(self, device: str = "cpu"):
        self.model = load_silero_vad()
        self.device = torch.device(device)
        self.model.to(self.device)

    def get_speech_chunks(self, audio: np.ndarray, sampling_rate: int = 16000) -> List[Dict]:
        """
        Detect speech segments in the audio and return timestamps.
        """
        # Ensure audio is float32 for silero-vad
        if audio.dtype != np.float32:
            audio_float32 = audio.astype(np.float32)
        else:
            audio_float32 = audio

        # Convert audio to torch tensor
        audio_tensor = torch.from_numpy(audio_float32.copy()).to(self.device)
        
        # Get speech timestamps with tuned parameters to reduce noise
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=0.6,               # Slightly more selective
            min_speech_duration_ms=400,  # Ignore segments shorter than 400ms
            min_silence_duration_ms=500, # Group words together more
            sampling_rate=sampling_rate,
            return_seconds=True
        )
        
        return speech_timestamps
