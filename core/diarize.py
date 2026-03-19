import os
from pyannote.audio import Pipeline
import torch
import numpy as np
import warnings
from typing import List, Dict

# Suppress pyannote's verbose torchcodec/ffmpeg loading warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

class DiarizationAnalyzer:
    def __init__(self, model_id: str = "pyannote/speaker-diarization-community-1", auth_token: str = None):
        self.pipeline = Pipeline.from_pretrained(
            model_id,
            token=auth_token or os.getenv("HF_TOKEN")
        )
        if torch.cuda.is_available():
            # For ROCm, torch.cuda.is_available() is True, and 'cuda' refers to the ROCm device
            self.pipeline.to(torch.device("cuda"))
        elif torch.backends.mps.is_available():
            self.pipeline.to(torch.device("mps"))

    def diarize(self, audio: np.ndarray, sampling_rate: int = 16000) -> List[Dict]:
        """
        Diarize preloaded audio and return speaker segments.
        """
        # Convert numpy to torch tensor and add channel dimension
        # Copy to ensure the array is writable before converting to a torch tensor
        waveform = torch.from_numpy(audio.copy()).unsqueeze(0)
        
        # Pyannote expects a dictionary for in-memory audio
        input_data = {"waveform": waveform, "sample_rate": sampling_rate}
        
        output = self.pipeline(input_data)
        
        # Handle both old (Annotation) and new (DiarizeOutput) pyannote versions
        if hasattr(output, "speaker_diarization"):
            diarization = output.speaker_diarization
        else:
            diarization = output
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments
