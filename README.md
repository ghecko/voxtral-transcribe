# voxtral-transcribe

**voxtral-transcribe** is a high-performance audio transcription and diarization pipeline optimized for Mistral's **Voxtral** models. It integrates Silero VAD for segmenting, Pyannote for speaker identification, and Hugging Face **Transformers** for native execution on AMD ROCm, NVIDIA CUDA, and CPU.

> [!NOTE]
> This repository was vide-coded to provide an immediate solution for Voxtral-based transcription needs. It prioritizes stability and speed of deployment.
> Not tested for cuda and cpu based hardware, and tested only with `mistralai/Voxtral-Mini-4B-Realtime-2602` model.

## Comparison with WhisperX

While inspired by WhisperX, this project has specific differences in its approach and precision:

- **Timestamp Precision**: Unlike WhisperX, which provides **word-level** timestamps via forced alignment (Wav2Vec2), **voxtral-transcribe** provides **segment-level (speaker-level)** timestamps. Timestamps mark the start of each speech turn or sentence.
- **Model Architecture**: Uses Mistral's Multimodal LLM (Voxtral) instead of the standard Whisper encoder-decoder. This allows for better contextual understanding and native speaker tagging.
- **Diarization**: Uses Pyannote 3.1+, but reconciles results at the segment level rather than word-level alignment.

## Key Features

- **Multi-Backend Support**: Specialized Docker environments for ROCm 7.2, CUDA 13, and CPU.
- **Format Agnostic**: Support for `.wav`, `.mp3`, `.m4a`, and more via FFmpeg-based loading.
- **VAD Segmentation**: Uses Silero VAD v5 with timestamps in seconds for precision.
- **Enterprise Diarization**: High-accuracy speaker tags (**speaker-level timestamps**) using Pyannote (community-1) or Voxtral's native contextual awareness.
- **Report Generation**: Automatic export to JSON, Markdown, and TXT.

## Architecture

1.  **Audio Preprocessing**: FFmpeg loads any format and converts to 16kHz mono float32.
2.  **VAD**: Silero VAD v5 extracts speech segments with precise second-based timestamps.
3.  **Diarization**: 
    - **Mode 1 (Standard)**: Pyannote extracts speaker turns.
    - **Mode 2 (Native Hack)**: Voxtral re-processes the transcript to insert tags based on context.
4.  **Inference**: Native Transformers backend with `VoxtralRealtimeForConditionalGeneration` for maximum stability and ROCm compatibility.

## Getting Started (Docker Implementation)

To ensure no modifications to your host system, VoxtralX runs entirely within Docker.

### 1. Build the Target Image

You don't need to build all images. Build only the one that matches your hardware:

```bash
# To build ONLY for AMD ROCm
docker compose build voxtral-transcribe-rocm

# To build ONLY for NVIDIA CUDA
docker compose build voxtral-transcribe-cuda

# To build ONLY for CPU
docker compose build voxtral-transcribe-cpu
```

### 2. Run Processing

#### Option A: Docker Compose (Recommended)
This handles the complex device mappings and volumes automatically.

```bash
# Set your HF token
export HF_TOKEN="your_token_here"

# Run with the default file (audio.mp3)
docker compose run --rm voxtral-transcribe-rocm

# Pass a CUSTOM audio file and arguments
docker compose run --rm voxtral-transcribe-rocm /data/your_audio.wav --output-dir /outputs --native-diarize
```

> [!TIP]
> Anything added after the service name (`voxtral-transcribe-rocm`) overrides the default command in the `docker-compose.yaml` and is passed directly to the application.

#### Option B: Manual Docker Run
If you prefer manual execution:
```bash
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    -v /path/to/audio:/data \
    -v /path/to/outputs:/outputs \
    -e HF_TOKEN="your_huggingface_token" \
    voxtral-transcribe:rocm \
    /data/my_audio.mp3 --output-dir /outputs
```

## CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `input` | Path to the input audio file | Required |
| `--output-dir` | Directory to save JSON/MD/TXT reports | `outputs` |
| `--model` | Voxtral model ID | `mistralai/Voxtral-Mini-4B-Realtime-2602` |
| `--hf-token` | HF token for Pyannote models | Environment Variable |
| `--device` | Backend device (`cuda`, `rocm`, `cpu`) | `cuda` |
| `--native-diarize` | Use Voxtral's contextual diarization hack | `False` |

## License

MIT
