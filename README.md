# voxtral-transcribe

**voxtral-transcribe** is a high-performance audio transcription and diarization pipeline optimized for Mistral's **Voxtral** models. It integrates Pyannote for speaker identification (with built-in VAD) and Hugging Face **Transformers** for native execution on AMD ROCm, NVIDIA CUDA, and CPU.

> [!NOTE]
> This repository was vibe-coded to provide an immediate solution for Voxtral-based transcription needs. It prioritizes stability and speed of deployment.
> Not tested for cuda and cpu based hardware, and tested only with `mistralai/Voxtral-Mini-4B-Realtime-2602` model.

## Comparison with WhisperX

While inspired by WhisperX, this project has specific differences in its approach and precision:

- **Timestamp Precision**: Unlike WhisperX, which provides **word-level** timestamps via forced alignment (Wav2Vec2), **voxtral-transcribe** provides **segment-level (speaker-level)** timestamps. Timestamps mark the start of each speech turn or sentence.
- **Model Architecture**: Uses Mistral's Multimodal LLM (Voxtral) instead of the standard Whisper encoder-decoder. This allows for better contextual understanding and native speaker tagging.
- **Diarization**: Uses Pyannote 3.1+ with built-in VAD segmentation and speaker attribution in a single pass.

## Key Features

- **Multi-Backend Support**: Specialized Docker environments for ROCm 7.2, CUDA 13, and CPU.
- **Format Agnostic**: Support for `.wav`, `.mp3`, `.m4a`, and more via FFmpeg-based loading.
- **Unified VAD + Diarization**: Pyannote's pipeline handles both voice activity detection and speaker attribution in one pass — no reconciliation needed.
- **Context-Carry**: Maintains transcription context across segments for natural sentence continuity.
- **Speaker Count Hints**: Pass `--num-speakers`, `--min-speakers`, or `--max-speakers` to improve diarization accuracy.
- **Report Generation**: Automatic export to JSON, Markdown, TXT, and SRT (subtitles).

## Architecture

1.  **Audio Preprocessing**: FFmpeg loads any format and converts to 16kHz mono float32.
2.  **Diarization + VAD**: Pyannote `speaker-diarization-community-1` performs voice activity detection and speaker attribution in a single pass.
3.  **Inference**: Native Transformers backend with `VoxtralRealtimeForConditionalGeneration` and cross-segment context-carry for maximum accuracy.

## Getting Started (Docker Implementation)

To ensure no modifications to your host system, Voxtral-transcribe runs entirely within Docker.

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

#### Prepare your data
Place the audio files you want to transcribe in the `./audio` directory (create it if it doesn't exist). These files will be accessible inside the container at the path `/data/`.

#### Option A: Docker Compose (Recommended)
This handles the complex device mappings and volumes automatically.

```bash
# Set your HF token
export HF_TOKEN="your_token_here"

# Run processing on a specific file
# Note: The file path must start with /data/ (mapping to your ./audio folder)
docker compose run --rm voxtral-transcribe-rocm /data/your_audio.mp3

# With speaker count hint for better accuracy
docker compose run --rm voxtral-transcribe-rocm /data/interview.mp3 --num-speakers 2
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
    /data/my_audio.mp3 --output-dir /outputs --num-speakers 2
```

## CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `input` | Path to the input audio file | Required |
| `--output-dir` | Directory to save JSON/MD/TXT/SRT reports | `outputs` |
| `--model` | Voxtral model ID | `mistralai/Voxtral-Mini-4B-Realtime-2602` |
| `--hf-token` | HF token for Pyannote models | Environment Variable |
| `--device` | Backend device (`cuda`, `rocm`, `cpu`) | `rocm` |
| `--num-speakers` | Exact number of speakers (improves accuracy) | Auto-detect |
| `--min-speakers` | Minimum number of speakers | Auto-detect |
| `--max-speakers` | Maximum number of speakers | Auto-detect |

## License

MIT