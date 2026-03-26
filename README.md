# voxtral-transcribe

**voxtral-transcribe** is a high-performance audio transcription and diarization pipeline built on Mistral's **Voxtral** models. It integrates Pyannote for speaker identification (with built-in VAD) and Hugging Face **Transformers** for native execution on NVIDIA Blackwell (GB10), NVIDIA CUDA, AMD ROCm, and CPU.

> [!NOTE]
> This repository was vibe-coded to provide an immediate solution for Voxtral-based transcription needs. It prioritizes stability and speed of deployment.
> Tested primarily on AMD ROCm and NVIDIA Blackwell (Asus Ascent GX10) with `mistralai/Voxtral-Mini-4B-Realtime-2602`.

## Comparison with WhisperX

While inspired by WhisperX, this project has specific differences in its approach and precision:

- **Timestamp Precision**: Unlike WhisperX, which provides **word-level** timestamps via forced alignment (Wav2Vec2), **voxtral-transcribe** provides **segment-level (speaker-level)** timestamps. Timestamps mark the start of each speech turn or sentence.
- **Model Architecture**: Uses Mistral's Multimodal LLM (Voxtral) instead of the standard Whisper encoder-decoder. This allows for better contextual understanding and native speaker tagging.
- **Diarization**: Uses Pyannote 3.1+ with built-in VAD segmentation and speaker attribution in a single pass.

## Key Features

- **Multi-Backend Support**: Specialized Docker environments for Blackwell (GB10), CUDA, ROCm 7.2, and CPU.
- **Auto Platform Detection**: Automatically detects Blackwell, CUDA, ROCm, or CPU at runtime and selects optimal device mapping and dtype.
- **NVFP4 Quantization**: Native FP4 quantization on Blackwell GPUs for ~4x throughput improvement over FP16.
- **Model Pre-Download**: Bake model weights into Docker images at build time for instant cold starts.
- **Configurable Model**: Set `MODEL_ID` as an environment variable or build argument to use any compatible model.
- **Format Agnostic**: Support for `.wav`, `.mp3`, `.m4a`, and more via FFmpeg-based loading.
- **Unified VAD + Diarization**: Pyannote's pipeline handles both voice activity detection and speaker attribution in one pass — no reconciliation needed.
- **Context-Carry**: Maintains transcription context across segments for natural sentence continuity.
- **Speaker Count Hints**: Pass `--num-speakers`, `--min-speakers`, or `--max-speakers` to improve diarization accuracy.
- **Performance Options**: SDPA flash attention (`--flash-attn`) and `torch.compile` (`--compile`) for additional speed gains.
- **Report Generation**: Automatic export to JSON, Markdown, TXT, and SRT (subtitles).

## Architecture

1. **Platform Detection**: `core/platform.py` auto-detects Blackwell (compute capability 10.x), CUDA, ROCm, or CPU and selects the optimal device map, dtype, and quantization strategy.
2. **Audio Preprocessing**: FFmpeg loads any format and converts to 16kHz mono float32.
3. **Model Loading**: Direct GPU placement on Blackwell/CUDA (`cuda:0`) skips Accelerate's profiling pass. ROCm uses `device_map='auto'` for HIP compatibility.
4. **Diarization + VAD**: Pyannote `speaker-diarization-community-1` performs voice activity detection and speaker attribution in a single pass.
5. **Inference**: Native Transformers backend with `VoxtralRealtimeForConditionalGeneration`, `torch.inference_mode()`, and cross-segment context-carry for maximum accuracy.

## Supported Platforms

| Platform | Dockerfile | Base Image | Quantization | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Blackwell (GB10)** | `Dockerfile.spark` | `nvcr.io/nvidia/pytorch:26.03-py3` | **fp8**, q8, q4, fp16 | Asus Ascent GX10, DGX Spark |
| **Blackwell (B200)** | `Dockerfile.spark` | `nvcr.io/nvidia/pytorch:26.03-py3` | **nvfp4**, fp8, q8, q4, fp16 | B200, B100 (SM100) |
| **CUDA** | `Dockerfile.cuda` | `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime` | q8, q4, fp16 | Pre-Blackwell NVIDIA GPUs |
| **ROCm** | `Dockerfile.rocm` | `rocm/pytorch:rocm7.2_ubuntu24.04_py3.13_pytorch_release_2.10.0` | q8, q4, fp16 | AMD GPUs (MI300X, RX 7900, etc.) |
| **CPU** | `Dockerfile.cpu` | — | fp16 | Fallback, significantly slower |

## Getting Started (Docker)

### 1. Build the Target Image

Build only the image that matches your hardware:

```bash
# Blackwell (Asus Ascent GX10, DGX Spark)
docker compose build voxtral-transcribe-spark

# AMD ROCm
docker compose build voxtral-transcribe-rocm

# NVIDIA CUDA (pre-Blackwell)
docker compose build voxtral-transcribe-cuda

# CPU
docker compose build voxtral-transcribe-cpu
```

By default, the model weights are downloaded during the build (baked into the image). To skip this:

```bash
PREDOWNLOAD_MODEL=0 docker compose build voxtral-transcribe-spark
```

To use a different model:

```bash
MODEL_ID=your-org/custom-voxtral docker compose build voxtral-transcribe-spark
```

### 2. Run Processing

#### Prepare your data

Place audio files in the `./audio` directory (create it if needed). These are accessible inside the container at `/data/`.

#### Option A: Docker Compose (Recommended)

```bash
# Set your HF token
export HF_TOKEN="your_token_here"

# Basic transcription
docker compose run --rm voxtral-transcribe-spark /data/your_audio.mp3

# With speaker count hint
docker compose run --rm voxtral-transcribe-spark /data/interview.mp3 --num-speakers 2

# With FP8 quantization + flash attention + torch.compile (Blackwell / GB10)
docker compose run --rm voxtral-transcribe-spark /data/meeting.mp3 \
    --precision fp8 --flash-attn --compile

# ROCm with 8-bit quantization
docker compose run --rm voxtral-transcribe-rocm /data/call.mp3 \
    --precision q8 --flash-attn
```

> [!TIP]
> Anything added after the service name overrides the default command and is passed directly to the application.

#### Option B: Manual Docker Run

```bash
# Blackwell (GX10)
docker run --rm --gpus all \
    -v ./audio:/data \
    -v ./outputs:/outputs \
    -e HF_TOKEN="your_token" \
    voxtral-transcribe:spark \
    /data/my_audio.mp3 --output-dir /outputs --precision fp8 --flash-attn

# ROCm
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    -v ./audio:/data \
    -v ./outputs:/outputs \
    -e HF_TOKEN="your_token" \
    voxtral-transcribe:rocm \
    /data/my_audio.mp3 --output-dir /outputs --num-speakers 2
```

## Quantization

### Recommended: bitsandbytes q4/q8 (Works Now)

For immediate use on the GB10, bitsandbytes NF4 or INT8 quantization works out of the box and provides good speedups. No pre-quantization step needed — just pass a flag:

```bash
# 4-bit NF4 quantization (smallest memory footprint, ~2 GB)
docker compose run --rm voxtral-transcribe-spark /data/audio.mp3 \
    --precision q4 --flash-attn --compile

# 8-bit quantization (better quality, ~4 GB)
docker compose run --rm voxtral-transcribe-spark /data/audio.mp3 \
    --precision q8 --flash-attn --compile
```

### FP8 Quantization (Experimental on GB10)

FP8 (E4M3) uses torchao's `float8_weight_only` via `TorchAoConfig`. It reduces model size from ~8 GB (FP16) to ~4 GB with minimal quality loss and uses native FP8 tensor cores on Blackwell.

> [!WARNING]
> torchao must match your PyTorch version exactly. On GB10 / DGX Spark with PyTorch 2.11+cu130,
> the pip `torchao` package may not have compatible aarch64 wheels yet. Install from the
> PyTorch index or run inside the Docker container where dependencies are matched:
> ```bash
> pip install torchao --index-url https://download.pytorch.org/whl/cu130
> ```

#### Quantize and Use

```bash
# Quantize the model (run on your GB10 or inside Docker)
python scripts/quantize_fp8.py --output-dir ./models/Voxtral-Mini-4B-FP8

# Use the quantized model
docker compose run --rm \
    -e MODEL_ID=./models/Voxtral-Mini-4B-FP8 \
    voxtral-transcribe-spark /data/audio.mp3 --precision fp8

# Or bake it into the Docker image
MODEL_ID=./models/Voxtral-Mini-4B-FP8 docker compose build voxtral-transcribe-spark
```

### NVFP4 Quantization (B200/B100 Only)

> [!WARNING]
> **NVFP4 is currently broken on the GB10 (SM120/SM121).** The CUTLASS FP4 GEMM kernels
> require 128-256 KiB of shared memory, but the GB10 only provides 99 KiB. This causes
> `Error Internal` crashes in `fusedQuantizeNvAbsMax`. Use FP8 instead on GB10 hardware.
>
> NVFP4 works on B200/B100 GPUs (SM100) which have sufficient shared memory.
>
> Tracking issues: [CUTLASS #2800](https://github.com/NVIDIA/cutlass/issues/2800), [CUTLASS #3096](https://github.com/NVIDIA/cutlass/issues/3096)

NVFP4 quantizes from FP16 (~8 GB) down to ~2 GB and provides the highest throughput on SM100 Blackwell GPUs.

#### Prerequisites (SM100 only)

```bash
sudo apt install python3-dev   # Python.h headers for compiling qutlass
pip install fp_quant
git clone https://github.com/IST-DASLab/qutlass.git
cd qutlass && pip install --no-build-isolation .
```

#### Quantize and Use (SM100 only)

```bash
python scripts/quantize_nvfp4.py --output-dir ./models/Voxtral-Mini-4B-NVFP4

docker compose run --rm \
    -e MODEL_ID=./models/Voxtral-Mini-4B-NVFP4 \
    voxtral-transcribe-spark /data/audio.mp3 --precision nvfp4
```

## CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `input` | Path to the input audio file | Required |
| `--output-dir` | Directory to save JSON/MD/TXT/SRT reports | `outputs` |
| `--model` | Voxtral model ID or local path | `$MODEL_ID` or `mistralai/Voxtral-Mini-4B-Realtime-2602` |
| `--device` | Compute backend (`auto`, `cuda`, `rocm`, `cpu`) | `auto` (detects platform) |
| `--precision` | Model precision (`fp16`, `fp8`, `nvfp4`, `q8`, `q4`) | `fp16` |
| `--flash-attn` | Enable SDPA flash attention (CK on ROCm, FA2 on CUDA/Blackwell) | Off |
| `--compile` | Apply `torch.compile` for faster inference (first run slower) | Off |
| `--hf-token` | HF token for Pyannote models | `$HF_TOKEN` env var |
| `--num-speakers` | Exact number of speakers (improves accuracy) | Auto-detect |
| `--min-speakers` | Minimum number of speakers | Auto-detect |
| `--max-speakers` | Maximum number of speakers | Auto-detect |

## Environment Variables

| Variable | Description | Used In |
| :--- | :--- | :--- |
| `MODEL_ID` | Model ID or local path to load | Dockerfile (build-arg) + runtime |
| `PREDOWNLOAD_MODEL` | Set to `0` to skip baking the model into the Docker image | Dockerfile (build-arg) |
| `HF_TOKEN` | Hugging Face token for gated models (Pyannote) | Runtime |
| `HF_HOME` | Cache directory for Hugging Face models | Dockerfile default: `/app/models` |

## Project Structure

```
voxtral-transcribe/
├── main.py                          # Entry point
├── core/
│   ├── platform.py                  # Platform detection (Blackwell/CUDA/ROCm/CPU)
│   ├── transcribe.py                # Model loading and inference
│   ├── audio.py                     # FFmpeg audio loading
│   ├── diarize.py                   # Pyannote diarization
│   └── format.py                    # Output formatting (JSON/MD/TXT/SRT)
├── scripts/
│   ├── download_model.py            # Pre-download model for Docker build
│   ├── quantize_fp8.py              # FP8 quantization (recommended for GB10)
│   └── quantize_nvfp4.py            # NVFP4 quantization (B200/B100 only)
├── Dockerfile.spark                 # Blackwell (GB10) — nvcr.io NGC container
├── Dockerfile.cuda                  # CUDA (pre-Blackwell)
├── Dockerfile.rocm                  # AMD ROCm 7.2
├── Dockerfile.cpu                   # CPU fallback
├── docker-compose.yaml              # Orchestration for all platforms
├── requirements.txt                 # Common Python dependencies
├── requirements-cuda.txt            # CUDA/Blackwell extras (includes bitsandbytes)
└── requirements-rocm.txt            # ROCm extras (includes bitsandbytes)
```

## License

MIT
