import argparse
import sys
import os
import time
import warnings

# Suppress pyannote's verbose warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
from rich.console import Console
from core.audio import load_audio
from core.diarize import DiarizationAnalyzer
from core.format import OutputFormatter
from core.transcribe import VoxtralTranscriber
from core.platform import detect_platform, platform_summary

console = Console()


def main():
    parser = argparse.ArgumentParser(description="voxtral-transcribe: Audio Processing with Voxtral")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_ID", "mistralai/Voxtral-Mini-4B-Realtime-2602"),
        help="Voxtral model ID or local path (default: $MODEL_ID or mistralai/Voxtral-Mini-4B-Realtime-2602)",
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "rocm", "cpu"], default="auto",
        help="Hardware device (default: auto-detect)",
    )
    parser.add_argument("--hf-token", help="Hugging Face token for Pyannote")
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers (improves accuracy)")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum number of speakers")
    parser.add_argument(
        "--precision", choices=["fp16", "fp8", "nvfp4", "q8", "q4"], default="fp16",
        help="Model precision: fp16 (default), fp8 (Blackwell/Ada+), nvfp4 (Blackwell SM100 only), q8, q4",
    )
    parser.add_argument(
        "--flash-attn", action="store_true",
        help="Enable SDPA flash attention (CK on ROCm, FA2 on CUDA/Blackwell)",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Apply torch.compile for faster inference (first run is slower)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        console.print(f"[bold red]Error:[/bold red] Input file {args.input} not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Platform info ---
    platform = detect_platform() if args.device == "auto" else args.device
    console.print(f"[bold]Platform:[/bold] {platform_summary(platform)}")

    start_time = time.time()

    # --- Stage 1: Load audio ---
    console.print(f"[bold blue]Loading audio:[/bold blue] {args.input}")
    audio = load_audio(args.input)
    audio_duration = len(audio) / 16000
    console.print(f"  Audio duration: {audio_duration:.1f}s")

    # --- Stage 2: Load model ---
    transcriber = VoxtralTranscriber(
        model_id=args.model, device=args.device, precision=args.precision,
        flash_attn=args.flash_attn, compile_model=args.compile,
    )

    # --- Stage 3: Diarization (includes built-in VAD) ---
    console.print("[bold cyan]Running Pyannote Diarization (includes VAD)...[/bold cyan]")
    diarizer = DiarizationAnalyzer(auth_token=args.hf_token)
    speaker_segments = diarizer.diarize(
        audio,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    # Detect unique speakers
    unique_speakers = set(s["speaker"] for s in speaker_segments)
    console.print(f"  Detected {len(unique_speakers)} speaker(s): {', '.join(sorted(unique_speakers))}")
    console.print(f"  {len(speaker_segments)} speech segments found")

    # Note: Pyannote segments may overlap during simultaneous speech.
    # We intentionally do NOT clamp overlaps — losing content is worse
    # than minor duplication, and the model handles mixed audio reasonably.

    # --- Stage 4: Transcribe each segment ---
    console.print(f"[bold green]Transcribing {len(speaker_segments)} segments...[/bold green]")

    final_data = []
    sampling_rate = 16000
    last_context = None  # For context-carry between same-speaker segments
    skipped = 0

    for i, seg in enumerate(speaker_segments):
        start_samp = int(seg["start"] * sampling_rate)
        end_samp = int(seg["end"] * sampling_rate)
        duration = seg["end"] - seg["start"]

        # Skip sub-segments shorter than 0.5s — too short for reliable transcription
        if duration < 0.5:
            skipped += 1
            continue

        segment_audio = audio[start_samp:end_samp]
        speaker = seg["speaker"]

        # Pass context from previous same-speaker segment for continuity
        context = last_context if (final_data and final_data[-1]["speaker"] == speaker) else None

        console.print(f"  [{i+1}/{len(speaker_segments)}] ({seg['start']:.2f}s - {seg['end']:.2f}s) -> [bold]{speaker}[/bold]")
        text = transcriber.transcribe_segment(segment_audio, context=context)

        # Skip empty transcriptions
        if not text or not text.strip():
            skipped += 1
            continue

        # Track context for next segment
        last_context = text

        # Merge with previous segment if speaker is the same
        if final_data and final_data[-1]["speaker"] == speaker:
            final_data[-1]["end"] = round(seg["end"], 3)
            final_data[-1]["text"] += " " + text
        else:
            final_data.append({
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "speaker": speaker,
                "text": text
            })

    # --- Stage 5: Save outputs ---
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    OutputFormatter.to_json(final_data, os.path.join(args.output_dir, f"{base_name}.json"))
    OutputFormatter.to_markdown(final_data, os.path.join(args.output_dir, f"{base_name}.md"))
    OutputFormatter.to_txt(final_data, os.path.join(args.output_dir, f"{base_name}.txt"))
    OutputFormatter.to_srt(final_data, os.path.join(args.output_dir, f"{base_name}.srt"))

    # --- Stats summary ---
    elapsed = time.time() - start_time
    console.print()
    console.print("[bold green]━━━ Summary ━━━[/bold green]")
    console.print(f"  Audio duration : {audio_duration:.1f}s")
    console.print(f"  Speakers       : {len(unique_speakers)}")
    console.print(f"  Segments       : {len(final_data)} (skipped {skipped})")
    console.print(f"  Processing time: {elapsed:.1f}s ({audio_duration/elapsed:.1f}x realtime)")
    console.print(f"  Outputs        : {args.output_dir}/{base_name}.{{json,md,txt,srt}}")
    console.print()

if __name__ == "__main__":
    main()
