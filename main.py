import argparse
import sys
import os
import warnings

# Suppress pyannote's verbose warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
from rich.console import Console
from core.audio import load_audio
from core.vad import VADAnalyzer
# NOW we can safely import and initialize GPU-based libraries
from core.diarize import DiarizationAnalyzer
from core.format import OutputFormatter
from core.transcribe import VoxtralTranscriber

console = Console()

def main():
    parser = argparse.ArgumentParser(description="VoxtralX: Audio Processing with Voxtral")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    parser.add_argument("--model", default="mistralai/Voxtral-Mini-4B-Realtime-2602", help="Voxtral model ID")
    parser.add_argument("--device", choices=["cuda", "rocm", "cpu"], default="rocm", help="Hardware device")
    parser.add_argument("--hf-token", help="Hugging Face token for Pyannote")
    parser.add_argument("--native-diarize", action="store_true", help="Use Voxtral native diarization hack instead of Pyannote")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        console.print(f"[bold red]Error:[/bold red] Input file {args.input} not found.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    console.print(f"[bold blue]Loading audio:[/bold blue] {args.input}")
    audio = load_audio(args.input)
    
    console.print("[bold green]Running VAD...[/bold green]")
    vad = VADAnalyzer(device="cpu") # Silero VAD is fast on CPU
    speech_chunks = vad.get_speech_chunks(audio)
    
    transcriber = VoxtralTranscriber(model_id=args.model, device=args.device)
    
    final_data = []
    sampling_rate = 16000
    
    console.print("[bold cyan]Running Pyannote Diarization...[/bold cyan]")
    diarizer = DiarizationAnalyzer(auth_token=args.hf_token)
    speaker_segments = diarizer.diarize(audio)
    
    console.print(f"[bold green]Processing {len(speech_chunks)} speech segments with speaker reconciliation...[/bold green]")
    
    segment_count = 0
    for chunk in speech_chunks:
        chunk_start = chunk["start"]
        chunk_end = chunk["end"]

        # Find all Pyannote speaker segments that overlap this VAD chunk
        overlapping = []
        for seg in speaker_segments:
            overlap_start = max(chunk_start, seg["start"])
            overlap_end = min(chunk_end, seg["end"])
            if overlap_end > overlap_start:
                overlapping.append({
                    "start": overlap_start,
                    "end": overlap_end,
                    "speaker": seg["speaker"]
                })

        # If no diarization info, treat the whole chunk as one unknown speaker
        if not overlapping:
            overlapping = [{"start": chunk_start, "end": chunk_end, "speaker": "Unknown"}]

        # Sort by start time and transcribe each sub-segment individually
        # Clamp boundaries so no two sub-segments share audio (prevents duplicates)
        overlapping.sort(key=lambda x: x["start"])
        prev_end = chunk_start
        for sub in overlapping:
            sub["start"] = max(sub["start"], prev_end)
            prev_end = sub["end"]
        overlapping = [s for s in overlapping if s["end"] > s["start"]]
        segment_count += len(overlapping)

        for sub in overlapping:
            start_samp = int(sub["start"] * sampling_rate)
            end_samp = int(sub["end"] * sampling_rate)
            # Guard against sub-segments that are too short to be useful (<0.3s)
            if (end_samp - start_samp) < int(0.3 * sampling_rate):
                continue
            segment_audio = audio[start_samp:end_samp]
            speaker = sub["speaker"]

            console.print(f"  Transcribing sub-segment ({sub['start']:.2f}s - {sub['end']:.2f}s) -> [bold]{speaker}[/bold]")
            text = transcriber.transcribe_segment(segment_audio)

            # Skip empty or whitespace-only segments
            if not text or not text.strip():
                continue

            # Merge with previous segment if speaker is the same
            if final_data and final_data[-1]["speaker"] == speaker:
                final_data[-1]["end"] = round(sub["end"], 3)
                final_data[-1]["text"] += " " + text
            else:
                final_data.append({
                    "start": round(sub["start"], 3),
                    "end": round(sub["end"], 3),
                    "speaker": speaker,
                    "text": text
                })


    # Save outputs
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    OutputFormatter.to_json(final_data, os.path.join(args.output_dir, f"{base_name}.json"))
    OutputFormatter.to_markdown(final_data, os.path.join(args.output_dir, f"{base_name}.md"))
    OutputFormatter.to_txt(final_data, os.path.join(args.output_dir, f"{base_name}.txt"))
    
    console.print(f"[bold green]Done! Outputs saved in {args.output_dir}[/bold green]")

if __name__ == "__main__":
    main()
