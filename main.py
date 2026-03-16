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
    
    for i, chunk in enumerate(speech_chunks):
        start_samp = int(chunk["start"] * sampling_rate)
        end_samp = int(chunk["end"] * sampling_rate)
        segment_audio = audio[start_samp:end_samp]
        
        # Determine speaker by finding overlapping diarization segment
        current_speaker = "Unknown"
        max_overlap = 0
        for seg in speaker_segments:
            overlap = min(chunk["end"], seg["end"]) - max(chunk["start"], seg["start"])
            if overlap > max_overlap:
                max_overlap = overlap
                current_speaker = seg["speaker"]
        
        console.print(f"  Trancribing segment {i+1}/{len(speech_chunks)} ({chunk['start']:.2f}s - {chunk['end']:.2f}s) -> [bold]{current_speaker}[/bold]")
        text = transcriber.transcribe_segment(segment_audio)
        
        # Skip empty or whitespace-only segments
        if not text or not text.strip():
            continue

        # Merge with previous segment if speaker is the same
        if final_data and final_data[-1]["speaker"] == current_speaker:
            final_data[-1]["end"] = round(chunk["end"], 3)
            final_data[-1]["text"] += " " + text
        else:
            final_data.append({
                "start": round(chunk["start"], 3),
                "end": round(chunk["end"], 3),
                "speaker": current_speaker,
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
