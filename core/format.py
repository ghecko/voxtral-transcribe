import json
from typing import List, Dict

class OutputFormatter:
    @staticmethod
    def to_json(data: List[Dict], output_path: str):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def to_markdown(data: List[Dict], output_path: str):
        lines = ["# Transcription Report\n"]
        for entry in data:
            speaker = entry.get("speaker", "Unknown")
            start = entry.get("start", 0)
            text = entry.get("text", "")
            lines.append(f"**[{speaker}] ({start:.2f}s):** {text}\n")
        
        with open(output_path, "w") as f:
            f.writelines(lines)

    @staticmethod
    def to_txt(data: List[Dict], output_path: str):
        lines = []
        for entry in data:
            speaker = entry.get("speaker", "Unknown")
            text = entry.get("text", "")
            lines.append(f"{speaker}: {text}\n")
            
        with open(output_path, "w") as f:
            f.writelines(lines)
