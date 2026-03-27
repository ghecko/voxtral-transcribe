import torch
from datasets import Dataset, load_dataset
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

# Configuration
MODEL_ID = "mistralai/Voxtral-Small-24B-2507"
SAVE_DIR = "Voxtral-Small-24B-2507-W4A16-GPTQ"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQ_LENGTH = 2048

print(f"--- Initializing {MODEL_ID} ---")

# Load Model & Processor
model = VoxtralForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.bfloat16,
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer

# Load calibration data (stream to avoid downloading 300GB+)
print("Loading calibration samples...")
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

# Pre-tokenize so llmcompressor skips its tokenization step
# (Voxtral's MistralCommonTokenizer is incompatible with llmcompressor's tokenize args)
samples = []
for i, example in enumerate(ds):
    if i >= NUM_CALIBRATION_SAMPLES:
        break
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    samples.append(tokens)

calibration_ds = Dataset.from_dict({
    "input_ids": [s["input_ids"] for s in samples],
    "attention_mask": [s["attention_mask"] for s in samples],
})
print(f"Collected {len(calibration_ds)} calibration samples")

# W4A16 GPTQ Recipe (4-bit weights, 16-bit activations)
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["language_model.lm_head", "re:audio_tower.*", "re:multi_modal_projector.*"],
    dampening_frac=0.1,
)

# Patch pynvml crash on unsupported GPU memory queries
import llmcompressor.utils.metric_logging as _ml
_orig_get_gpu = _ml.CompressionLogger._get_GPU_usage_nv

@staticmethod
def _safe_get_gpu(visible_ids):
    try:
        return _orig_get_gpu(visible_ids)
    except Exception:
        return []

_ml.CompressionLogger._get_GPU_usage_nv = _safe_get_gpu

# Apply quantization
print("Starting W4A16 GPTQ quantization...")
oneshot(
    model=model,
    dataset=calibration_ds,
    recipe=recipe,
    processor=processor,
    max_seq_length=MAX_SEQ_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

print(f"Done! W4A16 GPTQ model saved to {SAVE_DIR}")
