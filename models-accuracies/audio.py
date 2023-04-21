from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer


librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")

model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


def map_to_pred(batch):
    input_values = processor(
        batch["audio"]["array"], return_tensors="pt", padding="longest"
    ).input_values
    with torch.no_grad():
        logits = model(input_values.to("cuda")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription
    return batch


result = librispeech_eval.map(
    map_to_pred, batched=True, batch_size=1, remove_columns=["audio"]
)

print("WER:", wer(result["text"], result["transcription"]))
