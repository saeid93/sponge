from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer

# facebook/wav2vec2-base-960h:
# WER: 0.0338557516737675
# facebook/wav2vec2-large-960h:
# WER: 0.02765520389531345

librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")

model_names = ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-960h"]
results = {}
for model_name in model_names:
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to("cpu")
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    def map_to_pred(batch):
        input_values = processor(
            batch["audio"][0]["array"], return_tensors="pt", padding="longest"
        ).input_values
        with torch.no_grad():
            logits = model(input_values.to("cpu")).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        batch["transcription"] = transcription
        return batch

    results[model_name] = librispeech_eval.map(
        map_to_pred, batched=True, batch_size=1, remove_columns=["audio"]
    )

for model_name in model_names:
    print(f"{model_name}:")
    print(
        "WER:", wer(results[model_name]["text"], results[model_name]["transcription"])
    )
# from datasets import load_dataset
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# import soundfile as sf
# import torch
# from jiwer import wer


# librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")

# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cpu")
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# def map_to_pred(batch):
#     input_values = processor(batch["audio"][0]["array"], return_tensors="pt", padding="longest").input_values
#     with torch.no_grad():
#         logits = model(input_values.to("cpu")).logits

#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)
#     batch["transcription"] = transcription
#     return batch

# result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1) #, remove_columns=["speech"])

# print("WER:", wer(result["text"], result["transcription"]))
