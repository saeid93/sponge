from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from jiwer import wer

# facebook/wav2vec2-base-960h:
# WER: 0.0338557516737675
# facebook/wav2vec2-large-960h:
# WER: 0.02765520389531345
# facebook/s2t-small-librispeech-asr:
# WER: 0.0412863228358436
# facebook/s2t-medium-librispeech-asr:
# WER: 0.03512831698943495
# facebook/s2t-large-librispeech-asr:
# WER: 0.033264270371989584

# librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")

# model_names = ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-960h"]
# results = {}
# for model_name in model_names:
#     model = Wav2Vec2ForCTC.from_pretrained(model_name).to("cpu")
#     processor = Wav2Vec2Processor.from_pretrained(model_name)

#     def map_to_pred(batch):
#         input_values = processor(
#             batch["audio"][0]["array"], return_tensors="pt", padding="longest"
#         ).input_values
#         with torch.no_grad():
#             logits = model(input_values.to("cpu")).logits

#         predicted_ids = torch.argmax(logits, dim=-1)
#         transcription = processor.batch_decode(predicted_ids)
#         batch["transcription"] = transcription
#         return batch

#     results[model_name] = librispeech_eval.map(
#         map_to_pred, batched=True, batch_size=1, remove_columns=["audio"]
#     )

# for model_name in model_names:
#     print(f"{model_name}:")
#     print(
#         "WER:", wer(results[model_name]["text"], results[model_name]["transcription"])
#     )

# ----------------------------------------

from datasets import load_dataset
from evaluate import load
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor

model_names = [
    "facebook/s2t-small-librispeech-asr",
    "facebook/s2t-medium-librispeech-asr",
    "facebook/s2t-large-librispeech-asr",
]
results = {}

for model_name in model_names:
    librispeech_eval = load_dataset(
        "librispeech_asr", "clean", split="test"
    )  # change to "other" for other test dataset
    wer = load("wer")

    model = Speech2TextForConditionalGeneration.from_pretrained(model_name).to("cpu")
    processor = Speech2TextProcessor.from_pretrained(model_name, do_upper_case=True)

    def map_to_pred(batch):
        features = processor(
            batch["audio"]["array"],
            sampling_rate=16000,
            padding=True,
            return_tensors="pt",
        )
        input_features = features.input_features.to("cpu")
        attention_mask = features.attention_mask.to("cpu")

        gen_tokens = model.generate(
            input_features=input_features, attention_mask=attention_mask
        )
        batch["transcription"] = processor.batch_decode(
            gen_tokens, skip_special_tokens=True
        )[0]
        return batch

    results[model_name] = librispeech_eval.map(map_to_pred, remove_columns=["audio"])

# print(
#     "WER:", wer.compute(predictions=result["transcription"], references=result["text"])
# )

for model_name in model_names:
    print(f"{model_name}:")
    print(
        "WER:",
        wer.compute(
            predictions=results[model_name]["text"],
            references=results[model_name]["transcription"],
        ),
    )
