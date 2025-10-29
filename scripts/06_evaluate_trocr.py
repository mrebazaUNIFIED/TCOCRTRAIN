"""
Evaluar modelo TrOCR entrenado en IAM test set
"""
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
import torch
import evaluate

def evaluate_model(model_path="./trocr_iam_model", dataset_path="data/processed/hf_iam_dataset"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    processor = TrOCRProcessor.from_pretrained(model_path)
    dataset = load_from_disk(dataset_path)["test"]

    cer = evaluate.load("cer")
    wer = evaluate.load("wer")

    preds, refs = [], []

    for sample in tqdm(dataset, desc="Evaluando"):
        image = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        preds.append(pred)
        refs.append(sample["text"])

    cer_score = cer.compute(predictions=preds, references=refs)
    wer_score = wer.compute(predictions=preds, references=refs)

    print("\n✅ Resultados finales:")
    print(f"CER: {cer_score:.4f}")
    print(f"WER: {wer_score:.4f}")

if __name__ == "__main__":
    evaluate_model()
