import torch
from datasets import load_from_disk
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from PIL import Image

# --- Configuración ---
DATA_DIR = "data/processed/iam_hf_dataset"
MODEL_DIR = "./models/trocr-large-handwritten"
OUTPUT_DIR = "./models/trocr-finetuned-iam"

# --- Cargar dataset ---
print("----Cargando dataset----")
dataset = load_from_disk(DATA_DIR)

# --- Cargar modelo y processor ---
print("----Cargando modelo base---")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")

# 🔧 Solución al error de decoder_start_token
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.vocab_size = model.config.decoder.vocab_size  # evita warnings

# --- Preprocesamiento de datos ---
def preprocess(batch):
    images = [Image.open(p).convert("RGB") for p in batch["image"]]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    labels = processor.tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128).input_ids
    batch["pixel_values"] = pixel_values
    batch["labels"] = labels
    return batch

print("----Preprocesando dataset---")
train_ds = dataset["train"].map(preprocess, batched=True, batch_size=2)  # batch más pequeño
val_ds   = dataset["validation"].map(preprocess, batched=True, batch_size=2)

train_ds.set_format(type="torch", columns=["pixel_values", "labels"])
val_ds.set_format(type="torch", columns=["pixel_values", "labels"])

# --- Parámetros de entrenamiento ---
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,   # GPU grande? mantener 1
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,   # acumula gradientes para simular batch 4
    output_dir=OUTPUT_DIR,
    logging_dir="./logs",
    logging_steps=50,
    num_train_epochs=3,
    save_total_limit=2,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
)

# --- Entrenador ---
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=default_data_collator,
)

# --- Entrenar ---
print("----Entrenando modelo----")
trainer.train()

# --- Guardar modelo final ---
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✅ Modelo final guardado en {OUTPUT_DIR}")
