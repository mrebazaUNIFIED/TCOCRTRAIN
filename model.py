from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# Ruta donde se guardará el modelo
MODEL_DIR = "./models/trocr-large-handwritten"

print("🚀 Descargando modelo 'microsoft/trocr-large-handwritten'...")

# Descargar el procesador (tokenizer + feature extractor)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
processor.save_pretrained(MODEL_DIR)

# Descargar el modelo
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
model.save_pretrained(MODEL_DIR)

print(f"✅ Modelo descargado y guardado en: {MODEL_DIR}")
