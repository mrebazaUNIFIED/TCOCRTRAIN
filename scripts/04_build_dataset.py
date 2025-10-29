from datasets import Dataset, DatasetDict
from pathlib import Path
import json
from PIL import Image
import os

def load_split(split_dir):
    labels_file = Path(split_dir) / "labels" / "annotations.json"
    with open(labels_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        img_path = Path(split_dir) / "images" / item["image"]
        if img_path.exists():
            examples.append({
                "image": str(img_path),
                "text": item["text"]
            })
    return Dataset.from_list(examples)

def main():
    base_dir = Path("data/processed")
    output_path = base_dir / "iam_hf_dataset"

    print("📦 Cargando splits...")
    train_ds = load_split(base_dir / "train")
    val_ds   = load_split(base_dir / "val")
    test_ds  = load_split(base_dir / "test")

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })

    print("💾 Guardando dataset Hugging Face...")
    dataset_dict.save_to_disk(str(output_path))
    print(f"✅ Dataset guardado en {output_path}")

if __name__ == "__main__":
    main()
