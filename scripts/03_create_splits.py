"""
Script para crear splits train/val/test del IAM Dataset
VERSIÓN CORREGIDA - Asegura que test tenga datos
"""
from pathlib import Path
import json
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

def load_official_splits(ascii_dir):
    """
    Cargar splits oficiales del IAM Dataset desde los archivos de texto
    
    Returns:
        dict: Mapeo de writer_id a split (trainset, validationset1, testset)
    """
    splits = {}
    
    # Archivos de splits oficiales
    split_files = {
        'trainset': ascii_dir / 'trainset.txt',
        'validationset1': ascii_dir / 'validationset1.txt',
        'validationset2': ascii_dir / 'validationset2.txt',
        'testset': ascii_dir / 'testset.txt'
    }
    
    found_splits = False
    for split_name, split_file in split_files.items():
        if split_file.exists():
            found_splits = True
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('-')
                        if len(parts) > 0:
                            writer_id = parts[0]
                            splits[writer_id] = split_name
    
    if not found_splits:
        return None
    
    return splits

def create_splits(processed_dir="data/processed", use_official_splits=True, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Crear splits train/val/test
    
    Args:
        processed_dir: Directorio con datos procesados
        use_official_splits: Si True, intenta usar los splits oficiales de IAM
        train_ratio: Proporción para train (si no hay splits oficiales)
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
    """
    processed_path = Path(processed_dir)
    all_data_dir = processed_path / 'all'
    
    print("="*60)
    print("CREACIÓN DE SPLITS TRAIN/VAL/TEST")
    print("="*60)
    
    # Verificar que existe el directorio procesado
    if not all_data_dir.exists():
        print(f"\n❌ Error: {all_data_dir} no existe")
        print("   Ejecuta primero: python scripts/02_preprocess_iam.py")
        sys.exit(1)
    
    # Verificar que existen las imágenes
    images_dir = all_data_dir / 'images'
    if not images_dir.exists() or not list(images_dir.glob('*.png')):
        print(f"\n❌ Error: No hay imágenes en {images_dir}")
        sys.exit(1)
    
    print(f"\n✓ Directorio encontrado: {all_data_dir}")
    print(f"✓ Imágenes encontradas: {len(list(images_dir.glob('*.png')))}")
    
    # Cargar anotaciones
    annotations_file = all_data_dir / 'labels' / 'annotations.json'
    
    if not annotations_file.exists():
        print(f"\n❌ Error: Archivo de anotaciones no encontrado: {annotations_file}")
        sys.exit(1)
    
    print(f"✓ Anotaciones encontradas: {annotations_file}")
    
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"\nTotal de muestras: {len(annotations)}")
    
    # Verificar que los ratios suman 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Los ratios deben sumar 1.0"
    
    # Decidir método de split
    official_splits = None
    if use_official_splits:
        print("\nBuscando splits oficiales de IAM...")
        ascii_dir = Path("data/extracted/ascii")
        
        if ascii_dir.exists():
            official_splits = load_official_splits(ascii_dir)
            
            if official_splits:
                print(f"✓ Splits oficiales encontrados: {len(official_splits)} writers")
            else:
                print("⚠️  Archivos de splits oficiales no encontrados")
        else:
            print("⚠️  Directorio ascii no encontrado")
    
    # Obtener splits
    if official_splits:
        print("\n✓ Usando splits oficiales de IAM\n")
        
        # Clasificar anotaciones según writer ID
        train_annotations = []
        val_annotations = []
        test_annotations = []
        unknown_writers = set()
        
        for ann in annotations:
            img_name = ann['image']
            writer_id = img_name.split('-')[0]
            
            split = official_splits.get(writer_id)
            
            if split == 'trainset':
                train_annotations.append(ann)
            elif split in ['validationset1', 'validationset2']:
                val_annotations.append(ann)
            elif split == 'testset':
                test_annotations.append(ann)
            else:
                unknown_writers.add(writer_id)
                train_annotations.append(ann)
        
        if unknown_writers:
            print(f"⚠️  {len(unknown_writers)} writers sin split oficial (asignados a train)")
        
        # Si no hay test o val, hacer split manual
        if len(test_annotations) == 0 or len(val_annotations) == 0:
            print("⚠️  Splits oficiales incompletos, usando split aleatorio")
            official_splits = None
    
    if not official_splits:
        print("\n✓ Creando splits aleatorios\n")
        
        # Split aleatorio
        # Primero separar test
        train_val, test_annotations = train_test_split(
            annotations, 
            test_size=test_ratio, 
            random_state=42, 
            shuffle=True
        )
        
        # Luego separar train y val
        val_size = val_ratio / (train_ratio + val_ratio)  # Ajustar proporción
        train_annotations, val_annotations = train_test_split(
            train_val,
            test_size=val_size,
            random_state=42,
            shuffle=True
        )
    
    # Mostrar distribución
    print("Distribución de datos:")
    print(f"  Train: {len(train_annotations):5d} ({len(train_annotations)/len(annotations)*100:5.1f}%)")
    print(f"  Val:   {len(val_annotations):5d} ({len(val_annotations)/len(annotations)*100:5.1f}%)")
    print(f"  Test:  {len(test_annotations):5d} ({len(test_annotations)/len(annotations)*100:5.1f}%)")
    print(f"  Total: {len(annotations):5d} (100.0%)")
    
    # Verificación
    total_check = len(train_annotations) + len(val_annotations) + len(test_annotations)
    if total_check != len(annotations):
        print(f"\n⚠️  Advertencia: Total no coincide! ({total_check} vs {len(annotations)})")
    
    # Verificar que todos los splits tienen datos
    if len(train_annotations) == 0:
        print("\n❌ Error: Train set está vacío!")
        sys.exit(1)
    if len(val_annotations) == 0:
        print("\n❌ Error: Validation set está vacío!")
        sys.exit(1)
    if len(test_annotations) == 0:
        print("\n❌ Error: Test set está vacío!")
        sys.exit(1)
    
    # Crear directorios y copiar archivos para cada split
    for split_name, split_annotations in [
        ('train', train_annotations),
        ('val', val_annotations),
        ('test', test_annotations)
    ]:
        print(f"\n{'─'*60}")
        print(f"Procesando split: {split_name.upper()}")
        print(f"{'─'*60}")
        
        split_dir = processed_path / split_name
        images_dir_dest = split_dir / 'images'
        labels_dir_dest = split_dir / 'labels'
        
        # Limpiar directorios existentes
        if images_dir_dest.exists():
            shutil.rmtree(images_dir_dest)
        if labels_dir_dest.exists():
            shutil.rmtree(labels_dir_dest)
        
        # Crear directorios
        images_dir_dest.mkdir(parents=True, exist_ok=True)
        labels_dir_dest.mkdir(parents=True, exist_ok=True)
        
        print(f"Copiando {len(split_annotations)} archivos...")
        
        copied = 0
        missing = 0
        
        # Copiar imágenes
        for ann in tqdm(split_annotations, desc=f"Copiando {split_name}"):
            src_img = all_data_dir / 'images' / ann['image']
            dst_img = images_dir_dest / ann['image']
            
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                copied += 1
            else:
                missing += 1
                print(f"\n⚠️  Imagen no encontrada: {src_img}")
        
        # Guardar anotaciones
        annotations_file_dest = labels_dir_dest / 'annotations.json'
        with open(annotations_file_dest, 'w', encoding='utf-8') as f:
            json.dump(split_annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ {split_name}:")
        print(f"  - Imágenes copiadas: {copied}")
        if missing > 0:
            print(f"  - Imágenes faltantes: {missing}")
        print(f"  - Anotaciones guardadas: {annotations_file_dest}")
        
        # Verificar que se copiaron correctamente
        copied_files = list(images_dir_dest.glob('*.png'))
        print(f"  - Archivos en destino: {len(copied_files)}")
        
        if len(copied_files) != len(split_annotations):
            print(f"  ⚠️  ADVERTENCIA: Esperados {len(split_annotations)}, encontrados {len(copied_files)}")
    
    print("\n" + "="*60)
    print("✓ SPLITS CREADOS EXITOSAMENTE")
    print("="*60)
    print("\nEstructura final:")
    print(f"  {processed_dir}/")
    print(f"  ├── all/        (datos originales)")
    print(f"  ├── train/")
    print(f"  │   ├── images/  ({len(train_annotations)} imágenes)")
    print(f"  │   └── labels/  (annotations.json)")
    print(f"  ├── val/")
    print(f"  │   ├── images/  ({len(val_annotations)} imágenes)")
    print(f"  │   └── labels/  (annotations.json)")
    print(f"  └── test/")
    print(f"      ├── images/  ({len(test_annotations)} imágenes)")
    print(f"      └── labels/  (annotations.json)")
    
    print("\n✓ Siguiente paso: python scripts/05_verify_data.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Crear splits train/val/test')
    parser.add_argument('--no-official', action='store_true',
                       help='No usar splits oficiales, crear aleatorios')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Proporción para train (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Proporción para validación (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Proporción para test (default: 0.15)')
    
    args = parser.parse_args()
    
    # Verificar que suman 1.0
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 0.01:
        print(f"❌ Error: Los ratios deben sumar 1.0 (actual: {total})")
        sys.exit(1)
    
    create_splits(
        use_official_splits=not args.no_official,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )