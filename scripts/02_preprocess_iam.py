"""
Script para preprocesar el IAM Dataset y convertirlo al formato requerido por DocTR
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
import sys
from collections import defaultdict

class IAMPreprocessor:
    def __init__(self, extracted_dir="data/extracted", processed_dir="data/processed"):
        self.extracted_dir = Path(extracted_dir)
        self.processed_dir = Path(processed_dir)
        
        # Verificar que existen los directorios
        if not self.extracted_dir.exists():
            print(f"❌ Error: {extracted_dir} no existe")
            print("   Ejecuta primero: python scripts/01_extract_data.py")
            sys.exit(1)
        
        self.ascii_dir = self.extracted_dir / 'ascii'
        self.lines_dir = self.extracted_dir / 'lines'
        
        # Cargar mapeo de line_id a texto
        self.text_map = self.load_text_mapping()
        
    def load_text_mapping(self):
        """Cargar mapeo de line_id a texto desde el archivo ascii"""
        text_map = {}
        ascii_file = self.ascii_dir / 'lines.txt'
        
        if not ascii_file.exists():
            print(f"⚠️  Advertencia: {ascii_file} no encontrado")
            return text_map
        
        print("Cargando mapeo de textos...")
        with open(ascii_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Saltar comentarios
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split(' ')
                if len(parts) >= 9:
                    line_id = parts[0]
                    # status (ok, err)
                    status = parts[1]
                    
                    # Solo procesar líneas ok
                    if status == 'ok':
                        # El texto está después del campo 8
                        text = ' '.join(parts[8:])
                        # Limpiar caracteres especiales de IAM
                        text = text.replace('|', ' ').strip()
                        if text:  # Solo guardar si hay texto
                            text_map[line_id] = text
        
        print(f"✓ Cargadas {len(text_map)} líneas de texto")
        return text_map
    
    def preprocess_image(self, img, target_height=64):
        """
        Preprocesar imagen: binarización, normalización
        
        Args:
            img: Imagen en escala de grises
            target_height: Altura objetivo en píxeles
        """
        # Binarización usando Otsu
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Normalizar altura manteniendo aspect ratio
        height, width = binary.shape
        
        if height == 0:
            return None
        
        scale = target_height / height
        new_width = int(width * scale)
        
        if new_width == 0:
            return None
        
        resized = cv2.resize(binary, (new_width, target_height), 
                            interpolation=cv2.INTER_AREA)
        
        return resized
    
    def process_all(self):
        """Procesar todo el dataset"""
        print("\n" + "="*60)
        print("PREPROCESAMIENTO DEL IAM DATASET")
        print("="*60)
        
        # Buscar todas las imágenes de líneas
        if not self.lines_dir.exists():
            print(f"❌ Error: Directorio de líneas no encontrado: {self.lines_dir}")
            sys.exit(1)
        
        line_images = list(self.lines_dir.rglob('*.png'))
        
        if not line_images:
            print(f"❌ Error: No se encontraron imágenes en {self.lines_dir}")
            sys.exit(1)
        
        print(f"\n✓ Encontradas {len(line_images)} imágenes de líneas")
        
        # Procesar imágenes
        processed_data = []
        skipped = 0
        
        print("\nProcesando imágenes...")
        for img_path in tqdm(line_images):
            # Obtener line_id desde el nombre del archivo
            line_id = img_path.stem
            
            # Verificar si tenemos el texto para esta línea
            if line_id not in self.text_map:
                skipped += 1
                continue
            
            # Leer imagen
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                skipped += 1
                continue
            
            # Preprocesar
            processed_img = self.preprocess_image(img)
            
            if processed_img is None:
                skipped += 1
                continue
            
            # Guardar información
            processed_data.append({
                'line_id': line_id,
                'original_path': str(img_path),
                'text': self.text_map[line_id],
                'image': processed_img
            })
        
        print(f"\n✓ Procesadas: {len(processed_data)} imágenes")
        print(f"⚠️  Omitidas: {skipped} imágenes")
        
        if not processed_data:
            print("❌ Error: No se procesaron imágenes válidas")
            sys.exit(1)
        
        # Guardar datos procesados (sin dividir en splits todavía)
        self.save_processed_data(processed_data)
        
        return len(processed_data)
    
    def save_processed_data(self, data):
        """Guardar datos procesados"""
        output_dir = self.processed_dir / 'all'
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nGuardando datos procesados...")
        annotations = []
        
        for item in tqdm(data):
            # Guardar imagen
            img_filename = f"{item['line_id']}.png"
            img_path = images_dir / img_filename
            cv2.imwrite(str(img_path), item['image'])
            
            # Guardar anotación
            annotations.append({
                'image': img_filename,
                'text': item['text']
            })
        
        # Guardar archivo de anotaciones
        annotations_file = labels_dir / 'annotations.json'
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Datos guardados en: {output_dir}")
        print(f"  - Imágenes: {len(annotations)}")
        print(f"  - Anotaciones: {annotations_file}")

if __name__ == "__main__":
    preprocessor = IAMPreprocessor()
    total = preprocessor.process_all()
    
    print("\n" + "="*60)
    print("✓ PREPROCESAMIENTO COMPLETADO")
    print("="*60)
    print(f"Total de muestras procesadas: {total}")
    print("\nSiguiente paso: python scripts/03_create_splits.py")