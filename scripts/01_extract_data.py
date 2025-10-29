"""
Script para extraer todos los archivos .tgz del IAM Dataset
"""
import tarfile
import os
from pathlib import Path
from tqdm import tqdm
import sys

def extract_iam_data(raw_dir="data/raw", extracted_dir="data/extracted"):
    """
    Extraer todos los archivos .tgz del IAM dataset
    
    Args:
        raw_dir: Directorio con archivos .tgz
        extracted_dir: Directorio donde extraer
    """
    raw_path = Path(raw_dir)
    extracted_path = Path(extracted_dir)
    
    # Verificar que existe el directorio raw
    if not raw_path.exists():
        print(f"❌ Error: El directorio {raw_dir} no existe")
        print(f"   Por favor, crea el directorio y coloca los archivos .tgz ahí")
        sys.exit(1)
    
    # Mapeo de archivos y sus destinos
    tgz_files = {
        'ascii.tgz': 'ascii',
        'formsA-D.tgz': 'forms',
        'formsE-H.tgz': 'forms',
        'formsI-Z.tgz': 'forms',
        'lines.tgz': 'lines',
        'sentences.tgz': 'sentences',
        'words.tgz': 'words',
        'xml.tgz': 'xml'
    }
    
    print("="*60)
    print("EXTRACCIÓN DE ARCHIVOS IAM DATASET")
    print("="*60)
    
    # Verificar archivos disponibles
    available_files = []
    missing_files = []
    
    for tgz_file in tgz_files.keys():
        tgz_path = raw_path / tgz_file
        if tgz_path.exists():
            available_files.append(tgz_file)
        else:
            missing_files.append(tgz_file)
    
    if missing_files:
        print("\n⚠️  Archivos faltantes:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nDescárgalos de: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")
    
    if not available_files:
        print("\n❌ No se encontraron archivos .tgz en data/raw/")
        sys.exit(1)
    
    print(f"\n✓ Archivos encontrados: {len(available_files)}/{len(tgz_files)}")
    
    # Extraer cada archivo
    for tgz_file in tqdm(available_files, desc="Extrayendo archivos"):
        tgz_path = raw_path / tgz_file
        target_folder = tgz_files[tgz_file]
        output_path = extracted_path / target_folder
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\n  Extrayendo {tgz_file} -> {output_path}")
            with tarfile.open(tgz_path, 'r:gz') as tar:
                tar.extractall(path=output_path)
            print(f"  ✓ {tgz_file} extraído correctamente")
        except Exception as e:
            print(f"  ❌ Error extrayendo {tgz_file}: {str(e)}")
    
    print("\n" + "="*60)
    print("✓ EXTRACCIÓN COMPLETADA")
    print("="*60)
    print(f"Archivos extraídos en: {extracted_dir}/")

if __name__ == "__main__":
    extract_iam_data()