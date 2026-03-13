import os
import json
import random
from pathlib import Path
import xml.etree.ElementTree as ET



def parse_xmi_e3c(file_path):
    """Estrae testo ed entità dal formato UIMA XMI."""

    p = Path(file_path)
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Errore nella lettura di {file_path}: {e}")
        return None, None
    
    testo_completo = ""
    entita_estratte = []
    
    # 1. Trova il testo (SofaString)
    for elem in root.iter():
        if elem.tag.endswith('Sofa'):
            testo_completo = elem.attrib.get('sofaString', '')
            break
            
    if not testo_completo:
        return None, None

    # 2. Trova le entità cliniche e le loro coordinate
    for elem in root.iter():
        if elem.tag.endswith('CLINENTITY'):
            begin = int(elem.attrib.get('begin', 0))
            end = int(elem.attrib.get('end', 0))
            
            if begin != 0 and end != 0:
                parola_medica = testo_completo[begin:end]
                entita_estratte.append({
                    "testo": parola_medica,
                    "tipo": "CLINENTITY",
                    "inizio": begin,
                    "fine": end
                })
    filename = p.name
    lingua = p.parts[4]
    json_element = None

    if testo_completo and len(testo_completo) > 10 and len(entita_estratte) > 0:
        json_element = {
            "id_doc": filename,
            "lingua": lingua,
            "text": testo_completo,
            "entities": entita_estratte
        }
                
    return json_element



#CONFIGURAZIONE
E3C_ROOT_DIR = "data/raw/E3C-Corpus-2.0.0/data_annotation"
SEED = 42 
random.seed(SEED)

if __name__ == "__main__":
    print(f"Scansione di tutte le lingue nella cartella: {E3C_ROOT_DIR}...")
    dataset_completo = []
    conteggio_lingue = {}

    for root_dir, dirs, files in os.walk(E3C_ROOT_DIR):
        #Layer 1 (gold standard)
        if "layer1" in root_dir.lower():
            for filename in files:
                if filename.endswith(".xml"):
                    percorso_file = os.path.join(root_dir, filename)
            
                    json_element = parse_xmi_e3c(percorso_file) 
                    dataset_completo.append(json_element)

    print("\n=== STATISTICHE DATASET MULTILINGUE ===")
    for lang, count in conteggio_lingue.items():
        print(f"- {lang}: {count} documenti")
    print(f"TOTALE: {len(dataset_completo)} documenti clinici pronti per l'addestramento.")


    print(f"Trovati e processati correttamente {len(dataset_completo)} documenti clinici.")

    # Data Splitting
   
    random.shuffle(dataset_completo)

    # Split train/val (80/20)
    num_val = int(len(dataset_completo) * 0.2)
    val_set = dataset_completo[:num_val]
    train_pool = dataset_completo[num_val:]

    print(f"Documenti riservati per il Val (Test Set): {len(val_set)}")
    print(f"Documenti disponibili per il Training: {len(train_pool)}")

    # Crezione pool per One-Shot e Few-Shot
    one_shot_set = train_pool[:1]       
    few_shot_set_5 = train_pool[:5]     
    few_shot_set_10 = train_pool[:10]   
    full_train_set = train_pool         


    def salva_json(dati, nome_file):
        with open(nome_file, 'w', encoding='utf-8') as f:
            json.dump(dati, f, ensure_ascii=False, indent=2)
    
    salva_json(val_set, "data/processed/multi/dataset_val.json")
    salva_json(one_shot_set, "data/processed/multi/dataset_train_1_shot.json")
    salva_json(few_shot_set_5, "data/processed/multi/dataset_train_5_shot.json")
    salva_json(few_shot_set_10, "data/processed/multi/dataset_train_10_shot.json")
    salva_json(full_train_set, "data/processed/multi/dataset_train_full.json")

    print("\nFile JSON generati: ")
    print("- dataset_val.json")
    print("- dataset_train_1_shot.json")
    print("- dataset_train_5_shot.json")
    print("- dataset_train_10_shot.json")
    print("- dataset_train_full.json")
