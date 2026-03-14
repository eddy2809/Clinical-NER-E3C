import os
import json
import random
from pathlib import Path
import xml.etree.ElementTree as ET

def parse_xmi_e3c(file_path):
    """Estrae testo ed entità dal formato UIMA XMI in modo sicuro."""
    p = Path(file_path)
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Errore nella lettura di {file_path}: {e}")
        return None
    
    testo_completo = ""
    entita_estratte = []
    
    # 1. Trova il testo (SofaString)
    for elem in root.iter():
        if elem.tag.endswith('Sofa'):
            testo_completo = elem.attrib.get('sofaString', '')
            break
            
    if not testo_completo:
        return None

    # 2. Trova le entità cliniche
    for elem in root.iter():
        if elem.tag.endswith('CLINENTITY'):
            begin = int(elem.attrib.get('begin', 0))
            end = int(elem.attrib.get('end', 0))
            
            if begin != 0 and end != 0:
                entita_estratte.append({
                    "testo": testo_completo[begin:end],
                    "tipo": "CLINENTITY",
                    "inizio": begin,
                    "fine": end
                })

    # Estrae la lingua guardando due cartelle "sopra" (es: Italian/layer1/file.xml -> Italian)
    lingua = p.parent.parent.name 

    if testo_completo and len(testo_completo) > 10 and len(entita_estratte) > 0:
        return {
            "id_doc": p.name,
            "lingua": lingua,
            "text": testo_completo,
            "entities": entita_estratte
        }
    return None


def genera_dataset(e3c_root_dir, output_base_dir, modalita="multi", seed=42):
    """
    Scansiona la directory e genera i JSON. 
    modalita: 'ita' per estrarre solo l'italiano, 'multi' per tutte le lingue.
    """
    random.seed(seed)
    dataset_completo = []
    conteggio_lingue = {}
    

    for root_dir, dirs, files in os.walk(e3c_root_dir):
        # Cerchiamo solo nei Layer 1 (Gold Standard)
        if "layer1" in root_dir.lower():
            
            # Se siamo in modalità "ita", ignoriamo le cartelle che non contengono "italian"
            if modalita == "ita" and "italian" not in root_dir.lower():
                continue
                
            for filename in files:
                if filename.endswith(".xml"):
                    percorso_file = os.path.join(root_dir, filename)
                    json_element = parse_xmi_e3c(percorso_file) 
                    
                    if json_element:
                        dataset_completo.append(json_element)
                        
                        # Aggiorniamo il conteggio dinamico delle lingue
                        lingua = json_element["lingua"]
                        conteggio_lingue[lingua] = conteggio_lingue.get(lingua, 0) + 1

    # --- STATISTICHE ---
    print(f"Trovati e processati {len(dataset_completo)} documenti validi.")
    for lang, count in conteggio_lingue.items():
        print(f"  - {lang}: {count} documenti")

    # --- SPLIT DATASET ---
    random.shuffle(dataset_completo)
    num_val = int(len(dataset_completo) * 0.2)
    val_set = dataset_completo[:num_val]
    train_pool = dataset_completo[num_val:]

    print(f"Split effettuato: {len(val_set)} Test/Val | {len(train_pool)} Training")

    # --- SALVATAGGIO ---
    out_dir = Path(output_base_dir) / modalita
    out_dir.mkdir(parents=True, exist_ok=True)

    def salva_json(dati, nome_file):
        with open(out_dir / nome_file, 'w', encoding='utf-8') as f:
            json.dump(dati, f, ensure_ascii=False, indent=2)

    salva_json(val_set, "dataset_val.json")
    salva_json(train_pool[:1], "dataset_train_1_shot.json")
    salva_json(train_pool[:2], "dataset_train_2_shot.json")
    salva_json(train_pool[:5], "dataset_train_5_shot.json")
    salva_json(train_pool[:10], "dataset_train_10_shot.json")
    salva_json(train_pool[:20], "dataset_train_20_shot.json")
    salva_json(train_pool[:50], "dataset_train_50_shot.json")
    salva_json(train_pool[:100], "dataset_train_100_shot.json")
    salva_json(train_pool[:150], "dataset_train_150_shot.json")
    salva_json(train_pool[:200], "dataset_train_200_shot.json")
    salva_json(train_pool, "dataset_train_full.json")
    
    print(f"File JSON salvati nella cartella: {out_dir}\n")


if __name__ == "__main__":
    
    E3C_ROOT = Path("data") / "raw" / "E3C-Corpus-2.0.0" / "data_annotation"
    OUT_BASE = Path("data") / "processed" 
    
    # Eseguiamo per entrambe le modalità
    genera_dataset(E3C_ROOT, OUT_BASE, modalita="multi")
    