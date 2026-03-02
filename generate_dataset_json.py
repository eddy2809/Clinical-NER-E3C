import os
import json
import random
import xml.etree.ElementTree as ET

# --- CONFIGURAZIONE ---
XML_DIR = "E3C-Corpus-2.0.0\E3C-Corpus-2.0.0\data_annotation\Italian\layer1"  # CARTELLA CON TUTTI I TUOI FILE XML/XMI DELL'E3C
SEED = 42 # Seed fisso per la tesi: garantisce che i "secchielli" siano sempre identici se lo rilanci
random.seed(SEED)

def parse_xmi_e3c(file_path):
    """Estrae testo ed entità dal formato UIMA XMI (la logica che abbiamo testato con successo)."""
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
                
    return testo_completo, entita_estratte

# --- 1. LETTURA DI TUTTA LA CARTELLA ---
print(f"Analisi dei file nella cartella '{XML_DIR}' in corso...")
dataset_completo = []

for filename in os.listdir(XML_DIR):
    if not filename.endswith(".xml"): 
        continue
        
    percorso_file = os.path.join(XML_DIR, filename)
    testo, entita = parse_xmi_e3c(percorso_file)
    
    # Teniamo solo i referti validi (che hanno almeno un po' di testo e almeno una malattia annotata)
    if testo and len(testo) > 10 and len(entita) > 0:
        dataset_completo.append({
            "id_doc": filename,
            "text": testo,
            "entities": entita
        })

print(f"Trovati e processati correttamente {len(dataset_completo)} documenti clinici.")

# --- 2. CREAZIONE DEI "SECCHIELLI" PER GLI ESPERIMENTI (Data Splitting) ---
# Mescoliamo i documenti in modo casuale
random.shuffle(dataset_completo)

# Riserviamo il 20% dei dati per il Test finale (il prof vorrà vedere come si comporta su dati MAI visti)
num_test = int(len(dataset_completo) * 0.2)
test_set = dataset_completo[:num_test]
train_pool = dataset_completo[num_test:] # Il restante 80% lo usiamo per pescare i dati di addestramento

print(f"Documenti riservati per il Test (Test Set): {len(test_set)}")
print(f"Documenti disponibili per il Training: {len(train_pool)}")

# Creiamo le porzioni per confrontare One-Shot e Few-Shot
one_shot_set = train_pool[:1]       # Solo 1 documento
few_shot_set_5 = train_pool[:5]     # Solo 5 documenti
few_shot_set_10 = train_pool[:10]   # Solo 10 documenti
full_train_set = train_pool         # Tutti i documenti rimasti

# --- 3. SALVATAGGIO DEI FILE JSON ---
def salva_json(dati, nome_file):
    with open(nome_file, 'w', encoding='utf-8') as f:
        json.dump(dati, f, ensure_ascii=False, indent=2)

salva_json(test_set, "json_datasets/dataset_test.json")
salva_json(one_shot_set, "json_datasets/dataset_train_1_shot.json")
salva_json(few_shot_set_5, "json_datasets/dataset_train_5_shot.json")
salva_json(few_shot_set_10, "json_datasets/dataset_train_10_shot.json")
salva_json(full_train_set, "json_datasets/dataset_train_full.json")

print("\n✅ File JSON generati con successo!")
print("- dataset_test.json")
print("- dataset_train_1_shot.json")
print("- dataset_train_5_shot.json")
print("- dataset_train_10_shot.json")
print("- dataset_train_full.json")
