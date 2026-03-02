import os
import json
import xml.etree.ElementTree as ET

# --- CONFIGURAZIONE ---
XML_DIR = "E3C-Corpus-2.0.0\E3C-Corpus-2.0.0\data_annotation\Italian\layer1" # La cartella dove hai i file .xml (che in realtà sono XMI)
OUTPUT_FILE = "dataset_e3c_pulito.jsonl"

def parse_xmi_e3c(file_path):
    """Estrae testo ed entità dal formato UIMA XMI dell'E3C."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    testo_completo = ""
    entita_estratte = []
    
    # 1. Trova il testo (Cerca il tag che finisce per 'Sofa')
    for elem in root.iter():
        if elem.tag.endswith('Sofa'):
            testo_completo = elem.attrib.get('sofaString', '')
            break
            
    if not testo_completo:
        return None, None

    # 2. Trova le entità cliniche (Cerca i tag che finiscono per 'CLINENTITY')
    for elem in root.iter():
        if elem.tag.endswith('CLINENTITY'):
            # Prendiamo le coordinate di inizio e fine
            begin = int(elem.attrib.get('begin', 0))
            end = int(elem.attrib.get('end', 0))
            
            # Se abbiamo le coordinate, ritagliamo la parola dal testo completo!
            if begin != 0 and end != 0:
                parola_medica = testo_completo[begin:end]
                
                # Aggiungiamo l'entità alla nostra lista
                entita_estratte.append({
                    "testo": parola_medica,
                    "tipo": "CLINENTITY",
                    "inizio": begin,
                    "fine": end
                })
                
    return testo_completo, entita_estratte

# --- TESTIAMO LO SCRIPT SUL TUO FILE ---
# Ipotizzando che il file si chiami IT100002.xml
file_test = "E3C-Corpus-2.0.0\E3C-Corpus-2.0.0\data_annotation\Italian\layer1\IT100002.xml" 

# Se hai salvato il file nella stessa cartella dello script, decommenta e prova:
if os.path.exists(file_test):
    testo, entita = parse_xmi_e3c(file_test)
    
    print("=== TESTO ESTRATTO ===")
    print(testo[:200] + "...\n") # Stampo solo i primi 200 caratteri
    
    print("=== ENTITA' MEDICHE TROVATE ===")
    for ent in entita:
        print(f"- {ent['testo']} (Caratteri: {ent['inizio']} -> {ent['fine']})")
else:
    print(f"File {file_test} non trovato. Mettilo nella cartella ed esegui lo script!")