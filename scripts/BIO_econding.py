from extrac_json import parse_xmi_e3c
from transformers import AutoTokenizer

# Carichiamo il Tokenizer ufficiale di BERT per l'italiano
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-cased")

def allinea_etichette_bio(testo, entita_estratte, tokenizer):
    """
    Prende il testo grezzo e le coordinate, e restituisce i token di BERT 
    con le relative etichette B-I-O.
    """
    # 1. Chiediamo a BERT di spezzare il testo e, FONDAMENTALE, di dirci 
    # le coordinate esatte (offset_mapping) di ogni pezzetto che ha creato.
    tokenized = tokenizer(
        testo, 
        return_offsets_mapping=True, 
        truncation=True, 
        max_length=512 # BERT accetta massimo 512 token alla volta
    )
    
    offsets = tokenized["offset_mapping"]
    input_ids = tokenized["input_ids"]
    
    # Inizializziamo tutte le etichette a 'O' (Outside = non è una malattia)
    labels = ['O'] * len(input_ids)
    
    # 2. Logica di allineamento
    for idx, (start_char, end_char) in enumerate(offsets):
        # Ignoriamo i token speciali di BERT come [CLS] e [SEP] (hanno coordinate 0,0)
        if start_char == 0 and end_char == 0:
            labels[idx] = 'IGN' # Lo ignoreremo durante il training
            continue
            
        # Controlliamo se questo token cade all'interno delle coordinate di una nostra malattia
        for ent in entita_estratte:
            ent_inizio = ent['inizio']
            ent_fine = ent['fine']
            
            # Se il token è compreso nelle coordinate dell'entità...
            if start_char >= ent_inizio and end_char <= ent_fine:
                # È il primo pezzo della parola?
                if start_char == ent_inizio:
                    labels[idx] = 'B-CLINENTITY'
                else:
                    # È un pezzo successivo o una parola successiva della stessa malattia
                    labels[idx] = 'I-CLINENTITY'
                    
    # Restituiamo i token leggibili per controllo umano e le etichette allineate
    tokens_leggibili = tokenizer.convert_ids_to_tokens(input_ids)
    return tokens_leggibili, labels

# --- TEST DEL CODICE ---
# Usa le variabili `testo` ed `entita` che hai estratto con lo script precedente
# (Assicurati di aver eseguito lo script precedente prima di questo)
file_test = "E3C-Corpus-2.0.0\E3C-Corpus-2.0.0\data_annotation\Italian\layer1\IT100002.xml"
testo, entita = parse_xmi_e3c(file_test)


# if testo and entita:
#     tokens, labels = allinea_etichette_bio(testo, entita, tokenizer)
    
#     print("\n=== RISULTATO ALLINEAMENTO BIO PER BERT ===")
#     # Stampiamo i primi 50 token per vedere come ha lavorato
#     for tok, lab in zip(tokens[:50], labels[:50]):
#         # Stampiamo solo per vedere come ha formattato
#         print(f"{tok.ljust(15)} : {lab}")

if testo and entita:
    tokens, labels = allinea_etichette_bio(testo, entita, tokenizer)
    
    print("\n=== ENTITÀ ALLINEATE CORRETTAMENTE TROVATE DA BERT ===")
    
    # Contatore per capire se ha trovato qualcosa
    trovate = 0
    
    for tok, lab in zip(tokens, labels):
        # Stampiamo SOLO i token che NON sono 'O' e NON sono 'IGN'
        if lab != 'O' and lab != 'IGN':
            print(f"{tok.ljust(15)} : {lab}")
            trovate += 1
