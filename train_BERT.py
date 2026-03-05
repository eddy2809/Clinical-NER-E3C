import json
import torch
import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

# --- 1. CONFIGURAZIONE ESPERIMENTO ---
# Qui decidi quale "secchiello" usare per l'esperimento attuale del prof.
# Cambia questo file per fare i test: "dataset_train_1_shot.json", "dataset_train_10_shot.json", ecc.
FILE_TRAIN = "json_datasets/multilanguage/dataset_train_full.json" 
FILE_TEST = "json_datasets/multilanguage/dataset_test.json"
NOME_MODELLO_SALVATO = "model/mul_roberta_medico_full_shot_early"

#MODEL_NAME = "dbmdz/bert-base-italian-cased"
# MODEL_NAME = "bert-base-multilingual-cased"
MODEL_NAME = "xlm-roberta-large"

# Mappiamo le etichette BIO in numeri (BERT ragiona a numeri, non a stringhe)
label_list = ['O', 'B-CLINENTITY', 'I-CLINENTITY']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
seqeval = evaluate.load("seqeval") # Metrica standard per il NER

# --- 2. FUNZIONI DI SUPPORTO (Riutilizziamo la tua funzione vincente) ---
def allinea_etichette_bio(testo, entita_estratte, tokenizer):
    """La funzione che abbiamo appena testato e validato!"""
    tokenized = tokenizer(testo, return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = tokenized["offset_mapping"]
    
    # Inizializza tutto a 'O' (0)
    labels = [label2id['O']] * len(offsets)
    
    for idx, (start_char, end_char) in enumerate(offsets):
        if start_char == 0 and end_char == 0:
            labels[idx] = -100 # -100 è il codice speciale di PyTorch per "Ignora questo token nel calcolo dell'errore"
            continue
            
        for ent in entita_estratte:
            ent_inizio = ent['inizio']
            ent_fine = ent['fine']
            if start_char >= ent_inizio and end_char <= ent_fine:
                if start_char == ent_inizio:
                    labels[idx] = label2id['B-CLINENTITY']
                else:
                    labels[idx] = label2id['I-CLINENTITY']
                    
    # Non ci servono più gli offsets per il training
    tokenized["labels"] = labels
    del tokenized["offset_mapping"] 
    return tokenized

def prepara_dataset(file_json):
    """Legge il file JSON salvato prima e lo trasforma in un Dataset HuggingFace."""
    with open(file_json, 'r', encoding='utf-8') as f:
        dati = json.load(f)
        
    all_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for doc in dati:
        # Applichiamo la magia dell'allineamento a ogni documento!
        tok_doc = allinea_etichette_bio(doc["text"], doc["entities"], tokenizer)
        all_inputs["input_ids"].append(tok_doc["input_ids"])
        all_inputs["attention_mask"].append(tok_doc["attention_mask"])
        all_inputs["labels"].append(tok_doc["labels"])
        
    return Dataset.from_dict(all_inputs)

# --- 3. PREPARAZIONE DATI E MODELLO ---
print(f"Preparazione dei dati per l'esperimento: {FILE_TRAIN}...")
train_dataset = prepara_dataset(FILE_TRAIN)
eval_dataset = prepara_dataset(FILE_TEST)

print("Caricamento del modello BERT sulla 4090...")
modello = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Il DataCollator si assicura che tutte le frasi abbiano la stessa lunghezza riempiendole di zeri
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# --- 4. METRICHE DI VALUTAZIONE ---
def compute_metrics(p):
    """Calcola Precision, Recall e F1-Score (Quelli che metterai nel grafico per il prof!)"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Rimuoviamo i -100 (i token speciali ignorati)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- 5. IL MOTORE DEL TRAINING (Ottimizzato per RTX 4090) ---
training_args = TrainingArguments(
    output_dir=f"./risultati_{NOME_MODELLO_SALVATO}",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # La 4090 divora batch size grandi senza problemi (16 per full shot)
    per_device_eval_batch_size=8, #prima era 16 per full shot
    num_train_epochs=50, # 5 epoche sono perfette per il Few-Shot
    weight_decay=0.01,
    bf16=True, # Magia della RTX 4000: Bfloat16 accelera il training senza perdere precisione
    logging_steps=10,
    save_only_model=True,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    greater_is_better=True,
    save_total_limit=1,
    # warmup_ratio=0.1,             # Il 10% dei passi iniziali serve a "scaldare" il modello , strategia warmup
    # lr_scheduler_type="cosine",   # La curva di discesa morbida
)

trainer = Trainer(
    model=modello,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# VIA AL TRAINING!
print("\n🚀 INIZIO ADDESTRAMENTO 🚀")
trainer.train()

# Salviamo il modello finetunato
trainer.save_model(NOME_MODELLO_SALVATO)
print(f"Modello salvato in: {NOME_MODELLO_SALVATO}")