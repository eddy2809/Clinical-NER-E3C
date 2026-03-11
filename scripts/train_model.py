import json
import torch
import wandb
import os
import evaluate
import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)

#Configurazione
FILE_TRAIN = "data/processed/multi/dataset_train_full.json" 
FILE_TEST = "data/processed/multi/dataset_test.json"
NOME_MODELLO_SALVATO = "multi_bert_medico_full_shot_early"

#MODEL_NAME = "dbmdz/bert-base-italian-cased"
MODEL_NAME = "bert-base-multilingual-cased"
#MODEL_NAME = "xlm-roberta-large"

# Mapping etichette BIO in numeri
label_list = ['O', 'B-CLINENTITY', 'I-CLINENTITY']
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
seqeval = evaluate.load("seqeval") # Metrica NER


def BIO_encoding(testo, entita_estratte, tokenizer):
    
    """ Prende il testo grezzo e le coordinate, e restituisce i token di BERT 
    con le relative etichette B-I-O. """

    tokenized = tokenizer(testo, return_offsets_mapping=True, truncation=True, max_length=512)
    offsets = tokenized["offset_mapping"]
    # per avere i token come stringa (DEBUG)
    input_ids = tokenized["input_ids"]
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

    # Convertiamo gli ID in token testuali (DEBUG)
    tokens_text = tokenizer.convert_ids_to_tokens(input_ids)
    BIO_list = []
    for t, l in zip(tokens_text, labels):
        label_name = id2label[l] if l != -100 else "SPECIAL"
        BIO_list.append((t, label_name))
                    
    tokenized["labels"] = labels
    del tokenized["offset_mapping"] 
    return tokenized, BIO_list

def make_dataset(file_json):
    """Legge il file JSON e lo trasforma in un Dataset HuggingFace."""
    with open(file_json, 'r', encoding='utf-8') as f:
        dati = json.load(f)
        
    all_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for doc in dati:
        if doc is not None:
            tok_doc, token_text = BIO_encoding(doc["text"], doc["entities"], tokenizer)
            all_inputs["input_ids"].append(tok_doc["input_ids"])
            all_inputs["attention_mask"].append(tok_doc["attention_mask"])
            all_inputs["labels"].append(tok_doc["labels"])
        
    return Dataset.from_dict(all_inputs), token_text

# Evaluation metrics
def compute_metrics(p):
    """Calcola Precision, Recall e F1-Score"""
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



if __name__ == "__main__":

    wandb.init(
        project="clinical-ner-task", 
        name=NOME_MODELLO_SALVATO,
        tags=["BERT", "NER", "Clinical"]
    )
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)


    # --- 4. PREPARAZIONE DATI E MODELLO ---
    print(f"Preparazione dei dati per l'esperimento: {FILE_TRAIN}...")
    train_dataset,_ = make_dataset(FILE_TRAIN)
    eval_dataset,_ = make_dataset(FILE_TEST)

    print("Caricamento del modello BERT...")
    modello = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    # Il DataCollator si assicura che tutte le frasi abbiano la stessa lunghezza riempiendole di zeri
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)



    # TRAINING
    training_args = TrainingArguments(
        output_dir=f"./risultati_model/{NOME_MODELLO_SALVATO}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8, 
        num_train_epochs=100, 
        weight_decay=0.01,
        bf16=True, 
        save_only_model=True,   # Evita di salvare optimizer/scheduler (molto pesanti)
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=1,
        report_to="wandb",
        logging_strategy="epoch"
        
        # warmup_ratio=0.1,             # strategia warmup
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    
    print("\n INIZIO ADDESTRAMENTO ")
    trainer.train()
    trainer.save_model(f"model/{NOME_MODELLO_SALVATO}")
    print(f"Modello salvato in: model/{NOME_MODELLO_SALVATO}")
    trainer.state.save_to_json(f"model/{NOME_MODELLO_SALVATO}/trainer_state.json")