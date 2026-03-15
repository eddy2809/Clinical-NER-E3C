import json
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
)

#Configurazione
FILE_TRAIN = "data/processed/multi/dataset_train_full.json" 
FILE_EVAL = "data/processed/multi/dataset_val.json"
NOME_MODELLO_SALVATO = "multi_bert_medico_full_shot"

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
    """Legge il file JSON e lo trasforma in un Dataset HuggingFace, preparando token di input
    attention_mask e labels."""
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

#Evaluation metrics
def compute_metrics_exact_match(p):
    """Calcola Precision, Recall e F1-Score"""
    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)
    print(f"predizione: {predictions}, etichetta corretta: {labels}")
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
def estrai_entita(seq_tags):
    """
    Data una lista di tag BIO (es. ['O', 'B-CLINENTITY', 'I-CLINENTITY', 'O']),
    restituisce un set di tuple che rappresentano le entità trovate.
    Formato tupla: (tipo_entita, indice_inizio, indice_fine)
    """
    entita = set()
    tipo_corrente = None
    inizio_corrente = -1

    for i, tag in enumerate(seq_tags):
        if tag == 'O':
            if tipo_corrente is not None:
                # Chiude un'entità precedente
                entita.add((tipo_corrente, inizio_corrente, i - 1))
                tipo_corrente = None
                
        elif tag.startswith('B-'):
            if tipo_corrente is not None:
                # Chiude un'entità adiacente prima di aprirne una nuova
                entita.add((tipo_corrente, inizio_corrente, i - 1))
            tipo_corrente = tag[2:] # Prende solo 'CLINENTITY'
            inizio_corrente = i
            
        elif tag.startswith('I-'):
            if tipo_corrente is None:
                # Caso limite: una I- senza una B- precedente. 
                # ovvero l'inizio di una nuova entità.
                tipo_corrente = tag[2:]
                inizio_corrente = i
            elif tipo_corrente != tag[2:]:
                # Transizione I- di un tipo diverso
                entita.add((tipo_corrente, inizio_corrente, i - 1))
                tipo_corrente = tag[2:]
                inizio_corrente = i

    # Se l'array finisce mentre un'entità era ancora aperta
    if tipo_corrente is not None:
        entita.add((tipo_corrente, inizio_corrente, len(seq_tags) - 1))

    return entita

# effettua exact e partial match -> Lenient Evaluation (Valutazione Indulgente) o Relaxed Match (Corrispondenza Rilassata).
def compute_metrics_exact_partial_match(p):
    """Calcolo manuale di Precision, Recall, F1 con logica Exact/Partial Match (SemEval-style)"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # si passa da lista di id (o->O, 1->B, 2->I) a lista di stringhe (O, B-CLINENTITY, I-CLINENTITY)
    # 1. Pulizia dai token speciali (-100) assegnati da pytorch, sono token di padding per rendere le sequenze  passate a BERT di lunghezza uguale
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]


    exact_matches = 0
    partial_matches = 0
    total_true_entities = 0
    total_pred_entities = 0

    # 2. Calcolo dei Match (Exact vs Partial)
    for y_true, y_pred in zip(true_labels, true_predictions):
        ent_vere = estrai_entita(y_true)
        ent_pred = estrai_entita(y_pred)
        
        total_true_entities += len(ent_vere)
        total_pred_entities += len(ent_pred)
        
        # Teniamo traccia delle predizioni già "accoppiate" per non contarle due volte
        predizioni_usate = set()
        
        for vera in ent_vere:
            tipo_v, inizio_v, fine_v = vera
            
            for pred in ent_pred:
                if pred in predizioni_usate:
                    continue
                    
                tipo_p, inizio_p, fine_p = pred
                
                # CONDIZIONE DI OVERLAP: 
                # Hanno lo stesso tipo E i loro confini si intersecano/sovrappongono
                if tipo_v == tipo_p and max(inizio_v, inizio_p) <= min(fine_v, fine_p):
                    
                    if inizio_v == inizio_p and fine_v == fine_p:
                        exact_matches += 1
                    else:
                        partial_matches += 1
                        
                    predizioni_usate.add(pred)
                    break 

    
    tp_score = exact_matches + (0.5 * partial_matches)
    
    # Precision
    precision = tp_score / total_pred_entities if total_pred_entities > 0 else 0.0
    
    # Recall
    recall = tp_score / total_true_entities if total_true_entities > 0 else 0.0
    
    # F1 Score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_matches_count": exact_matches,      
        "partial_matches_count": partial_matches   
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


    #PREPARAZIONE DATI E MODELLO
    print(f"Preparazione dei dati per l'esperimento: {FILE_TRAIN}...")
    train_dataset,_ = make_dataset(FILE_TRAIN)
    eval_dataset,_ = make_dataset(FILE_EVAL)

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
        save_only_model=True,   
        metric_for_best_model="loss",
        load_best_model_at_end=True,
        greater_is_better=True,
        save_total_limit=1,
        report_to="wandb",
        logging_strategy="epoch",
        lr_scheduler_type="linear"
    )

    trainer = Trainer(
        model=modello,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_exact_partial_match,
    )

    
    print("\n Inizio Training ")
    trainer.train()
    trainer.save_model(f"model/{NOME_MODELLO_SALVATO}")
    print(f"Modello salvato in: model/{NOME_MODELLO_SALVATO}")
    trainer.state.save_to_json(f"model/{NOME_MODELLO_SALVATO}/trainer_state.json")







    print("Generazione della matrice di confusione")
    
    risultati_eval = trainer.predict(eval_dataset)
    logits = risultati_eval.predictions 
    labels_vere = risultati_eval.label_ids
    
    
    preds_ids = np.argmax(logits, axis=2)
    
    #togliamo i -100 dalla lista di etichette vere
    y_true = []
    y_pred = []
    
    for seq_vera, seq_pred in zip(labels_vere, preds_ids):
        for vera, pred in zip(seq_vera, seq_pred):
            if vera != -100:
                y_true.append(vera) # lista di etichette vere
                y_pred.append(pred) # lista di etichette predette
                
    
    wandb.log({
        "matrice_di_confusione_finale": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=label_list
        )
    })
    

    wandb.finish()