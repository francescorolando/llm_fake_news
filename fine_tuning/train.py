# train.py
# Entry point del progetto: addestra il modello con 4 configurazioni diverse
# e mostra un riepilogo finale con le accuracy di ognuna.
# Esegui con: python3 train.py

import os
import sys
import torch
import random
import numpy as np
from config import SEED, CONFIGS, RESULTS_DIR       # parametri globali
from data.dataset import load_data                   # carica e tokenizza il dataset
from model.model import load_model                   # carica DistilBERT
from training.metrics import compute_metrics         # calcola l'accuracy
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

def set_seed(seed: int):
    """Fissa il seed su tutte le librerie per risultati riproducibili."""
    random.seed(seed)          # seed per Python standard
    np.random.seed(seed)       # seed per numpy
    torch.manual_seed(seed)    # seed per PyTorch (pesi e dropout)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   # seed per GPU se disponibile


def train_single_config(config: dict, tokenized_dataset, tokenizer):
    """
    Addestra il modello con una singola configurazione di iperparametri.
    Viene chiamata 4 volte, una per ogni configurazione in CONFIGS.
    """
    print(f"\n{'='*55}")
    print(f"  Configurazione: {config['name']}")
    print(f"  lr={config['lr']}  batch={config['batch']}  epochs={config['epochs']}  wd={config['wd']}")
    print(f"{'='*55}")

    model = load_model()   # carica un modello fresco per ogni configurazione

    # crea una cartella separata per ogni configurazione
    # es. results/baseline, results/high_lr, ecc.
    run_dir = os.path.join(RESULTS_DIR, config["name"])
    os.makedirs(run_dir, exist_ok=True)

    # configura gli iperparametri presi dal dizionario config
    training_args = TrainingArguments(
        output_dir=run_dir,                              # dove salvare i checkpoint
        num_train_epochs=config["epochs"],               # numero di epoche
        per_device_train_batch_size=config["batch"],     # batch size training
        per_device_eval_batch_size=32,                   # batch size valutazione
        learning_rate=config["lr"],                      # learning rate
        weight_decay=config["wd"],                       # regolarizzazione
        eval_strategy="epoch",                           # valuta alla fine di ogni epoca
        save_strategy="epoch",                           # salva alla fine di ogni epoca
        save_total_limit=1,                              # tiene solo l'ultimo checkpoint
        load_best_model_at_end=True,                     # carica il modello migliore alla fine
        logging_steps=50,                                # stampa la loss ogni 50 step
        seed=SEED,                                       # seed per riproducibilità
    )

    # padding dinamico: porta ogni batch alla lunghezza della frase più lunga
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # assembla tutti gli ingredienti nel Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],        # 1000 articoli di training
        eval_dataset=tokenized_dataset["validation"],    # 500 articoli di validation
        compute_metrics=compute_metrics,                 # calcola accuracy ad ogni epoca
        data_collator=data_collator,
    )

    trainer.train()                                      # avvia il training
    risultati = trainer.evaluate()                       # valuta sul validation set
    accuracy = risultati["eval_accuracy"]
    print(f"Accuracy {config['name']}: {accuracy:.4f}")
    return accuracy                                      # restituisce l'accuracy per il riepilogo


def main():
    set_seed(SEED)
    print(f"Seed: {SEED}")

    # carica e tokenizza il dataset una volta sola
    # le 4 configurazioni usano gli stessi dati
    print("\n--- Caricamento dataset ---")
    tokenized_dataset, tokenizer = load_data()
    print(f"Train:      {len(tokenized_dataset['train'])} esempi")
    print(f"Validation: {len(tokenized_dataset['validation'])} esempi")

    os.makedirs(RESULTS_DIR, exist_ok=True)   # crea la cartella results se non esiste

    # scorre le 4 configurazioni e addestra un modello per ognuna
    riepilogo = []
    for config in CONFIGS:
        accuracy = train_single_config(config, tokenized_dataset, tokenizer)
        riepilogo.append((config["name"], accuracy))   # salva nome e accuracy

    # stampa il riepilogo finale con tutte le accuracy
    print("\n" + "="*55)
    print("  RIEPILOGO CONFIGURAZIONI")
    print("="*55)
    for nome, acc in riepilogo:
        print(f"  {nome:<20} accuracy: {acc:.4f}")

    # trova e stampa la configurazione migliore
    migliore = max(riepilogo, key=lambda x: x[1])
    print(f"\n  Migliore: {migliore[0]} ({migliore[1]:.4f})")


if __name__ == "__main__":
    main()

    