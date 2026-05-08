# Sommario del training

## 1. Obiettivo

Hai eseguito un fine-tuning di `distilbert-base-uncased` su un task di classificazione binaria di notizie vere vs false.

## 2. Dataset

- Dataset usato: `GonzaloA/fake_news` da Hugging Face
- Task: classificare `0 = FAKE` e `1 = REAL`
- Split usati:
  - `train`: campionato a 1000 esempi
  - `validation`: campionato a 500 esempi
- Colonne tokenizzate: `text` → input, `label` → `labels`
- Tokenizzazione:
  - modello: `distilbert-base-uncased`
  - truncation attivata
  - padding dinamico gestito da `DataCollatorWithPadding`
  - `max_length = 256`

## 3. Modello

- Modello: `distilbert-base-uncased`
- Tipo: encoder-only per classificazione di sequenze
- Numero di label: `2`
- Viene caricato da `model/model.py` tramite `AutoModelForSequenceClassification`
- Ogni configurazione parte da un modello fresco (`load_model()` viene richiamato 4 volte)

## 4. Architettura del training

- Script principale: `train.py`
- Pipeline:
  1. `set_seed(SEED)` per riproducibilità (seed = 42)
  2. `load_data()` per caricare e tokenizzare il dataset
  3. 4 configurazioni distinte di iperparametri
  4. per ogni configurazione:
     - costruzione di `TrainingArguments`
     - creazione di un `Trainer`
     - `trainer.train()`
     - valutazione su validation con `trainer.evaluate()`
- Metrica usata: accuracy, calcolata in `training/metrics.py`

## 5. Iperparametri testati

Le 4 configurazioni eseguite sono:

1. **baseline**
   - `lr = 2e-5`
   - `batch = 16`
   - `epochs = 3`
   - `wd = 0.01`

2. **high_lr**
   - `lr = 5e-5`
   - `batch = 16`
   - `epochs = 3`
   - `wd = 0.01`

3. **more_epochs**
   - `lr = 2e-5`
   - `batch = 16`
   - `epochs = 5`
   - `wd = 0.01`

4. **small_batch**
   - `lr = 2e-5`
   - `batch = 8`
   - `epochs = 3`
   - `wd = 0.01`

## 6. Risultati ottenuti

Il training ha prodotto i seguenti risultati di accuracy sul validation set per ciascuna configurazione:

- `baseline`: `0.9780`
- `high_lr`: `0.9740`
- `more_epochs`: `0.9780`
- `small_batch`: `0.9800`

### Configurazione migliore

- Migliore configurazione: `small_batch`
- Accuracy migliore: `0.9800`

## 7. Dove sono salvati i checkpoint

I checkpoint sono salvati in cartelle separate sotto `fine_tuning/results/`:

- `fine_tuning/results/baseline/`
- `fine_tuning/results/high_lr/`
- `fine_tuning/results/more_epochs/`
- `fine_tuning/results/small_batch/`

Ogni cartella contiene gli ultimi checkpoint e i file di stato del trainer.

## 8. File principali modificati / usati

- `config.py`: definisce modello, dataset, iperparametri e cartelle di output
- `data/dataset.py`: carica il dataset e costruisce il dataset tokenizzato
- `model/model.py`: carica DistilBERT per la classificazione
- `training/metrics.py`: calcola l'accuracy
- `train.py`: orchestri il training con 4 configurazioni

## 9. Note

- La valutazione finale è stata fatta su validation set, non sul test set.
- Il progetto è strutturato per confrontare l'effetto di learning rate, numero di epoche e batch size.
- La migliore configurazione è stata quella con batch size ridotto (`batch = 8`) e learning rate standard (`2e-5`).
