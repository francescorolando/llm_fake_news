# llm_fake_news

Fake news classification using two approaches: fine-tuning of DistilBERT and a local LLM inference pipeline via Ollama with zero-shot and few-shot prompting strategies.

Built for the LLM Course 2026 — University of Turin.

**Authors:** Francesco Screti & Francesco Rolando

---

## Project Structure

```
llm_fake_news/
├── README.md
├── fine_tuning/          # parte 1 — fine-tuning DistilBERT
└── ollama_pipeline/      # parte 2 — pipeline Ollama
```

---

## Parte 1 — Fine-tuning DistilBERT

### Modello

**distilbert-base-uncased** — encoder-only per classificazione di sequenze.

Scelto per la leggerezza (40% meno parametri rispetto a BERT), la velocità di training su hardware consumer e le performance competitive per task di classificazione testuale.

### Dataset

**GonzaloA/fake_news** (Hugging Face) — classificazione binaria: `0 = FAKE`, `1 = REAL`.

- Train: 1000 esempi (campionati, seed=42)
- Validation: 500 esempi (campionati, seed=42)
- Tokenizzazione: distilbert-base-uncased, max_length=256, padding dinamico

### Risultati

| Configurazione  | LR   | Batch | Epochs | Accuracy   |
| --------------- | ---- | ----- | ------ | ---------- |
| baseline        | 2e-5 | 16    | 3      | 0.9780     |
| high_lr         | 5e-5 | 16    | 3      | 0.9740     |
| more_epochs     | 2e-5 | 16    | 5      | 0.9780     |
| **small_batch** | 2e-5 | **8** | 3      | **0.9800** |

La configurazione migliore è **small_batch** (accuracy: 0.9800). Il batch size ridotto aumenta la frequenza degli aggiornamenti dei pesi, migliorando la generalizzazione. Il learning rate elevato (5e-5) degrada le performance, mentre aumentare le epoche non produce benefici apprezzabili.

---

## Parte 2 — Pipeline Ollama

### Modello

**gemma2:2b** via Ollama (inferenza locale, nessuna API esterna).

### Task

Stesso task della Parte 1 per rendere possibile un confronto diretto tra fine-tuning e prompting.

### Strategie di prompting

**Zero-shot** — solo istruzione e testo, nessun esempio.

**Few-shot** — istruzione seguita da 4 esempi etichettati (2 FAKE, 2 REAL) prima del testo da classificare.

### Risultati

Valutazione su 20 campioni bilanciati (10 FAKE, 10 REAL) dallo split di validation.

| Strategia | Accuracy | Accuracy FAKE | Accuracy REAL |
| --------- | -------- | ------------- | ------------- |
| zero-shot | **0.95** | 1.00          | 0.90          |
| few-shot  | 0.90     | 0.90          | 0.90          |

### Confronto generale

| Approccio                         | Accuracy  |
| --------------------------------- | --------- |
| Zero-shot (gemma2:2b)             | 0.95      |
| Few-shot (gemma2:2b)              | 0.90      |
| Fine-tuning DistilBERT — baseline | 0.978     |
| Fine-tuning DistilBERT — best     | **0.980** |

Il fine-tuning supera il prompting, ma il gap è contenuto considerando che gemma2:2b non ha ricevuto alcun training specifico sul task. Il risultato zero-shot (0.95) è notevole e suggerisce che il prompting è una baseline solida quando non si dispone di dati etichettati sufficienti.

Il few-shot ha performato peggio del zero-shot perché gli esempi forniti erano stilisticamente stereotipati. Un articolo dal tono istituzionale ma falso è stato classificato come reale per analogia stilistica con gli esempi.

---

## Setup

### Requisiti

```bash
pip install -r fine_tuning/requirements.txt
pip install requests
```

### Ollama

Installa Ollama da [ollama.com](https://ollama.com), poi:

```bash
ollama pull gemma2:2b
```

### Esecuzione pipeline Ollama

```bash
cd ollama_pipeline

# esegui la pipeline (ollama deve essere in esecuzione)
ollama serve &
python pipeline.py 1    # salva results/outputs_run1.json
python pipeline.py 2    # salva results/outputs_run2.json
python pipeline.py 3    # salva results/outputs_run3.json

# analisi singola run
python comparator.py

# analisi multi-run con media e stabilità
python comparator.py --runs 3
```
