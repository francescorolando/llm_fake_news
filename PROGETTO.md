# Progetto finale — LLM Course 2026

**Autori:** Francesco Rolando & Francesco Screti  
**Repository:** github.com/francescorolando/llm_fake_news

---

## Parte 1 — Fine-tuning

### Modello scelto e motivazione

**distilbert-base-uncased** — versione distillata di BERT con il 40% di parametri in meno e il 97% delle performance originali. Scelto per la leggerezza (66.9M parametri), la velocità di training su hardware consumer e le performance competitive per task di classificazione testuale. L'architettura encoder-only è ideale per task di classificazione in cui non è necessaria la generazione di testo.

### Dataset scelto e motivazione

**GonzaloA/fake\_news** (Hugging Face) — dataset di articoli in inglese con etichette binarie: `0 = FAKE`, `1 = REAL`. Scelto per la disponibilità pubblica, il bilanciamento delle classi e la rilevanza pratica del task. Sono stati usati 1000 esempi per il training e 500 per la validation, campionati con seed=42 per riproducibilità.

### Risultati ottenuti

| Configurazione | LR | Batch | Epochs | Accuracy |
|---|---|---|---|---|
| baseline | 2e-5 | 16 | 3 | 0.9740 |
| high\_lr | 5e-5 | 16 | 3 | 0.9740 |
| more\_epochs | 2e-5 | 16 | 5 | 0.9720 |
| **small\_batch** | 2e-5 | **8** | 3 | **0.9800** |

La configurazione migliore è **small\_batch** con accuracy 0.9800. Il batch size ridotto aumenta la frequenza degli aggiornamenti dei pesi, migliorando la generalizzazione. Il learning rate elevato non porta benefici, e aumentare le epoche non migliora le performance — il modello converge già alla seconda epoca.

### Difficoltà incontrate

- Gestione dei checkpoint: i file del training pesano circa 3GB e non possono essere versionati su git; è stato necessario escluderli tramite `.gitignore` e documentare come rigenerarli
- Compatibilità tra versioni di librerie: alcune versioni di `transformers` e `torch` disponibili su conda-forge erano troppo datate rispetto ai requisiti del progetto, risolta usando pip
- Conflitto tra ambienti virtuali: Copilot ha creato automaticamente un ambiente `.venv` locale nella cartella del progetto senza dichiararlo esplicitamente, causando conflitti con l'ambiente conda `llmcourse` usato per il corso
- I path relativi negli script cambiavano comportamento a seconda della cartella da cui veniva lanciato il training; risolto usando `os.path.dirname(__file__)` in `config.py`

---

## Parte 2 — Pipeline Ollama

### Task scelto e motivazione

**Classificazione binaria di fake news** — stesso task della Parte 1, scelto deliberatamente per rendere possibile un confronto diretto tra fine-tuning e prompting sullo stesso problema. Questo permette di valutare quanto un LLM generalista senza training specifico si avvicini alle performance di un modello fine-tuned.

### Strategie di prompting usate

**Zero-shot** — il modello riceve solo l'istruzione e il testo da classificare, senza esempi. Il prompt richiede esplicitamente una risposta di una sola parola (FAKE o REAL).

**Few-shot** — il modello riceve l'istruzione seguita da 4 esempi etichettati (2 FAKE, 2 REAL) prima del testo da classificare. Gli esempi sono stati scelti per coprire stili diversi di notizie false e reali.

Il confronto tra le due strategie sullo stesso input è implementato in `pipeline.py` e analizzato in `comparator.py`.

### Esempio di input/output

**Input:**
> "History is once again being made thanks to President Obama. On Wednesday, Obama nominated Abid Riaz Qureshi to serve as federal judge, making him the first Muslim ever nominated for a federal judgeship in United States history."

| Strategia | Output raw | Label | Corretto |
|---|---|---|---|
| Zero-shot | `FAKE` | FAKE | ✓ |
| Few-shot | `REAL` | REAL | ✗ |

Ground truth: **FAKE**

**Input:**
> "hillary clinton campaign still whining about the fbi november — the hillary clinton whineaton continues after having mounted..."

| Strategia | Output raw | Label | Corretto |
|---|---|---|---|
| Zero-shot | `FAKE` | FAKE | ✗ |
| Few-shot | `FAKE` | FAKE | ✗ |

Ground truth: **REAL**

### Considerazioni sui risultati

| Strategia | Accuracy | Accuracy FAKE | Accuracy REAL |
|---|---|---|---|
| Zero-shot | **0.950** | 1.00 | 0.90 |
| Few-shot | 0.900 | 0.90 | 0.90 |

**Zero-shot supera few-shot (0.95 vs 0.90).** Risultato controintuitivo ma spiegabile: gli esempi del few-shot erano stilisticamente stereotipati (titoli sensazionalistici, teorie del complotto esplicite). Un articolo dal tono istituzionale ma falso è stato classificato come reale per analogia stilistica con gli esempi forniti.

**L'errore condiviso** (articolo Clinton) mostra un limite strutturale del prompting: articoli reali scritti con linguaggio aggressivo o partigiano vengono classificati come fake da entrambe le strategie. Questo tipo di errore richiederebbe esempi di addestramento specifici.

**Confronto con la Parte 1:**

| Approccio | Accuracy |
|---|---|
| Zero-shot (gemma2:2b) | 0.950 |
| Few-shot (gemma2:2b) | 0.900 |
| DistilBERT fine-tuned | **0.980** |

Il fine-tuning rimane superiore, ma il gap è contenuto considerando che gemma2:2b non ha ricevuto alcun training specifico sul task. Il prompting è una baseline solida quando non si dispone di dati etichettati.

---

## Presentazione finale

### 1. Introduzione al progetto

**Task scelto:** classificazione binaria di fake news (FAKE/REAL). Scelto per la rilevanza pratica, la disponibilità di dataset pubblici e la possibilità di confrontare i due approcci sullo stesso problema.

**Dataset:** GonzaloA/fake\_news (Hugging Face). Scelto per il bilanciamento delle classi, la qualità delle etichette e le dimensioni adeguate per un fine-tuning su hardware consumer.

**Modelli:**
- Parte 1: distilbert-base-uncased — encoder-only leggero e performante per classificazione
- Parte 2: gemma2:2b via Ollama — LLM locale senza dipendenze da API esterne

### 2. Approccio e sviluppo

**Come abbiamo affrontato il problema:**
- Parte 1: training con 4 configurazioni di iperparametri per analizzare l'impatto di learning rate, batch size e numero di epoche
- Parte 2: implementazione di una pipeline modulare con zero-shot e few-shot, con salvataggio dei risultati per confronto multi-run

**Difficoltà teoriche:**
- Comprendere perché batch size ridotto migliora la generalizzazione
- Interpretare il comportamento controintuitivo del few-shot (peggiore del zero-shot)
- Capire il trade-off tra fine-tuning e prompting in termini di risorse e performance

**Difficoltà pratiche:**
- Conflitto tra ambiente conda e venv creato automaticamente da Copilot
- Checkpoint da 3GB non versionabili su git
- Path relativi che cambiavano comportamento a seconda della cartella di lancio
- Ollama non risponde sempre con una sola parola: necessaria normalizzazione dell'output con `parse_label()`

**Come le abbiamo risolte:**
- Ambiente: eliminato il venv, usato esclusivamente conda `llmcourse`
- Checkpoint: esclusi da git, documentata la procedura per rigenerarli
- Path: usato `os.path.dirname(__file__)` ovunque
- Output Ollama: funzione `parse_label()` che cerca le sottostringhe FAKE/REAL nel testo normalizzato

**Uso degli LLM come supporto:**

*Claude* è stato usato per la progettazione dell'architettura del progetto, la scrittura del codice della pipeline Ollama, il debug dei conflitti tra ambienti, le spiegazioni sui concetti (checkpoint, ambienti virtuali, git fork) e la scrittura del README e dell'app Streamlit.

*Copilot* è stato usato per il codice del training e la prima versione dell'app Streamlit per la Parte 1.

**Cosa ha funzionato bene:** Claude per ragionamento strutturato su problemi complessi e spiegazioni dettagliate; Copilot per generazione rapida di boilerplate.

**Cosa non ha funzionato:** Copilot ha creato un ambiente `.venv` locale nella cartella del progetto senza dichiararlo, causando conflitti con l'ambiente conda del corso. Inoltre alcune spiegazioni di Claude sui path relativi in Python erano inizialmente imprecise e hanno richiesto correzioni manuali.

**Qualcosa spiegato in modo sbagliato o fuorviante:** nella prima versione del codice generato, il plot delle loss era costruito filtrando separatamente gli step di training e di evaluation — array di lunghezze diverse plottati insieme, producendo un grafico scorretto. Il problema è stato identificato manualmente revisionando il codice.

### 3. Risultati

**Plot delle loss:** disponibili nell'app Streamlit, sezione Fine-tuning. Per ogni configurazione viene mostrato l'andamento di training loss e validation loss per step, letti dai file `trainer_state.json` nei checkpoint.

**Esempi concreti di input/output DistilBERT:**

Input: *"Breaking: new study links 5G towers to memory loss in laboratory rats, scientists demand immediate shutdown."*
→ Predizione: **FAKE** (confidenza: 0.96)

Input: *"The European Central Bank held interest rates steady on Thursday, citing easing inflation and stable growth projections."*
→ Predizione: **REAL** (confidenza: 0.94)

**Schema pipeline Ollama:**
```
samples.json
     ↓
pipeline.py
     ├── zero_shot_prompt(text) → POST /api/generate → parse_label() → FAKE/REAL
     └── few_shot_prompt(text)  → POST /api/generate → parse_label() → FAKE/REAL
     ↓
results/outputs_run{N}.json
     ↓
comparator.py → metriche, stabilità multi-run, disaccordi
```

**Confronto strategie di prompting:**

| Strategia | Accuracy | FAKE | REAL |
|---|---|---|---|
| Zero-shot | **0.950** | 1.00 | 0.90 |
| Few-shot | 0.900 | 0.90 | 0.90 |

Zero-shot supera few-shot. La qualità degli esempi nel few-shot è determinante: esempi stereotipati introducono bias stilistici che penalizzano articoli ambigui.
