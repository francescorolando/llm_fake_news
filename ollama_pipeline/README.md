# Parte 2 — Pipeline Ollama

## Task scelto e motivazione

Task: classificazione binaria di notizie vere vs false (FAKE / REAL).

È stato scelto lo stesso task della Parte 1 (fine-tuning di DistilBERT su
GonzaloA/fake_news) per rendere possibile un confronto diretto tra le due
strategie: fine-tuning di un encoder-only vs prompting di un LLM locale
senza training specifico.

## Modello usato

**gemma2:2b** via Ollama (esecuzione locale).

Scelto per il buon bilanciamento tra capacità e velocità su hardware consumer,
e perché disponibile localmente senza dipendenze da API esterne.

## Dataset

Stesso dataset della Parte 1: **GonzaloA/fake_news** (Hugging Face).

Sono stati estratti 20 campioni bilanciati dallo split di validation
(10 FAKE, 10 REAL), usando lo stesso seed (42) del collega per coerenza.
I campioni sono salvati in `data/samples.json`.

## Strategie di prompting usate

### Zero-shot

Il modello riceve solo l'istruzione e il testo dell'articolo, senza esempi.

```
Classify the following news article as FAKE or REAL.
Answer with exactly one word: FAKE or REAL.
Do not add any explanation.

Article: {testo}
Label:
```

### Few-shot

Il modello riceve l'istruzione seguita da 4 esempi etichettati (2 FAKE, 2 REAL)
prima dell'articolo da classificare.

```
Classify the following news article as FAKE or REAL.
Answer with exactly one word: FAKE or REAL.
Do not add any explanation.

Article: {esempio 1}
Label: FAKE

Article: {esempio 2}
Label: REAL

... (4 esempi totali)

Article: {testo}
Label:
```

## Struttura della pipeline

```
samples.json
     │
     ▼
pipeline.py
     ├── zero_shot_prompt(text)  ──►  prompts.py
     ├── few_shot_prompt(text)   ──►  prompts.py
     │
     ▼
POST http://localhost:11434/api/generate  (Ollama API)
     │
     ▼
parse_label()   — normalizza l'output del modello
     │
     ▼
results/outputs.json
     │
     ▼
comparator.py  — calcola e stampa le metriche
```

## Risultati

| Strategia | Accuracy globale | Accuracy FAKE | Accuracy REAL |
| --------- | ---------------- | ------------- | ------------- |
| Zero-shot | **0.95** (19/20) | 1.00          | 0.90          |
| Few-shot  | 0.90 (18/20)     | 0.90          | 0.90          |

Delta few-shot − zero-shot: **−0.05**

### Confronto con la Parte 1

| Approccio                         | Accuracy |
| --------------------------------- | -------- |
| Zero-shot (gemma2:2b)             | 0.95     |
| Few-shot (gemma2:2b)              | 0.90     |
| Fine-tuning DistilBERT — baseline | 0.978    |
| Fine-tuning DistilBERT — best     | **0.98** |

## Esempio di input/output

**Input** (id 1, GT = FAKE):

> History is once again being made thanks to President Obama. On Wednesday,
> Obama nominated Abid Riaz Qureshi to serve as...

| Strategia | Output raw | Label parsata | Corretto? |
| --------- | ---------- | ------------- | --------- |
| Zero-shot | `FAKE`     | FAKE          | ✓         |
| Few-shot  | `REAL`     | REAL          | ✗         |

**Input** (id 14, GT = REAL):

> hillary clinton campaign still whining about the fbi november — the hillary
> clinton whineaton continues after having m...

| Strategia | Output raw | Label parsata | Corretto? |
| --------- | ---------- | ------------- | --------- |
| Zero-shot | `FAKE`     | FAKE          | ✗         |
| Few-shot  | `FAKE`     | FAKE          | ✗         |

## Considerazioni sui risultati

**Zero-shot supera few-shot (0.95 vs 0.90).**
Risultato controintuitivo ma spiegabile: i 4 esempi usati nel few-shot sono
costruiti su notizie false stilisticamente ovvie (titoli sensazionalistici,
teorie del complotto esplicite). Quando il modello ha incontrato un articolo
dal tono istituzionale ma falso (id 1), gli esempi l'hanno fuorviato
classificandolo REAL per analogia stilistica.

**L'errore condiviso (id 14)** mostra un limite strutturale del prompting:
articoli reali scritti con linguaggio aggressivo o partigiano vengono
classificati come fake da entrambe le strategie. Questo tipo di errore
richiederebbe esempi di addestramento specifici per essere corretto.

**Il fine-tuning rimane superiore (0.98 vs 0.95)**, ma il gap è contenuto
considerando che gemma2:2b non ha ricevuto nessun training specifico sul task.
Il prompting risulta quindi una baseline solida e immediata, utile soprattutto
quando non si dispone di dati etichettati sufficienti per il fine-tuning.
