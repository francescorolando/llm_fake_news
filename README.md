# llm_fake_news

Fake news classification using two approaches: fine-tuning of DistilBERT and a local LLM inference pipeline via Ollama with zero-shot and few-shot prompting strategies.

Built for the LLM Course 2026 — University of Turin.

**Authors:** Francesco Rolando & Francesco Screti

---

## Project Structure

```
llm_fake_news/
├── README.md
├── PROGETTO.md
├── app.py                    # Streamlit dashboard + live demo
├── fine_tuning/              # Part 1 — DistilBERT fine-tuning
│   ├── config.py
│   ├── train.py
│   ├── data/
│   │   └── dataset.py
│   ├── model/
│   │   └── model.py
│   ├── training/
│   │   └── metrics.py
│   └── results/              # checkpoints (not versioned)
└── ollama_pipeline/          # Part 2 — Ollama pipeline
    ├── prompts.py
    ├── pipeline.py
    ├── comparator.py
    ├── data/
    │   └── samples.json
    └── results/
        └── outputs_run*.json
```

---

## Setup

### Dependencies

```bash
pip install -r fine_tuning/requirements.txt
pip install requests streamlit plotly scikit-learn matplotlib
```

### Ollama

Install Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull gemma2:2b
```

### DistilBERT Checkpoints

Checkpoints are not versioned due to size (~3GB). Regenerate by running:

```bash
cd fine_tuning
python train.py
```

Training takes approximately 90 minutes. Without checkpoints, the app works but the fine-tuning section and DistilBERT live demo will not be available.

---

## Usage

### Ollama Pipeline

```bash
# in a separate terminal
ollama serve

# run the pipeline (repeat for multiple runs)
cd ollama_pipeline
python pipeline.py

# analyze results
python comparator.py
python comparator.py --runs 3
```

### Streamlit App

```bash
streamlit run app.py
```

---

## Results

| Approach                            | Accuracy  |
| ----------------------------------- | --------- |
| DistilBERT fine-tuned (small_batch) | **0.980** |
| Ollama zero-shot (gemma2:2b)        | 0.950     |
| Ollama few-shot (gemma2:2b)         | 0.900     |
