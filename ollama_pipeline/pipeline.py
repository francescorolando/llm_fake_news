"""
pipeline principale: carica i sample, interroga ollama con
zero-shot e few-shot, salva i risultati

uso:
    python pipeline.py        → salva in results/outputs_run1.json
    python pipeline.py 2      → salva in results/outputs_run2.json
"""

import json
import os
import sys
import requests

from prompts import zero_shot_prompt, few_shot_prompt

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:2b"

SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "data", "samples.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def query_ollama(prompt: str, model: str = MODEL) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def parse_label(text: str) -> str:
    """normalizza l'output del modello — ollama non risponde sempre con una parola sola"""
    upper = text.strip().upper()
    if "FAKE" in upper:
        return "FAKE"
    if "REAL" in upper:
        return "REAL"
    return "UNKNOWN"


def run_pipeline(run_id: int = 1):
    output_path = os.path.join(RESULTS_DIR, f"outputs_run{run_id}.json")

    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for i, sample in enumerate(samples):
        text = sample["text"]
        ground_truth = sample["label_str"]

        print(f"[{i+1}/{len(samples)}] ground truth: {ground_truth}")

        zs_raw = query_ollama(zero_shot_prompt(text))
        fs_raw = query_ollama(few_shot_prompt(text))

        zs_label = parse_label(zs_raw)
        fs_label = parse_label(fs_raw)

        print(f"  zero-shot → {zs_label}  |  few-shot → {fs_label}")

        results.append(
            {
                "id": i,
                "text_preview": text[:200],
                "ground_truth": ground_truth,
                "zero_shot": {
                    "raw": zs_raw,
                    "label": zs_label,
                    "correct": zs_label == ground_truth,
                },
                "few_shot": {
                    "raw": fs_raw,
                    "label": fs_label,
                    "correct": fs_label == ground_truth,
                },
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nrisultati salvati in {output_path}")
    return results


if __name__ == "__main__":
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_pipeline(run_id)
