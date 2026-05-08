"""
comparatore: legge outputs.json e stampa metriche
zero-shot vs few-shot, incluse accuracy per classe
"""

import json
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results", "outputs.json")


def compute_metrics(results: list, strategy: str) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r[strategy]["correct"])

    # accuracy per classe
    fake = [r for r in results if r["ground_truth"] == "FAKE"]
    real = [r for r in results if r["ground_truth"] == "REAL"]

    fake_correct = sum(1 for r in fake if r[strategy]["correct"])
    real_correct = sum(1 for r in real if r[strategy]["correct"])

    unknown = sum(1 for r in results if r[strategy]["label"] == "UNKNOWN")

    return {
        "accuracy":       correct / total,
        "acc_fake":       fake_correct / len(fake) if fake else 0,
        "acc_real":       real_correct / len(real) if real else 0,
        "unknown_count":  unknown,
        "correct":        correct,
        "total":          total,
    }


def print_metrics(label: str, m: dict):
    print(f"\n{'─' * 40}")
    print(f"  {label}")
    print(f"{'─' * 40}")
    print(f"  accuracy globale : {m['accuracy']:.2f}  ({m['correct']}/{m['total']})")
    print(f"  accuracy FAKE    : {m['acc_fake']:.2f}")
    print(f"  accuracy REAL    : {m['acc_real']:.2f}")
    if m["unknown_count"] > 0:
        print(f"  ⚠ output non parsabili: {m['unknown_count']}")


def print_errors(results: list, strategy: str):
    errors = [r for r in results if not r[strategy]["correct"]]
    if not errors:
        print("  nessun errore")
        return
    for r in errors:
        print(f"  [id {r['id']}] GT={r['ground_truth']} → {r[strategy]['label']}")
        print(f"    testo: {r['text_preview'][:120]}...")


def compare():
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    zs = compute_metrics(results, "zero_shot")
    fs = compute_metrics(results, "few_shot")

    print_metrics("zero-shot", zs)
    print_metrics("few-shot", fs)

    # delta
    delta = fs["accuracy"] - zs["accuracy"]
    direction = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
    print(f"\n  delta (few-shot - zero-shot): {delta:+.2f} {direction}")

    # errori per strategia
    print("\n\nerrori zero-shot:")
    print_errors(results, "zero_shot")

    print("\nerrori few-shot:")
    print_errors(results, "few_shot")

    # casi di disaccordo tra le due strategie
    disagreements = [
        r for r in results
        if r["zero_shot"]["label"] != r["few_shot"]["label"]
    ]
    print(f"\ncasi in cui le due strategie divergono: {len(disagreements)}")
    for r in disagreements:
        zs_l = r["zero_shot"]["label"]
        fs_l = r["few_shot"]["label"]
        gt = r["ground_truth"]
        winner = "few-shot" if fs_l == gt else ("zero-shot" if zs_l == gt else "nessuno")
        print(f"  [id {r['id']}] ZS={zs_l} | FS={fs_l} | GT={gt} → corretto: {winner}")
        print(f"    {r['text_preview'][:120]}...")


if __name__ == "__main__":
    compare()
