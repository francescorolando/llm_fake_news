from datasets import load_dataset
from transformers import AutoTokenizer
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    MODEL,
    DATASET,
    DATASET_CONFIG,
    MAX_LENGTH,
    TEXT_COL,
    MAX_TRAIN_SAMPLES,
    MAX_EVAL_SAMPLES,
)


def load_data():
    # carichiamo il dataset usando Hugging Face Datasets
    dataset = load_dataset(DATASET, DATASET_CONFIG)
    print(dataset)  # stampiamo le informazioni sul dataset per verifica

    # carichiamo il tokenizer pre-addestrato di DistilBERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # tokenizziamo il dataset
    def tokenize_function(examples):
        return tokenizer(
            examples[TEXT_COL],
            truncation=True,
            padding=False,
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["Unnamed: 0", "title", TEXT_COL],
    )

    # il trainer si aspetta "labels"
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # campionamento per velocizzare il training
    tokenized_dataset["train"] = tokenized_dataset["train"].select(
        range(MAX_TRAIN_SAMPLES)
    )
    tokenized_dataset["validation"] = tokenized_dataset["validation"].select(
        range(MAX_EVAL_SAMPLES)
    )

    return tokenized_dataset, tokenizer


if __name__ == "__main__":
    tokenized_dataset, tokenizer = load_data()
    print(
        tokenized_dataset["train"][:5]
    )  # stampiamo i primi 5 esempi tokenizzati per verifica
    print(tokenizer)  # stampiamo il tokenizer per verifica
