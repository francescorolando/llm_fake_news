"""
costruzione dei prompt per il task di classificazione fake news
"""

# esempi fissi per il few-shot — presi fuori dal validation set
# per non contaminare la valutazione
FEW_SHOT_EXAMPLES = [
    (
        "Breaking: Scientists confirm that drinking bleach cures cancer, "
        "government hiding the truth for decades.",
        "FAKE",
    ),
    (
        "The Federal Reserve raised interest rates by 0.25 percentage points "
        "on Wednesday, citing persistent inflation concerns.",
        "REAL",
    ),
    (
        "NASA admits moon landing was filmed in a Hollywood studio, "
        "leaked documents reveal.",
        "FAKE",
    ),
    (
        "A new study published in The Lancet links sedentary lifestyle "
        "to increased risk of cardiovascular disease.",
        "REAL",
    ),
]

_INSTRUCTION = (
    "Classify the following news article as FAKE or REAL.\n"
    "Answer with exactly one word: FAKE or REAL.\n"
    "Do not add any explanation."
)


def zero_shot_prompt(text: str) -> str:
    return f"{_INSTRUCTION}\n\nArticle: {text}\nLabel:"


def few_shot_prompt(text: str) -> str:
    examples = "\n\n".join(
        f"Article: {article}\nLabel: {label}" for article, label in FEW_SHOT_EXAMPLES
    )
    return f"{_INSTRUCTION}\n\n{examples}\n\nArticle: {text}\nLabel:"
