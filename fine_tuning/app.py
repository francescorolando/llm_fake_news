import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import os

# Page configuration
st.set_page_config(
    page_title="LLM Fine-Tuning Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .config-header {
        color: #1f77b4;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Project data
CONFIGS = [
    {
        "name": "baseline",
        "lr": 2e-5,
        "batch": 16,
        "epochs": 3,
        "wd": 0.01,
        "accuracy": 0.9780,
    },
    {
        "name": "high_lr",
        "lr": 5e-5,
        "batch": 16,
        "epochs": 3,
        "wd": 0.01,
        "accuracy": 0.9740,
    },
    {
        "name": "more_epochs",
        "lr": 2e-5,
        "batch": 16,
        "epochs": 5,
        "wd": 0.01,
        "accuracy": 0.9780,
    },
    {
        "name": "small_batch",
        "lr": 2e-5,
        "batch": 8,
        "epochs": 3,
        "wd": 0.01,
        "accuracy": 0.9800,
    },
]

# Sidebar navigation
st.sidebar.title("LLM Fine-Tuning Dashboard")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Model Architecture",
        "Dataset",
        "Hyperparameter Configurations",
        "Results and Analysis",
    ],
)

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "Overview":
    st.title("Fine-Tuning LLM: Project Overview")

    st.markdown("""
    This project implements fine-tuning of DistilBERT for binary classification 
    of news articles (fake vs. real).
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "DistilBERT", "base-uncased")

    with col2:
        st.metric("Task", "Binary Classification", "FAKE / REAL")

    with col3:
        st.metric("Best Accuracy", "0.9800", "small_batch")

    st.divider()

    st.subheader("Project Summary")

    summary_cols = st.columns(2)

    with summary_cols[0]:
        st.markdown("""
        #### Training Structure
        - Dataset: GonzaloA/fake_news from Hugging Face
        - Training split: 1000 examples
        - Validation split: 500 examples
        - Configurations tested: 4
        - Evaluation metric: Accuracy on validation set
        
        #### Configurations Tested
        1. Baseline: LR=2e-5, Batch=16, Epochs=3
        2. High LR: LR=5e-5, Batch=16, Epochs=3
        3. More Epochs: LR=2e-5, Batch=16, Epochs=5
        4. Small Batch: LR=2e-5, Batch=8, Epochs=3
        """)

    with summary_cols[1]:
        st.markdown("""
        #### Results Summary
        | Configuration | Accuracy |
        |---------------|----------|
        | baseline | 0.9780 |
        | high_lr | 0.9740 |
        | more_epochs | 0.9780 |
        | small_batch | 0.9800 |
        
        #### Key Findings
        - Reduced batch size improves generalization
        - Higher learning rate degrades performance
        - Additional epochs show no improvement
        """)

    st.divider()

    st.subheader("Project Structure")
    st.code("""
fine_tuning/
├── app.py                  # Streamlit dashboard
├── config.py              # Global configuration
├── train.py               # Training entry point
├── plot.py                # Visualization script
├── requirements.txt
├── data/
│   └── dataset.py         # Dataset loading
├── model/
│   └── model.py           # DistilBERT loading
├── training/
│   ├── metrics.py         # Accuracy computation
│   └── trainer.py
└── results/               # Checkpoints per config
    ├── baseline/
    ├── high_lr/
    ├── more_epochs/
    └── small_batch/
    """)

# ============================================================================
# PAGE: MODEL ARCHITECTURE
# ============================================================================
elif page == "Model Architecture":
    st.title("Model Architecture: DistilBERT")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Description
        
        DistilBERT is a distilled version of BERT that maintains 97% of BERT's 
        performance while reducing parameters by 40%.
        
        #### Key Characteristics:
        - Architecture: Encoder-only (similar to BERT)
        - Pre-training: General text corpus
        - Task: Sequence classification
        - Output: Logits for 2 classes (FAKE=0, REAL=1)
        
        #### Advantages:
        - 40% fewer parameters than BERT
        - 97% of BERT's performance
        - Ideal for fine-tuning with limited hardware
        - Faster training time
        """)

    with col2:
        st.metric("Model Name", "distilbert", "base-uncased")

    st.divider()

    st.subheader("Model Statistics")

    try:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Parameters", f"{total_params:,}")
        with col2:
            st.metric("Trainable Parameters", f"{trainable_params:,}")
        with col3:
            st.metric("Number of Classes", "2")
        with col4:
            st.metric("Max Sequence Length", "256 tokens")
    except:
        st.info("Model will be downloaded on first use.")

    st.divider()

    st.subheader("Fine-tuning Configuration")
    st.markdown("""
    - Loading: AutoModelForSequenceClassification.from_pretrained()
    - Number of classes: 2 (FAKE, REAL)
    - Tokenizer: distilbert-base-uncased tokenizer
    - Loss function: CrossEntropyLoss (automatic)
    - Optimizer: AdamW (default in Trainer)
    """)

# ============================================================================
# PAGE: DATASET
# ============================================================================
elif page == "Dataset":
    st.title("Dataset: GonzaloA/fake_news")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Dataset Overview
        
        GonzaloA/fake_news is a dataset from Hugging Face for binary 
        classification of news articles.
        
        #### Task:
        - Classify articles as FAKE (0) or REAL (1)
        - 2 balanced classes
        """)

    with col2:
        st.metric("Dataset", "GonzaloA/", "fake_news")

    st.divider()

    st.subheader("Data Splits and Sampling")

    dataset_stats = pd.DataFrame(
        {
            "Split": [
                "Train (full)",
                "Train (used)",
                "Validation (full)",
                "Validation (used)",
            ],
            "Number of Examples": ["N/A", "1000", "N/A", "500"],
            "Usage": ["Reference", "Fine-tuning", "Reference", "Evaluation"],
        }
    )

    st.dataframe(dataset_stats, use_container_width=True)

    st.divider()

    st.subheader("Text Processing")

    process_cols = st.columns(3)
    with process_cols[0]:
        st.metric("Input Column", "text")
    with process_cols[1]:
        st.metric("Label Column", "label")
    with process_cols[2]:
        st.metric("Max Length", "256 tokens")

    st.markdown("""
    #### Tokenization Pipeline:
    1. Tokenizer: distilbert-base-uncased
    2. Truncation: Enabled (max 256 tokens)
    3. Padding: Dynamic via DataCollatorWithPadding
    4. Removed columns: Unnamed: 0, title
    5. Label mapping: label -> labels (required by Trainer)
    """)

    st.divider()

    st.subheader("Example Data Entry")

    example = {
        "text": "Breaking news: researchers discover new efficient AI training method...",
        "label": 1,
        "input_ids": "[101, 3231, 2831, ...] (256 tokens)",
        "attention_mask": "[1, 1, 1, ..., 0] (256 tokens)",
    }

    st.json(example)

# ============================================================================
# PAGE: HYPERPARAMETER CONFIGURATIONS
# ============================================================================
elif page == "Hyperparameter Configurations":
    st.title("Hyperparameter Configurations")

    st.markdown("""
    Four configurations were tested to analyze the impact of:
    - Learning rate
    - Batch size
    - Number of epochs
    - Weight decay (fixed at 0.01)
    """)

    st.divider()

    # Configuration comparison table
    configs_data = []
    for config in CONFIGS:
        configs_data.append(
            {
                "Configuration": config["name"].upper(),
                "Learning Rate": f"{config['lr']:.0e}",
                "Batch Size": config["batch"],
                "Epochs": config["epochs"],
                "Weight Decay": config["wd"],
                "Accuracy": f"{config['accuracy']:.4f}",
            }
        )

    df_configs = pd.DataFrame(configs_data)

    def highlight_best(row):
        if row["Accuracy"] == "0.9800":
            return ["background-color: #90EE90"] * len(row)
        return [""] * len(row)

    styled_df = df_configs.style.apply(highlight_best, axis=1)
    st.dataframe(styled_df, use_container_width=True)

    st.divider()

    # Detailed configuration views
    for i, config in enumerate(CONFIGS, 1):
        with st.expander(f"Configuration {i}: {config['name'].upper()}"):
            cols = st.columns(2)

            with cols[0]:
                st.markdown(f"""
                #### Hyperparameters
                - Learning rate: {config['lr']:.0e}
                - Batch size: {config['batch']}
                - Epochs: {config['epochs']}
                - Weight decay: {config['wd']}
                
                #### Checkpoint Location
                - Directory: results/{config['name']}/
                - Checkpoint frequency: Every epoch
                - Best model: Loaded at end of training
                """)

            with cols[1]:
                st.metric("Validation Accuracy", f"{config['accuracy']:.4f}")
                rank = [c["accuracy"] for c in CONFIGS].index(config["accuracy"]) + 1
                st.metric("Rank", f"#{rank}")

            # Configuration-specific notes
            if config["name"] == "baseline":
                st.info("Baseline configuration with standard parameters.")
            elif config["name"] == "high_lr":
                st.warning(
                    "Higher learning rate (5e-5) resulted in performance degradation (-0.0040)."
                )
            elif config["name"] == "more_epochs":
                st.info("Increased epochs (5 instead of 3) showed no improvement.")
            elif config["name"] == "small_batch":
                st.success(
                    "Reduced batch size (8 instead of 16) achieved best accuracy (+0.0020)."
                )

# ============================================================================
# PAGE: RESULTS AND ANALYSIS
# ============================================================================
elif page == "Results and Analysis":
    st.title("Results and Analysis")

    st.markdown("""
    Comprehensive analysis of training results across all four configurations.
    """)

    st.divider()

    # 1. Accuracy comparison
    st.subheader("Accuracy Comparison")

    configs_names = [c["name"].capitalize() for c in CONFIGS]
    accuracies = [c["accuracy"] for c in CONFIGS]

    fig_accuracy = go.Figure(
        data=[
            go.Bar(
                x=configs_names,
                y=accuracies,
                marker=dict(
                    color=[
                        "#90EE90" if acc == max(accuracies) else "#87CEEB"
                        for acc in accuracies
                    ],
                    line=dict(color="black", width=2),
                ),
                text=[f"{acc:.4f}" for acc in accuracies],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>",
            )
        ]
    )

    fig_accuracy.update_layout(
        title="Validation Set Accuracy Comparison",
        xaxis_title="Configuration",
        yaxis_title="Accuracy",
        hovermode="x unified",
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig_accuracy, use_container_width=True)

    st.divider()

    # 2. Parameter impact analysis
    st.subheader("Parameter Impact Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Learning Rate Impact")
        lr_data = pd.DataFrame(
            {
                "Learning Rate": ["2e-5 (baseline)", "5e-5 (high_lr)"],
                "Accuracy": [0.9780, 0.9740],
                "Change": ["-", "-0.0040"],
            }
        )
        st.dataframe(lr_data, use_container_width=True)
        st.caption("Higher learning rate degrades performance.")

    with col2:
        st.markdown("#### Batch Size Impact")
        batch_data = pd.DataFrame(
            {
                "Batch Size": ["16 (baseline)", "8 (small_batch)"],
                "Accuracy": [0.9780, 0.9800],
                "Change": ["-", "+0.0020"],
            }
        )
        st.dataframe(batch_data, use_container_width=True)
        st.caption("Reduced batch size improves generalization.")

    with col3:
        st.markdown("#### Epochs Impact")
        epochs_data = pd.DataFrame(
            {
                "Epochs": ["3 (baseline)", "5 (more_epochs)"],
                "Accuracy": [0.9780, 0.9780],
                "Change": ["-", "No change"],
            }
        )
        st.dataframe(epochs_data, use_container_width=True)
        st.caption("Additional epochs provide no benefit.")

    st.divider()

    # 3. Final ranking
    st.subheader("Performance Ranking")

    ranking_data = []
    sorted_configs = sorted(CONFIGS, key=lambda x: x["accuracy"], reverse=True)
    ranks = ["1st", "2nd", "3rd", "4th"]

    for rank, config in zip(ranks, sorted_configs):
        ranking_data.append(
            {
                "Rank": rank,
                "Configuration": config["name"].upper(),
                "Accuracy": f"{config['accuracy']:.4f}",
                "Learning Rate": f"{config['lr']:.0e}",
                "Batch Size": config["batch"],
                "Epochs": config["epochs"],
            }
        )

    df_ranking = pd.DataFrame(ranking_data)
    st.dataframe(df_ranking, use_container_width=True)

    st.divider()

    # 4. Performance distribution
    st.subheader("Performance Distribution")

    fig_box = go.Figure(
        data=[
            go.Box(
                y=accuracies,
                x=configs_names,
                marker=dict(color="lightblue"),
                hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>",
            )
        ]
    )

    fig_box.update_layout(
        title="Accuracy Distribution Across Configurations",
        yaxis_title="Accuracy",
        xaxis_title="Configuration",
        height=400,
    )

    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # 5. Conclusions
    st.subheader("Key Findings and Conclusions")

    st.markdown("""
    ### Primary Findings
    
    **Best Configuration: small_batch (Accuracy: 0.9800)**
    - Standard learning rate (2e-5)
    - Reduced batch size (8 instead of 16)
    - 3 epochs
    - Increases weight update frequency
    
    **Learning Rate Impact:**
    - Higher learning rate (5e-5) degraded performance (-0.0040)
    - Oversized update steps cause training instability
    
    **Batch Size Impact:**
    - Reduced batch size improved generalization (+0.0020)
    - Increased weight update frequency proved beneficial
    
    **Epochs Impact:**
    - Increasing from 3 to 5 epochs showed no benefit
    - 3 iterations over 1000 samples sufficient for convergence
    - No evidence of overfitting
    
    ### Recommendations
    
    - Use small_batch configuration for production deployment
    - Accuracy of 0.9800 provides excellent performance for fake news detection
    - Consider experimenting with learning rates between 1e-5 and 2e-5
    - Current configuration achieves convergence without requiring additional epochs
    """)

    st.success("Training completed successfully.")

# Footer
st.divider()
st.markdown("""
---
LLM Fine-Tuning Dashboard | Project: Francesco Screti & Francesco Rolando  
Powered by Streamlit and Plotly
""")
