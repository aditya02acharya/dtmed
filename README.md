# DTMed: Decision Transformer for T2DM Drug Prescription

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

## Overview

**DTMed** is a novel application of Decision Transformer architecture for optimizing Type 2 Diabetes Mellitus (T2DM) drug prescription decisions. This repository contains the python code which leverages sequence modeling to enhance personalized diabetes treatment recommendations.

### Key Features

- **Sequence-based Decision Making**: Treats T2DM drug prescription as a conditional sequence modeling problem
- **Transformer Architecture**: Utilizes the power of attention mechanisms for complex medical decision-making
- **Personalized Treatment**: Generates optimal drug prescriptions based on patient history and desired outcomes
- **Minimal Implementation**: Clean, efficient codebase based on minimal Decision Transformer implementation
- **Reproducible Results**: Complete scripts and configurations to reproduce all experimental results

## Background

Type 2 Diabetes Mellitus affects millions worldwide and requires personalized treatment approaches. Traditional rule-based prescription systems often fail to account for the complex, sequential nature of treatment decisions and patient responses over time.

Our approach applies **Decision Transformer**, a reinforcement learning method that:
- Treats prescription decisions as sequence modeling
- Conditions on desired treatment outcomes (e.g., target HbA1c levels)
- Learns from historical patient data and treatment responses
- Generates optimal action sequences (drug prescriptions) without explicit value function learning

## Architecture

The Decision Transformer for T2DM works by:

1. **Input Conditioning**: Takes patient state (demographics, lab values, medical history), past prescriptions, and desired clinical outcomes
2. **Sequence Modeling**: Uses causally-masked transformer to model the relationship between states, actions, and rewards
3. **Action Generation**: Outputs optimal drug prescription decisions for achieving target therapeutic goals


## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/aditya02acharya/dtmed.git
cd dtmed
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Data Preparation

1. Prepare your T2DM dataset in the required format:
```python
# Example data structure
{
    'states': patient_features,      # Demographics, lab values, comorbidities
    'actions': prescription_history,  # Previous drug prescriptions
    'rewards': clinical_outcomes,     # HbA1c improvements, side effects
    'returns_to_go': target_outcomes  # Desired therapeutic goals
}
```

### Training

Train the Decision Transformer model:

```bash
python train.py \
    --dataset_path data/processed/t2dm_dataset.pkl \
    --model_type decision_transformer \
    --embed_dim 128 \
    --n_layer 3 \
    --n_head 1 \
    --activation_function relu \
    --dropout 0.1 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --warmup_steps 10000 \
    --num_epochs 100 \
    --batch_size 64 \
    --eval_episodes 100
```

### Evaluation

Evaluate the trained model:

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pt \
    --dataset_path data/processed/test_dataset.pkl \
    --num_eval_episodes 1000
```

## Dataset

The model expects T2DM patient data with the following structure:

### Features
- **Demographics**: Age, BMI
- **Clinical Parameters**: EGFR, HbA1c, blood pressure, lipid profile, cholestrol level
- **Comorbidities**: Cardiovascular disease, kidney disease

### Outcomes
- **Primary**: HbA1c reduction
- **Secondary**: Next likely drug recommendation

### Data Format
```json
{
    "patient_id": "string",
    "states": "numpy array of shape (sequence_length, state_dim)",
    "actions": "numpy array of shape (sequence_length, action_dim)", 
    "rewards": "numpy array of shape (sequence_length,)",
    "dones": "numpy array of shape (sequence_length,)",
    "rtg": "numpy array of shape (sequence_length,)"
}
```


## Model Configuration

Key hyperparameters and their descriptions:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 128 | Embedding dimension |
| `n_layer` | 3 | Number of transformer layers |
| `n_head` | 1 | Number of attention heads |
| `max_ep_len` | 1000 | Maximum episode length |
| `context_length` | 30 | Context length for transformer |
| `dropout` | 0.1 | Dropout rate |
| `learning_rate` | 1e-4 | Learning rate |
| `batch_size` | 64 | Training batch size |


## Disclaimer

**Important**: This tool is for research purposes only and should not be used for actual clinical decision-making without proper validation and regulatory approval. Always consult qualified healthcare professionals for medical decisions.
