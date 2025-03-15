from huggingface_hub import HfApi
import os
import json

# Create a model card with description
model_card = """
---
language: en
tags:
- spam-detection
- text-classification
- gpt2
license: mit
---

# GPT-2 Spam Classifier

This model is a fine-tuned version of GPT-2 small for spam detection. It was trained on the SMS Spam Collection Dataset.

## Model Details

- Base model: GPT-2 small (124M parameters)
- Task: Binary classification (spam vs. not spam)
- Training Data: SMS Spam Collection Dataset
- Fine-tuning approach: Last layer + classification head

## Usage

The model expects text input and returns a binary classification (spam/not spam).

## Performance

- Training accuracy: ~95%
- Validation accuracy: ~95%
- Test accuracy: ~93%

## Limitations

This model was trained on SMS messages and may not generalize well to other types of text content.
"""

# Save the model card
with open("README.md", "w") as f:
    f.write(model_card)

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Save the configuration
config = {
    "base_model": "gpt2",
    "vocab_size": BASE_CONFIG["vocab_size"],
    "context_length": BASE_CONFIG["context_length"],
    "emb_dim": BASE_CONFIG["emb_dim"],
    "n_layers": BASE_CONFIG["n_layers"],
    "n_heads": BASE_CONFIG["n_heads"],
    "num_classes": 2
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

api = HfApi()

# Create a new repository
repo_name = "gpt2-spam-classifier"
api.create_repo(repo_name, private=False, exist_ok=True)

# Push files to the repository
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=f"{os.environ.get('HF_USERNAME', 'your-username')}/{repo_name}"
)

api.upload_file(
    path_or_fileobj="config.json",
    path_in_repo="config.json",
    repo_id=f"{os.environ.get('HF_USERNAME', 'your-username')}/{repo_name}"
)

api.upload_file(
    path_or_fileobj="review_classifier.pth",
    path_in_repo="pytorch_model.bin",
    repo_id=f"{os.environ.get('HF_USERNAME', 'your-username')}/{repo_name}"
)

print(f"Model published to: https://huggingface.co/{os.environ.get('HF_USERNAME', 'your-username')}/{repo_name}")