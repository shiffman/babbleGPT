# =============================================================================
# TRAINING SCRIPT - Using Pre-trained GPT-2 Tokenizer
# =============================================================================
# This script trains a small GPT-2 language model from scratch using Hugging Face
# Transformers library. It uses the existing GPT-2 tokenizer but with a much
# smaller model architecture suitable for experimentation and learning.
#
# Key concepts for students:
# - Language modeling: predicting the next word in a sequence
# - Tokenization: converting text to numbers the model can understand
# - Model architecture: layers, attention heads, embedding dimensions
# - Training loop: optimizer, loss function, evaluation
# =============================================================================

# Import PyTorch for deep learning
import torch
# Hugging Face datasets library for loading text data
from datasets import load_dataset
# Hugging Face transformers components we need
from transformers import (
    GPT2Config,                       # Model architecture configuration
    GPT2LMHeadModel,                  # The actual GPT-2 model with language modeling head
    GPT2TokenizerFast,                # Fast tokenizer for converting text to tokens
    DataCollatorForLanguageModeling,  # Handles batching and padding for training
    Trainer,                          # High-level training loop
    TrainingArguments,                # Training configuration and hyperparameters
)

# =============================================================================
# TOKENIZER SETUP
# =============================================================================
# Load the pre-trained GPT-2 tokenizer (Byte-Pair Encoding)
# This tokenizer already knows how to split English text into ~50k subword tokens
tok = GPT2TokenizerFast.from_pretrained("gpt2")

# GPT-2 doesn't have a padding token by default, so we use the end-of-sequence token
# This is needed for batching sequences of different lengths during training
tok.pad_token = tok.eos_token

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================
# Create a much smaller GPT-2 model than the original (117M -> ~2M parameters)
cfg = GPT2Config(
    vocab_size=tok.vocab_size,    # ~50k tokens from GPT-2 tokenizer
    n_positions=512,              # Maximum sequence length the model can handle
    n_ctx=512,                    # Context size (same as n_positions for GPT-2)
    n_embd=256,                   # Embedding dimension (original GPT-2 uses 768)
    n_layer=8,                    # Number of transformer layers (original uses 12)
    n_head=8,                     # Number of attention heads (original uses 12)
    bos_token_id=tok.eos_token_id, # Beginning-of-sequence token
    eos_token_id=tok.eos_token_id, # End-of-sequence token
    pad_token_id=tok.pad_token_id, # Padding token for batching
)

# =============================================================================
# COMMAND LINE ARGUMENTS AND PATHS
# =============================================================================
# Allow users to specify different training data files and optional tags
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
# Default training data (you can point this to any text file)
ap.add_argument("--data", default="data/sample.txt")
# Optional tag to distinguish different training runs
ap.add_argument("--tag", default=None)
args = ap.parse_args()

# If no tag provided, use the filename (without extension) as the tag
tag = args.tag or Path(args.data).stem
# Directory where the final trained model will be saved
hf_dir = f"baby-gpt2-{tag}"
# Directory where training checkpoints will be saved
runs_dir = f"runs/{hf_dir}"

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
# Create the actual model with random weights (we're training from scratch)
# GPT2LMHeadModel = transformer layers + language modeling head for next-token prediction
model = GPT2LMHeadModel(cfg)

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
# Load your text file using Hugging Face datasets
# The "text" loader reads plain text files line by line
ds = load_dataset("text", data_files={"train": args.data})
# Split into 90% training, 10% evaluation for monitoring training progress
split = ds["train"].train_test_split(test_size=0.1, seed=42)

# Convert text to tokens (numbers) that the model can process
# We add an end-of-sequence token after each line to help the model learn document boundaries
def tok_fn(batch):
    return {
        "input_ids": [
            # Convert each text example to token IDs and add EOS token
            tok.encode(x, add_special_tokens=False) + [tok.eos_token_id]
            for x in batch["text"]
        ]
    }

# Apply tokenization to both training and validation splits
# remove_columns removes the original text, keeping only token IDs
tokd = split.map(tok_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)

# =============================================================================
# TEXT CHUNKING
# =============================================================================
# Language models work best with fixed-length sequences
# We concatenate all tokens and split into blocks of exactly 512 tokens
block_size = 512

def group_texts(examples):
    # Concatenate all tokenized examples into one long sequence
    ids = sum(examples["input_ids"], [])
    # Calculate how many complete blocks we can make
    n = (len(ids) // block_size) * block_size
    # Skip if we don't have enough tokens for even one block
    if n == 0:
        return {"input_ids": [], "labels": []}
    # Truncate to exact multiple of block_size
    ids = ids[:n]
    # Split into chunks of block_size
    chunks = [ids[i:i+block_size] for i in range(0, n, block_size)]
    # For language modeling, input and labels are the same (next-token prediction)
    return {"input_ids": chunks, "labels": chunks.copy()}

# Apply the chunking to our tokenized datasets
tokd = tokd.map(group_texts, batched=True)

# =============================================================================
# DATA COLLATION
# =============================================================================
# This handles batching and creates the training targets for language modeling
# mlm=False means we're doing causal (autoregressive) language modeling, not masked LM
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# These hyperparameters control how the model learns
args = TrainingArguments(
    output_dir=runs_dir,              # Where to save checkpoints during training
    per_device_train_batch_size=4,    # How many examples to process at once
    per_device_eval_batch_size=4,     # Batch size for evaluation
    gradient_accumulation_steps=8,     # Accumulate gradients over 8 steps (effective batch size = 4*8=32)
    learning_rate=3e-4,               # How fast the model learns (0.0003)
    lr_scheduler_type="cosine",       # Learning rate decreases over time following cosine curve
    warmup_ratio=0.1,                 # Start with low learning rate for first 10% of training
    weight_decay=0.01,                # Regularization to prevent overfitting
    num_train_epochs=100,             # How many times to go through the entire dataset
    logging_strategy="steps",         # Log training metrics every N steps
    logging_first_step=True,          # Log the very first training step
    logging_steps=1,                  # Log every step (useful for monitoring)
    eval_strategy="steps",            # Evaluate on validation set every N steps
    eval_steps=50,                    # Run evaluation every 50 training steps
    save_steps=100,                   # Save model checkpoint every 100 steps
    save_total_limit=3,               # Only keep the 3 most recent checkpoints
    gradient_checkpointing=True,       # Trade compute for memory (enables larger models)
    max_grad_norm=1.0,                # Clip gradients to prevent exploding gradients
    report_to=[],                     # Don't send metrics to external services
)
# =============================================================================
# TRAINING EXECUTION
# =============================================================================
trainer = Trainer(
    model=model,                    # Our baby GPT-2 model
    args=args,                      # Training configuration
    train_dataset=tokd["train"],    # Tokenized training data
    eval_dataset=tokd["test"],      # Tokenized validation data
    data_collator=collator,         # Handles batching and label creation
)

# Start training! This might take a while depending on your data size and hardware
trainer.train()

# Save the final trained model in Hugging Face format
trainer.save_model(hf_dir)
# Also save the tokenizer so we can decode the model's outputs
tok.save_pretrained(hf_dir)
