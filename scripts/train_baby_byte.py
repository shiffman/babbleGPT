# =============================================================================
# BYTE-LEVEL TRAINING SCRIPT
# =============================================================================
# This script trains a GPT-2 model using a custom byte-level tokenizer instead
# of the standard GPT-2 tokenizer. A byte-level tokenizer treats each byte
# (0-255) as a token, making it more like character-level modeling.
#
# Key differences from standard GPT-2 training:
# - Much smaller vocabulary (256 bytes + special tokens vs ~50k subwords)
# - More granular tokenization (closer to character-level)
# =============================================================================

# Import PyTorch for deep learning
import torch
# Hugging Face datasets library for loading text data
from datasets import load_dataset
# Hugging Face transformers components
from transformers import (
    GPT2Config,                       # Model architecture configuration
    GPT2LMHeadModel,                  # The GPT-2 model with language modeling head
    GPT2TokenizerFast,                # Fast tokenizer (will load our custom one)
    DataCollatorForLanguageModeling,  # Handles batching for language modeling
    Trainer,                          # High-level training loop
    TrainingArguments,                # Training configuration
)

# =============================================================================
# CUSTOM TOKENIZER LOADING
# =============================================================================
# Load our custom byte-level tokenizer (created by build_tokenizer.py)
# This tokenizer has only ~258 tokens: 256 bytes + special tokens
tok = GPT2TokenizerFast.from_pretrained("tokenizer")

# =============================================================================
# MODEL CONFIGURATION FOR BYTE-LEVEL TOKENIZATION
# =============================================================================
cfg = GPT2Config(
    vocab_size=tok.vocab_size,      # tokenizer vocabulary size (number of token IDs)
    n_positions=256,                # max positions / sequence length for positional embeddings
    n_ctx=256,                      # context window size used by the model (equals n_positions)
    n_embd=192,                     # hidden size / token embedding dimension
    n_layer=6,                      # number of Transformer blocks
    n_head=8,                       # attention heads per layer
    bos_token_id=tok.bos_token_id,  # beginning-of-sequence token ID
    eos_token_id=tok.eos_token_id,  # end-of-sequence token ID
    pad_token_id=tok.pad_token_id,  # padding token ID
)



# Create the model with byte-level configuration
model = GPT2LMHeadModel(cfg)

# =============================================================================
# COMMAND LINE ARGUMENTS
# =============================================================================
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
# Training data file path
ap.add_argument("--data", default="data/sample.txt")
# Optional tag to identify this training run
ap.add_argument("--tag", default=None)
args = ap.parse_args()

# Use filename as tag if none provided
tag = args.tag or Path(args.data).stem
# Output directories for the byte-level model
hf_dir = f"baby-byte-{tag}"
runs_dir = f"runs/{hf_dir}"

# =============================================================================
# DATA LOADING
# =============================================================================
# Load the training text file
ds = load_dataset("text", data_files={"train": args.data})
# Create train/validation split for monitoring training progress
split = ds["train"].train_test_split(test_size=0.1, seed=42)

# =============================================================================
# BYTE-LEVEL TOKENIZATION
# =============================================================================
# Convert text to byte-level tokens
# Each character becomes one or more byte tokens (UTF-8 encoding)
def tok_fn(batch):
    return {
        "input_ids": [
            # Encode text using our byte-level tokenizer + add EOS token
            tok.encode(x, add_special_tokens=False) + [tok.eos_token_id]
            for x in batch["text"]
        ]
    }

# Apply tokenization to both splits
# load_from_cache_file=False ensures we use our current tokenizer
tokd = split.map(tok_fn, batched=True, remove_columns=["text"], load_from_cache_file=False)


# =============================================================================
# SEQUENCE CHUNKING FOR TRAINING
# =============================================================================
# Group tokens into fixed-size blocks for efficient training
# Smaller block size since byte-level tokens are more numerous
block_size = 256

def group_texts(examples):
    # Concatenate all token sequences into one long sequence
    ids = sum(examples["input_ids"], [])
    # Calculate how many complete blocks we can create
    n = (len(ids) // block_size) * block_size
    if n == 0:
        return {"input_ids": [], "labels": []}
    # Truncate to exact multiple of block_size
    ids = ids[:n]
    # Split into chunks of block_size tokens each
    chunks = [ids[i:i+block_size] for i in range(0, n, block_size)]
    # For causal language modeling, labels are the same as inputs
    return {"input_ids": chunks, "labels": chunks.copy()}

# Apply chunking to our tokenized data
tokd = tokd.map(group_texts, batched=True)

# =============================================================================
# DATA COLLATOR SETUP
# =============================================================================
# Prepare batches for causal language modeling (predict next token)
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Optimized for Apple Silicon (MPS) but works on CPU/CUDA too
args = TrainingArguments(
    output_dir=runs_dir,              # Where to save training checkpoints
    per_device_train_batch_size=8,    # Larger batch size since model is smaller
    per_device_eval_batch_size=8,     # Evaluation batch size
    gradient_accumulation_steps=4,    # Effective batch size = 8*4 = 32
    learning_rate=3e-4,               # Learning rate (0.0003)
    lr_scheduler_type="cosine",       # Learning rate schedule
    warmup_ratio=0.1,                 # Warm up learning rate for first 10%
    weight_decay=0.1,                 # Regularization (higher than GPT-2 version)
    num_train_epochs=30,              # Fewer epochs since byte-level learns faster
    logging_steps=10,                 # Log every 10 steps
    eval_strategy="steps",            # Evaluate periodically during training
    eval_steps=200,                   # Evaluate every 200 training steps
    save_steps=200,                   # Save checkpoint every 200 steps
    save_total_limit=3,               # Keep only 3 most recent checkpoints
    gradient_checkpointing=True,      # Save memory by recomputing gradients
    report_to=[],                     # Don't send metrics to external services
)


# =============================================================================
# TRAINING EXECUTION
# =============================================================================
trainer = Trainer(
    model=model,                    # byte-level GPT model
    args=args,                      # Training configuration
    train_dataset=tokd["train"],    # Training data (tokenized and chunked)
    eval_dataset=tokd["test"],      # Validation data for monitoring
    data_collator=collator,         # Handles batching and label creation
)

# Start the training process
trainer.train()

# Evaluate the final model performance
metrics = trainer.evaluate()
print(metrics)

# Save the trained model in Hugging Face format
trainer.save_model(hf_dir)
# Save the tokenizer alongside the model
tok.save_pretrained(hf_dir)
