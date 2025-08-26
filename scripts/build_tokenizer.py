# =============================================================================
# BYTE-LEVEL TOKENIZER BUILDER
# =============================================================================
# This script creates a custom byte-level tokenizer that treats text more like
# characters than words. Instead of learning subword patterns like standard GPT-2,
# it works at the byte level (256 possible byte values + special tokens).
#
# Why byte-level tokenization?
# - More universal: works with any language/script without preprocessing
# - Smaller vocabulary: ~258 tokens vs ~50,000 for standard GPT-2
# - No out-of-vocabulary issues: can represent any text
# =============================================================================

import os
import argparse
# Hugging Face tokenizers library for training custom tokenizers
from tokenizers import ByteLevelBPETokenizer
# Hugging Face transformers for wrapping our tokenizer
from transformers import GPT2TokenizerFast

# =============================================================================
# MAIN TOKENIZER TRAINING FUNCTION
# =============================================================================
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Input text file to train the tokenizer on
    parser.add_argument("--input", required=True)
    # Output directory where tokenizer files will be saved
    parser.add_argument("--out", default="tokenizer")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)

    # Create a byte-level BPE (Byte-Pair Encoding) tokenizer
    tok = ByteLevelBPETokenizer()
    
    # Train the tokenizer on your input text
    tok.train(
        files=[args.input],                          # Text file(s) to learn from
        vocab_size=258,                              # 256 bytes + 2 special tokens
        min_frequency=1,                             # Include all byte pairs (even rare ones)
        special_tokens=["<|pad|>", "<|endoftext|>"], # Special tokens for padding and end-of-text
    )

    # Save the raw tokenizer files (vocab.json and merges.txt)
    tok.save_model(args.out)

    # Wrap our custom tokenizer in Hugging Face format for compatibility
    # with transformers library (needed for training and inference)
    hf_tok = GPT2TokenizerFast(
        vocab_file=os.path.join(args.out, "vocab.json"),  # Vocabulary mapping
        merges_file=os.path.join(args.out, "merges.txt"),  # Learned merge rules
    )
    
    # Define special tokens that have special meaning
    hf_tok.add_special_tokens(
        {
            "pad_token": "<|pad|>",        # Padding token for batching sequences
            "eos_token": "<|endoftext|>",  # End-of-sequence token
            "bos_token": "<|endoftext|>",  # Beginning-of-sequence token (same as EOS)
        }
    )
    
    # Save in Hugging Face format (creates tokenizer.json, config files, etc.)
    hf_tok.save_pretrained(args.out)

    # Print summary information about the created tokenizer
    print("Saved tokenizer to", args.out)
    print("Vocab size:", hf_tok.vocab_size)

if __name__ == "__main__":
    main()
