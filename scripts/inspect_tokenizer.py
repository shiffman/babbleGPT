# =============================================================================
# TOKENIZER INSPECTION SCRIPT
# =============================================================================
# This utility script helps you understand how your custom tokenizer works
# by showing the vocabulary size and how different characters/strings are
# converted to token IDs. This is useful for debugging and understanding
# the tokenization process.
# =============================================================================

from transformers import GPT2TokenizerFast

# Load the custom tokenizer we built with build_tokenizer.py
tok = GPT2TokenizerFast.from_pretrained("tokenizer")

# Print the total vocabulary size (should be ~258 for byte-level)
print("vocab_size:", tok.vocab_size)
print()

# Test tokenization on various examples to see how it works
print("Tokenization examples:")
for s in ["a", " ", "é", "🙂", "<|endoftext|>"]:
    # Encode the string to token IDs (without adding special tokens automatically)
    ids = tok.encode(s, add_special_tokens=False)
    # Show the original string and its token IDs
    print(f"{repr(s):20} -> {ids}")

print()
print("Notes:")
print("- Simple ASCII characters usually map to 1 token")
print("- Unicode characters (like é, 🙂) map to multiple byte tokens")
print("- Special tokens have their own dedicated IDs")
