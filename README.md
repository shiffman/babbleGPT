# Train Your Own Language Model

This repository demonstrates the complete process of training a small GPT-2 language model in Python and deploying it in JavaScript.

## Structure

```
babyGPT/
├── scripts/                         # Python training and utility scripts
│   ├── build_tokenizer.py           # Create custom byte-level tokenizer
│   ├── train_baby_gpt2.py           # Train model with GPT-2 tokenizer
│   ├── train_baby_byte.py           # Train model with custom byte-level tokenizer
│   ├── compile_data.py              # Concatenate txt files from data subfolders
│   ├── export_js.sh                 # Convert PyTorch model to ONNX for JavaScript
│   └── upload_hf.py                 # Upload trained models to Hugging Face Hub
├── data/                            # Training text files
│   ├── sample.txt                   # Sample training data
├── node-js/                         # Node.js implementation
│   ├── index.js
│   └── package.json
├── html/                            # Vanilla JavaScript demo
│   ├── index.html
│   ├── sketch.js
│   └── style.css
├── p5/                              # p5.js demo
│   ├── index.html
│   └── sketch.js
├── tokenizer/                       # Custom tokenizer output (created during training)
├── baby-gpt2-*/                     # Trained GPT-2 models (created during training)
├── baby-byte-*/                     # Trained byte-level models (created during training)
└── runs/                            # Training checkpoints and logs
```

## TLDR!

```bash
python3 -m venv a2z-env
source a2z-env/bin/activate
pip install -U torch transformers datasets tokenizers "optimum[onnxruntime]" onnx huggingface_hub
python scripts/build_tokenizer.py --input data/sample.txt
python scripts/train_baby_byte.py --data data/sample.txt
./scripts/export_js.sh byte sample

# Host your model on Hugging Face
huggingface-cli login
python scripts/upload_hf.py --source node-js/model-byte-sample --repo-id yourusername/baby-gpt-sample

# Run web server for demo
python -m http.server -d html
```

## Guide

### Step 1: Environment Setup

First, create a Python virtual environment and install dependencies:

```bash
# Create virtual environment
python3 -m venv a2z-env

# Activate it
source a2z-env/bin/activate

# Install required Python packages
pip install -U torch transformers datasets tokenizers "optimum[onnxruntime]" onnx huggingface_hub
```

### Data Preparation

If you have a collection of text files in a subfolder, you can concatenate them into a single training file:

```bash
# Concatenate all .txt files from a subfolder in data/
python scripts/compile_data.py --subfolder coding-train-transcripts

# This creates: data/coding-train-transcripts.txt
```

### Training

You have two options for tokenization. Byte-level converges fast on tiny datasets. GPT-2 generalizes better on natural language but may need more steps.

### Train with Byte-Level Tokenizer

- Simpler vocabulary (~258 tokens), works with any language

```bash
# 1. Create custom tokenizer from your text data
python scripts/build_tokenizer.py --input data/your-file.txt

# 2. Train the model
python scripts/train_baby_byte.py --data data/your-file.txt

# Your trained model will be saved as: baby-byte-your-file/
```

### Train with GPT-2 Tokenizer

```bash
# Train directly (no custom tokenizer needed)
python scripts/train_baby_gpt2.py --data data/your-file.txt

# Your trained model will be saved as: baby-gpt2-your-file
```

### Convert to JavaScript (ONNX)

Convert your PyTorch model to ONNX format for JavaScript:

```bash
# For byte-level model:
./scripts/export_js.sh byte model-name

# For GPT-2 model:
./scripts/export_js.sh gpt2 model-name

# This creates: node-js/model-byte-model-name/ or node-js/model-gpt2-model-name/
```

### Host Your Model

Upload your trained model to Hugging Face Hub to host it for client JS:

```bash
# First, authenticate with Hugging Face (do this once)
huggingface-cli login

# Upload your ONNX model (after converting with export_js.sh)
python scripts/upload_hf.py --source node-js/model-gpt2-yourmodel --repo-id yourusername/baby-gpt-yourmodel

# It'll be public, you can make it private (but it won't work from your sketch)
python scripts/upload_hf.py --source node-js/model-byte-shakespeare --repo-id yourusername/baby-gpt-shakespeare --private
```

Your model will be available at `https://huggingface.co/yourusername/baby-gpt-yourmodel`.

### Run in JavaScript

#### Node.js (Local Files)

```bash
# Navigate to js directory
cd node-js

# Install JavaScript dependencies
npm install

# Run with your local model files
node index.js model-byte-model-name
# or
node index.js model-gpt2-model-name
```

#### Client-side JS

The HTML and p5.js demos are configured to load models from Hugging Face Hub by default. Simply update the model ID in the code:

**Vanilla JS (html/sketch.js):**

```js
// Change this line to your uploaded model:
const modelId = 'yourusername/baby-gpt-yourmodel';
```

**p5.js (p5/sketch.js):**

```js
// Change this line to your uploaded model:
const modelId = 'yourusername/baby-gpt-yourmodel';
```

Local server:

```bash
python -m http.server -d html
# or for p5.js version:
python -m http.server -d p5
```

#### Local Model Files

If you prefer to use local model files instead of Hugging Face Hub:

```bash
# Copy your model to the html directory
cp -r node-js/model-byte-yourmodel html/models/

# Update the code to use local models (see the comments in sketch.js files)
```

## Parameters

### Training Parameters

- **`learning_rate`**: How fast the model learns (3e-4 = 0.0003)
- **`num_train_epochs`**: How many times to go through your data
- **`per_device_train_batch_size`**: How many examples to process at once
- **`block_size`**: Using 256–512. (Larger improves quality but needs more VRAM/CPU RAM.)

### Generation Parameters

- **`temperature`**: Randomness (0.0 = deterministic, 1.0 = very random)
- **`top_p`**: Focus on most likely words (0.9 = consider only top 90%)
- **`max_new_tokens`**: How many new words/tokens to generate
- **`repetition_penalty`**: Avoid repeating words (1.1 = slight penalty)

### Adjusting Model Size

Edit the model configuration in the training scripts:

```python
# In train_baby_byte.py or train_baby_gpt2.py
cfg = GPT2Config(
    n_embd=256,    # Increase for larger model (128, 256, 512)
    n_layer=8,     # More layers = more complexity (4, 8, 12)
    n_head=8,      # Attention heads (4, 8, 12)
)
```

**Out of memory during training**

- Reduce `per_device_train_batch_size` to 2 or 1
- Reduce `block_size` to 128 or 256
- Enable `gradient_checkpointing=True`

## Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)
- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
