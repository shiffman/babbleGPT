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
cp -r node-js/model-byte-sample html/models/
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

### Share Your Model

You can upload your trained model to Hugging Face Hub to share it with others:

```bash
# First, authenticate with Hugging Face (do this once)
huggingface-cli login

# Upload your ONNX model (after converting with export_js.sh)
python scripts/upload_hf.py --source node-js/model-gpt2-yourmodel --repo-id yourusername/baby-gpt-yourmodel

# Make it private (optional)
python scripts/upload_hf.py --source node-js/model-byte-shakespeare --repo-id yourusername/baby-gpt-shakespeare --private
```

Your model will be available at `https://huggingface.co/yourusername/baby-gpt-yourmodel` and can be loaded by others using `GPT2LMHeadModel.from_pretrained('yourusername/baby-gpt-yourmodel')`.

### Run in JavaScript

Now you can use your model in JavaScript!

#### Option A: Node.js

```bash
# Navigate to js directory
cd node-js

# Install JavaScript dependencies
npm install

# Run with your model
node index.js model-byte-model-name
# or
node index.js model-gpt2-model-name
```

#### Web Browser (Vanilla JS)

```bash
# Copy your model to the html directory
# Either one:
cp -r node-js/model-byte-model-name html/models/
cp -r node-js/model-gpt2-model-name html/models/
```

Edit html/sketch.js and change this line:

```js
// Either
const modelId = 'models/model-gpt2-model-name';
const modelId = 'models/model-byte-model-name';
```

#### With p5.js

```bash
# Copy your model to the p5 directory
# Either one:
cp -r node-js/model-byte-model-name p5/models/
cp -r node-js/model-gpt2-model-name p5/models/
```

Edit p5/sketch.js and change this line:

```js
// Either
const localModel = 'models/model-gpt2-model-name';
const localModel = 'models/model-byte-model-name';
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
