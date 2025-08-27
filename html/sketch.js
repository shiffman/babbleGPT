let generator = null;

window.addEventListener('DOMContentLoaded', init);

async function init() {
  // DOM elements
  const loader = document.getElementById('loader');
  const progressBar = document.getElementById('progress');
  const status = document.getElementById('status');
  const prompt = document.getElementById('prompt');
  const generate = document.getElementById('generate');
  const output = document.getElementById('output');

  // Disable the generate button until the model is loaded
  generate.disabled = true;

  try {
    // Automatically detect the available device
    const device = navigator.gpu ? 'webgpu' : 'wasm';
    // Get the base URL where model files are located
    const baseURL = new URL('./', window.location.href).href;

    // import Transformers.js library from CDN
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0');

    // If you want to use a local model, set these flags
    // env.allowLocalModels = true;
    // env.allowRemoteModels = false;
    // env.localModelPath = baseURL;

    // Specify which model to load
    // This can be local!
    // const modelId = 'models/model-byte-coding-train-transcripts';

    // Or if you upload to HF!
    const modelId = 'shiffman/gpt2-coding-train-transcripts';

    // Show loading status to user
    setStatus(status, progressBar, 'initializing…', null);

    // Create the text generation pipeline with progress tracking
    // This loads the ONNX model, tokenizer, and sets up inference
    generator = await pipeline('text-generation', modelId, {
      device, // Use detected device (webgpu or wasm)
      dtype: 'fp32', // Use 32-bit floats
      progress_callback: (p) => {
        // Function called during loading
        const pct = 100 * p.progress;
        const label = p.status || 'loading…';
        setStatus(status, progressBar, label, pct);
      },
    });

    // Model loaded successfully!
    setStatus(status, progressBar, 'ready', 100);
    // Hide the loading UI
    loader.style.display = 'none';
    // Enable the generate button
    generate.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus(status, progressBar, 'error — see console', null);
  }

  // handler for the generate button
  generate.addEventListener('click', async () => {
    // Safety check: make sure model is loaded
    if (!generator) return;

    // Disable UI during generation
    generate.disabled = true;
    output.textContent = 'Generating...';

    try {
      const text = prompt.value;

      // Configure text generation parameters
      const opts = {
        max_new_tokens: 80, // Maximum number of new tokens to generate
        do_sample: true, // Use sampling instead of greedy decoding
        temperature: 0.7, // Randomness (0.0 = deterministic, 1.0 = very random)
        top_p: 0.9, // Nucleus sampling: consider top 90% probability mass
        top_k: 0, // Top-k sampling disabled (use top_p instead)
        repetition_penalty: 1.1, // Penalty for repeating tokens (1.0 = no penalty)
        no_repeat_ngram_size: 4, // Don't repeat any 4-token sequences
      };

      // Generate text using the model
      const out = await generator(text, opts);
      output.textContent = out[0].generated_text;
    } catch (err) {
      console.error(err);
      output.textContent = 'Generation error — see console';
    } finally {
      generate.disabled = false;
    }
  });
}

// Update the loading status and progress bar
function setStatus(el, bar, label, percentOrNull) {
  el.textContent = label;
  bar.style.width = `${Math.max(0, Math.min(100, percentOrNull))}%`;
  bar.style.animation = 'none';
}
