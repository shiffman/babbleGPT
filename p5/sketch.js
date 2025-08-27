let generator;
let inputText;

async function setup() {
  createCanvas(400, 200);
  background(240);

  // Input field with default prompt
  inputText = createInput('Processing is');
  // Button to trigger text generation
  let button = createButton('Generate Text');
  button.mousePressed(generateText);

  // Load Transformers.js library dynamically from CDN
  // This gives us access to the pipeline function and environment settings
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

  // Automatically choose the available device for inference
  const device = navigator.gpu ? 'webgpu' : 'wasm';

  // Create the text generation pipeline using the local model
  generator = await pipeline('text-generation', modelId, {
    device, // Use 'webgpu' or 'wasm' based on browser support
    dtype: 'fp32', // 32-bit floating point
    progress_callback: logProgress,
  });
}

async function generateText() {
  // Check if model is loaded
  if (!generator) return;

  const prompt = inputText.value();

  // Generate text
  const output = await generator(prompt, {
    max_new_tokens: 80, // Generate up to 80 new tokens
    do_sample: true, // Use sampling (more creative than greedy)
    temperature: 0.7, // Controls randomness (0.0-1.0, higher = more random)
    top_p: 0.9, // Nucleus sampling: only consider top 90% probability mass
    top_k: 0, // Top-k sampling disabled
    repetition_penalty: 1.1, // Slight penalty for repeating words
    no_repeat_ngram_size: 4, // Don't repeat 4-word phrases
  });

  const textOut = output[0].generated_text;
  background(240);
  text(textOut, 10, 10, width - 20, height - 20);
}

function logProgress(progress) {
  console.log(`Loading model: ${progress.status} ${progress.progress || ''}`);
}
