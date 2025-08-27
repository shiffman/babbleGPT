import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { env, pipeline } from '@huggingface/transformers';

const here = path.dirname(fileURLToPath(import.meta.url));

// Get model name from command line argument
const arg = process.argv[2];

// Configure Transformers.js to use local models only
env.localModelPath = here;
env.allowRemoteModels = false;

// (But you could also use a remote model)
// env.allowRemoteModels = true;
// const arg = 'gpt2'; // Use Hugging Face's hosted GPT-2

const pipe = await pipeline('text-generation', arg);

const out = await pipe('Processing is', {
  max_new_tokens: 80, // Generate up to 80 new tokens
  do_sample: true, // Use sampling instead of greedy decoding
  temperature: 0.7, // Controls randomness (0.0 = deterministic, 1.0 = very random)
  top_p: 0.9, // Nucleus sampling: only consider tokens in top 90% probability mass
  top_k: 0, // Top-k sampling disabled (0 = off)
  repetition_penalty: 1.1, // Slight penalty for repeating tokens
  no_repeat_ngram_size: 4, // Don't repeat any 4-token sequences
});

console.log(out[0].generated_text);
