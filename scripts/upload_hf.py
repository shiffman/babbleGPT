# =============================================================================
# HUGGING FACE MODEL UPLOAD SCRIPT
# =============================================================================
# This script uploads a locally trained ONNX model to the Hugging Face Hub,
# making it accessible for sharing, deployment, and inference. It handles all
# the necessary model files including tokenizer configuration and ONNX weights.
#
# The script validates the model directory structure and uploads everything
# needed for others to use your trained model with libraries like Transformers.js
#
# Usage:   python upload_hf.py --source node-js/model-gpt2-srs --repo-id username/my-model
# Example: python upload_hf.py --source node-js/model-byte-shakespeare --repo-id shiffman/baby-gpt-shakespeare --private
# =============================================================================

import argparse
from pathlib import Path
import sys

# Hugging Face Hub library for repository management and file uploads
from huggingface_hub import HfFolder, create_repo, upload_folder

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def die(msg: str) -> None:
    """Print error message and exit with error code"""
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(1)

def require(path: Path, descr: str) -> None:
    """Check that a required file exists, exit with error if missing"""
    if not path.exists():
        die(f"Missing {descr}: {path}")

# =============================================================================
# MAIN UPLOAD FUNCTION  
# =============================================================================
def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Upload a trained ONNX model to Hugging Face Hub for sharing and deployment"
    )
    # Source directory containing the model files (typically from export_js.sh output)
    parser.add_argument("--source", required=True, 
                       help="Path to local model folder (contains config.json and onnx/model.onnx)")
    # Repository identifier on Hugging Face Hub (format: username/repo-name)
    parser.add_argument("--repo-id", required=True, 
                       help="Hub repository ID like USERNAME/REPO_NAME")
    # Whether to create a private repository (default is public)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=False, 
                       help="Create/update as private repository")
    # Custom commit message for this upload
    parser.add_argument("--commit-message", default="Add/Update ONNX model + tokenizer", 
                       help="Git commit message for this upload")
    
    args = parser.parse_args()

    # Check if user is authenticated with Hugging Face
    # Users must run 'huggingface-cli login' first to set up authentication
    token = HfFolder.get_token()
    if not token:
        die("No Hugging Face token found. Run:  huggingface-cli login")

    # Validate source directory exists and is accessible
    src = Path(args.source).resolve()
    if not src.is_dir():
        die(f"--source is not a directory: {src}")

    # =============================================================================
    # VALIDATE REQUIRED MODEL FILES
    # =============================================================================
    # Check for essential configuration files that define the model architecture
    require(src / "config.json", "model configuration")
    require(src / "tokenizer.json", "tokenizer configuration")  
    require(src / "tokenizer_config.json", "tokenizer metadata")
    require(src / "special_tokens_map.json", "special tokens mapping")
    
    # Vocabulary files depend on tokenizer type (BPE vs others)
    # GPT-2 style models need both vocab.json and merges.txt
    vocab = src / "vocab.json"
    merges = src / "merges.txt"
    if not vocab.exists() and not merges.exists():
        print("[warn] Neither vocab.json nor merges.txt found (ok for some tokenizer types)")
    if merges.exists() and not vocab.exists():
        die("Found merges.txt but missing vocab.json (BPE tokenizers need both)")

    # Verify the actual ONNX model file exists
    onnx_file = src / "onnx" / "model.onnx"
    require(onnx_file, "ONNX model file")

    # =============================================================================
    # CREATE AND UPLOAD TO HUGGING FACE HUB
    # =============================================================================
    # Create the repository on Hugging Face Hub (or update if it exists)
    create_repo(
        repo_id=args.repo_id,           # Repository identifier
        repo_type="model",              # This is a model repository (not dataset/space)
        private=bool(args.private),     # Public or private visibility
        exist_ok=True,                  # Don't error if repository already exists
    )

    # Upload all files from the source directory to the Hub repository
    upload_folder(
        repo_id=args.repo_id,           # Where to upload
        repo_type="model",              # Repository type
        folder_path=str(src),           # Local source directory
        path_in_repo=".",               # Upload to repository root
        commit_message=args.commit_message,  # Git commit message
        # Exclude files that shouldn't be uploaded (PyTorch weights, caches, etc.)
        ignore_patterns=["*.pt", "*.bin", "__pycache__/*", "runs/*"],
    )

    # Print success message with direct link to the uploaded model
    visibility = "private" if args.private else "public"
    print(f"[ok] Successfully uploaded to https://huggingface.co/{args.repo_id} ({visibility})")
    print(f"[info] Your model can now be loaded with: GPT2LMHeadModel.from_pretrained('{args.repo_id}')")

if __name__ == "__main__":
    main()
