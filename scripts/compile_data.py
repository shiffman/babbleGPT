# =============================================================================
# TEXT FILE CONCATENATION SCRIPT
# =============================================================================
# This script takes all .txt files from a subfolder within the data/
# directory and concatenates them into a single output file. This is useful for
# preparing training data from collections of individual text files.
#
# Usage:   python process_data.py --subfolder coding-train-transcripts
# Output:  data/coding-train-transcripts.txt (containing all files concatenated)
# =============================================================================

import os
import argparse
import glob
from pathlib import Path

# =============================================================================
# MAIN CONCATENATION FUNCTION
# =============================================================================
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Concatenate all .txt files from a data subfolder into one file"
    )
    # Subfolder name within the data directory
    parser.add_argument("--subfolder", required=True, 
                       help="Name of the subfolder within data/ (e.g., coding-train-transcripts)")
    # Optional output filename (defaults to subfolder name + .txt)
    parser.add_argument("--output", 
                       help="Output filename (defaults to <subfolder>.txt)")
    # Optional separator between files (defaults to double newline)
    parser.add_argument("--separator", default="\n\n", 
                       help="Separator to insert between files (default: double newline)")
    
    args = parser.parse_args()
    
    # Get script directory and project root
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    
    # Construct paths
    data_dir = root_dir / "data"
    subfolder_path = data_dir / args.subfolder
    
    # Check if subfolder exists
    if not subfolder_path.exists():
        print(f"Error: Subfolder '{args.subfolder}' not found in data directory")
        print(f"Looking for: {subfolder_path}")
        return 1
    
    if not subfolder_path.is_dir():
        print(f"Error: '{args.subfolder}' is not a directory")
        return 1
    
    # Determine output filename
    output_filename = args.output if args.output else f"{args.subfolder}.txt"
    output_path = data_dir / output_filename
    
    # Find all .txt files in the subfolder
    txt_files = list(subfolder_path.glob("*.txt"))
    
    if not txt_files:
        print(f"Warning: No .txt files found in {subfolder_path}")
        return 0
    
    # Sort files by name for consistent ordering
    txt_files.sort(key=lambda x: x.name)
    
    print(f"Found {len(txt_files)} .txt files in '{args.subfolder}'")
    print(f"Concatenating to: {output_path}")
    
    # Concatenate all files
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for i, txt_file in enumerate(txt_files):
            try:
                # Read the content of each file
                with open(txt_file, 'r', encoding='utf-8') as input_file:
                    content = input_file.read()
                
                # Write content to output file
                output_file.write(content)
                
                # Add separator between files (except after the last file)
                if i < len(txt_files) - 1:
                    output_file.write(args.separator)
                
                # Progress indicator for large collections
                if len(txt_files) > 10 and (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(txt_files)} files...")
                    
            except Exception as e:
                print(f"Warning: Could not read {txt_file}: {e}")
                continue
    
    # Print summary information
    output_size = output_path.stat().st_size
    print(f"Successfully concatenated {len(txt_files)} files")
    print(f"Output file size: {output_size:,} bytes")
    print(f"Saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())