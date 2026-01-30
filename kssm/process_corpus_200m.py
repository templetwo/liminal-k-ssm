#!/usr/bin/env python3
"""
K-SSM v3 Corpus Processor: 200M Token Edition

Processes downloaded texts into training-ready format:
1. Clean and chunk text files
2. Build JSONL corpus
3. Tokenize into numpy arrays
4. Create train/val splits

Usage:
    python3 process_corpus_200m.py --build     # Build JSONL corpus
    python3 process_corpus_200m.py --tokenize  # Tokenize to numpy
    python3 process_corpus_200m.py --all       # Do both
"""

import json
import re
import gc
from pathlib import Path
from typing import List, Generator, Dict
import numpy as np
from tqdm import tqdm

# Try to import tiktoken (GPT-2 tokenizer)
try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("gpt2")
    VOCAB_SIZE = 50257
except ImportError:
    print("WARNING: tiktoken not installed. Using basic tokenization.")
    print("Install with: pip install tiktoken")
    TOKENIZER = None
    VOCAB_SIZE = 50257

# Directories
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw_200m"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = Path("data/cache_v3_200m")  # Top-level data dir

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Processing parameters
MIN_CHUNK_WORDS = 100  # Minimum words per chunk
CHUNK_SIZE = 1024      # Target chunk size in words
OVERLAP = 128          # Overlap between chunks in words

# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean raw text from Gutenberg and other sources.

    - Remove Gutenberg headers/footers
    - Normalize whitespace
    - Remove excessive newlines
    - Preserve paragraph structure
    """

    # Remove Gutenberg header
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
    ]

    for marker in start_markers:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) > 1:
                text = parts[1]
                break

    # Remove Gutenberg footer
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]

    for marker in end_markers:
        if marker in text:
            parts = text.rsplit(marker, 1)
            if len(parts) > 1:
                text = parts[0]
                break

    # Remove page numbers and chapter markers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone page numbers
    text = re.sub(r'\n\s*CHAPTER\s+[IVXLCDM]+\s*\n', '\n\n', text)  # Roman numeral chapters

    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)  # Windows line endings
    text = re.sub(r'\r', '\n', text)    # Old Mac line endings
    text = re.sub(r'\t', ' ', text)     # Tabs to spaces
    text = re.sub(r' +', ' ', text)     # Multiple spaces to single

    # Normalize paragraph breaks (max 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text

# =============================================================================
# TEXT CHUNKING
# =============================================================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.

    Strategy:
    - Split on paragraph boundaries
    - Combine paragraphs into ~chunk_size word chunks
    - Overlap between chunks for context continuity
    - Filter out very short chunks

    Args:
        text: Cleaned text
        chunk_size: Target chunk size in words
        overlap: Overlap size in words

    Returns:
        List of text chunks
    """

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If adding this paragraph exceeds chunk_size, save current chunk
        if current_words > 0 and current_words + para_words > chunk_size:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
                chunks.append(chunk_text)

            # Start new chunk with overlap
            # Keep last few paragraphs that sum to ~overlap words
            overlap_paras = []
            overlap_words = 0
            for p in reversed(current_chunk):
                p_words = len(p.split())
                if overlap_words + p_words > overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_words += p_words

            current_chunk = overlap_paras + [para]
            current_words = overlap_words + para_words
        else:
            current_chunk.append(para)
            current_words += para_words

    # Add final chunk
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk_text)

    return chunks

# =============================================================================
# CORPUS BUILDING
# =============================================================================

def process_text_file(file_path: Path, source_category: str) -> Generator[Dict, None, None]:
    """
    Process a single text file into corpus entries.

    Yields:
        Dict with keys: text, source, category, doc_id, chunk_id, license
    """

    try:
        # Read file
        text = file_path.read_text(encoding='utf-8', errors='ignore')

        # Clean
        text = clean_text(text)

        # Skip if too short
        if len(text.split()) < MIN_CHUNK_WORDS:
            return

        # Chunk
        chunks = chunk_text(text)

        # Extract metadata from filename: {id}_{author}_{title}.txt
        filename = file_path.stem
        parts = filename.split('_', 2)
        doc_id = parts[0] if len(parts) > 0 else filename

        # Yield chunks
        for i, chunk in enumerate(chunks):
            yield {
                'text': chunk,
                'source': source_category,
                'doc_id': f"{source_category}_{doc_id}",
                'chunk_id': i,
                'license': 'Public Domain',
                'file': str(file_path.relative_to(RAW_DIR)),
            }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def build_corpus():
    """
    Build JSONL corpus from all downloaded texts.

    Appends to existing corpus if present (never overwrites).
    """

    print("=" * 70)
    print("BUILDING CORPUS")
    print("=" * 70)

    corpus_path = PROCESSED_DIR / "kssm_corpus_200m.jsonl"

    # Track existing sources
    existing_sources = set()
    existing_chunks = 0

    if corpus_path.exists():
        print(f"\nExisting corpus found: {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing_sources.add(entry.get('file', ''))
                    existing_chunks += 1
                except json.JSONDecodeError:
                    continue

        print(f"Existing chunks: {existing_chunks:,}")
        print(f"Will append new sources only")

    # Find all text files
    all_files = list(RAW_DIR.glob("**/*.txt"))
    print(f"\nTotal text files found: {len(all_files)}")

    # Categorize files
    categories = {
        'gutenberg': RAW_DIR / 'gutenberg',
    }

    new_chunks = 0
    total_chars = 0

    with open(corpus_path, 'a', encoding='utf-8') as f:
        for file_path in tqdm(all_files, desc="Processing texts"):

            # Skip if already processed
            rel_path = str(file_path.relative_to(RAW_DIR))
            if rel_path in existing_sources:
                continue

            # Determine category
            category = 'unknown'
            for cat_name, cat_dir in categories.items():
                if cat_dir in file_path.parents:
                    category = cat_name
                    break

            # Process file
            for entry in process_text_file(file_path, category):
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                new_chunks += 1
                total_chars += len(entry['text'])

    print("\n" + "=" * 70)
    print("CORPUS BUILD COMPLETE")
    print("=" * 70)
    print(f"New chunks added: {new_chunks:,}")
    print(f"Total chunks: {existing_chunks + new_chunks:,}")
    print(f"New text (chars): {total_chars:,}")
    print(f"Output: {corpus_path}")
    print(f"Size: {corpus_path.stat().st_size / 1024 / 1024:.1f} MB")

    return corpus_path

# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize_corpus():
    """
    Tokenize JSONL corpus into numpy arrays.

    Creates:
    - tokens_train.npy (95% of data)
    - tokens_val.npy (5% of data)
    - metadata.json (stats)
    """

    if TOKENIZER is None:
        print("ERROR: tiktoken not installed. Cannot tokenize.")
        print("Install with: pip install tiktoken")
        return

    print("=" * 70)
    print("TOKENIZING CORPUS")
    print("=" * 70)

    corpus_path = PROCESSED_DIR / "kssm_corpus_200m.jsonl"

    if not corpus_path.exists():
        print(f"ERROR: Corpus not found: {corpus_path}")
        print("Run with --build first")
        return

    # Count lines
    print("\nCounting corpus entries...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"Total entries: {total_lines:,}")

    # Tokenize all text
    print("\nTokenizing text...")
    all_tokens = []

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Tokenizing"):
            try:
                entry = json.loads(line)
                text = entry['text']
                tokens = TOKENIZER.encode(text)
                all_tokens.extend(tokens)
            except (json.JSONDecodeError, KeyError) as e:
                continue

    print(f"\nTotal tokens: {len(all_tokens):,}")

    # Convert to numpy array
    print("\nConverting to numpy array...")
    all_tokens = np.array(all_tokens, dtype=np.int32)

    # Train/val split (95/5)
    split_idx = int(len(all_tokens) * 0.95)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    print(f"Train tokens: {len(train_tokens):,} ({len(train_tokens)/1e6:.1f}M)")
    print(f"Val tokens: {len(val_tokens):,} ({len(val_tokens)/1e6:.1f}M)")

    # Save arrays
    print("\nSaving token arrays...")

    train_path = CACHE_DIR / "tokens_train.npy"
    val_path = CACHE_DIR / "tokens_val.npy"

    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)

    print(f"Train: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Val: {val_path} ({val_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Save metadata
    metadata = {
        'total_tokens': len(all_tokens),
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'vocab_size': VOCAB_SIZE,
        'tokenizer': 'tiktoken-gpt2',
        'split_ratio': 0.95,
        'corpus_file': str(corpus_path),
    }

    metadata_path = CACHE_DIR / "tokens_train_meta.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    val_meta_path = CACHE_DIR / "tokens_val_meta.json"
    with open(val_meta_path, 'w') as f:
        json.dump({'n_tokens': len(val_tokens)}, f, indent=2)

    print(f"Metadata: {metadata_path}")

    # Free memory
    del all_tokens
    del train_tokens
    del val_tokens
    gc.collect()

    print("\n" + "=" * 70)
    print("TOKENIZATION COMPLETE")
    print("=" * 70)
    print(f"Total tokens: {metadata['total_tokens']:,} ({metadata['total_tokens']/1e6:.1f}M)")
    print(f"Cache directory: {CACHE_DIR}")

    return metadata

# =============================================================================
# STATISTICS
# =============================================================================

def show_corpus_stats():
    """Show detailed corpus statistics."""

    print("=" * 70)
    print("CORPUS STATISTICS")
    print("=" * 70)

    corpus_path = PROCESSED_DIR / "kssm_corpus_200m.jsonl"

    if not corpus_path.exists():
        print(f"\nCorpus not found: {corpus_path}")
        print("Run with --build first")
        return

    # Analyze corpus
    print(f"\nAnalyzing {corpus_path}...")

    source_stats = {}
    total_chunks = 0
    total_chars = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading corpus"):
            try:
                entry = json.loads(line)
                source = entry.get('source', 'unknown')

                if source not in source_stats:
                    source_stats[source] = {
                        'chunks': 0,
                        'chars': 0,
                    }

                source_stats[source]['chunks'] += 1
                source_stats[source]['chars'] += len(entry['text'])

                total_chunks += 1
                total_chars += len(entry['text'])

            except json.JSONDecodeError:
                continue

    # Print statistics
    print("\n" + "=" * 70)
    print("CORPUS BREAKDOWN")
    print("=" * 70)

    print(f"\n{'Source':<20s} {'Chunks':>10s} {'Chars':>15s} {'Est. Tokens':>15s}")
    print("-" * 70)

    for source in sorted(source_stats.keys()):
        stats = source_stats[source]
        est_tokens = int(stats['chars'] * 0.75)  # Rough estimate
        print(f"{source:<20s} {stats['chunks']:>10,d} {stats['chars']:>15,d} {est_tokens:>15,d}")

    print("-" * 70)
    est_total_tokens = int(total_chars * 0.75)
    print(f"{'TOTAL':<20s} {total_chunks:>10,d} {total_chars:>15,d} {est_total_tokens:>15,d}")

    print(f"\nFile: {corpus_path}")
    print(f"Size: {corpus_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Check if tokenized
    if (CACHE_DIR / "tokens_train_meta.json").exists():
        with open(CACHE_DIR / "tokens_train_meta.json", 'r') as f:
            metadata = json.load(f)

        print(f"\n{'='*70}")
        print("TOKENIZED CORPUS")
        print("=" * 70)
        print(f"Total tokens: {metadata['total_tokens']:,} ({metadata['total_tokens']/1e6:.1f}M)")
        print(f"Train tokens: {metadata['train_tokens']:,} ({metadata['train_tokens']/1e6:.1f}M)")
        print(f"Val tokens: {metadata['val_tokens']:,} ({metadata['val_tokens']/1e6:.1f}M)")
        print(f"Tokenizer: {metadata['tokenizer']}")
        print(f"Vocab size: {metadata['vocab_size']:,}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='K-SSM 200M Token Corpus Processor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 process_corpus_200m.py --build      # Build JSONL corpus from raw texts
  python3 process_corpus_200m.py --tokenize   # Tokenize corpus to numpy arrays
  python3 process_corpus_200m.py --all        # Build and tokenize
  python3 process_corpus_200m.py --stats      # Show corpus statistics
        """
    )

    parser.add_argument('--build', action='store_true',
                       help='Build JSONL corpus from raw texts')
    parser.add_argument('--tokenize', action='store_true',
                       help='Tokenize corpus into numpy arrays')
    parser.add_argument('--all', action='store_true',
                       help='Build and tokenize (complete pipeline)')
    parser.add_argument('--stats', action='store_true',
                       help='Show corpus statistics')

    args = parser.parse_args()

    if args.all:
        build_corpus()
        tokenize_corpus()
        show_corpus_stats()
    elif args.build:
        build_corpus()
    elif args.tokenize:
        tokenize_corpus()
    elif args.stats:
        show_corpus_stats()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
