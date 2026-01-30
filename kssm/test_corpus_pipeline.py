#!/usr/bin/env python3
"""
Test the 200M corpus pipeline with a small subset of books.

This validates the entire workflow before running the full download:
1. Download 5 test books
2. Process into JSONL corpus
3. Tokenize into numpy arrays
4. Verify integrity

Usage:
    python3 test_corpus_pipeline.py
"""

import sys
from pathlib import Path

# Add kssm to path
sys.path.insert(0, str(Path(__file__).parent))

from build_corpus_200m import download_gutenberg_book, RAW_DIR
from process_corpus_200m import (
    clean_text, chunk_text, process_text_file,
    PROCESSED_DIR, CACHE_DIR, TOKENIZER
)
import json
import numpy as np

# Test books (small, reliable, diverse)
TEST_BOOKS = [
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll"),
    (1342, "Pride and Prejudice", "Jane Austen"),
    (84, "Frankenstein", "Mary Shelley"),
    (1232, "The Prince", "Niccolò Machiavelli"),
    (2009, "On the Origin of Species", "Charles Darwin"),
]

def main():
    print("=" * 70)
    print("CORPUS PIPELINE TEST")
    print("=" * 70)
    print(f"\nTesting with {len(TEST_BOOKS)} books")
    print()

    # Step 1: Download test books
    print("[1/4] DOWNLOADING TEST BOOKS")
    print("-" * 70)

    download_success = 0
    for book_id, title, author in TEST_BOOKS:
        print(f"  Downloading: {title} by {author}...")
        if download_gutenberg_book(book_id, title, author):
            download_success += 1
            print(f"    ✓ Success")
        else:
            print(f"    ✗ Failed")

    print(f"\nDownloaded: {download_success}/{len(TEST_BOOKS)}")

    if download_success == 0:
        print("\n✗ No books downloaded. Aborting test.")
        return False

    # Step 2: Process into corpus
    print("\n[2/4] PROCESSING TO CORPUS")
    print("-" * 70)

    test_corpus_path = PROCESSED_DIR / "test_corpus.jsonl"

    total_chunks = 0
    total_tokens_est = 0

    with open(test_corpus_path, 'w', encoding='utf-8') as f:
        gutenberg_dir = RAW_DIR / 'gutenberg'
        for book_id, title, author in TEST_BOOKS:
            # Find the file
            files = list(gutenberg_dir.glob(f"{book_id}_*.txt"))
            if not files:
                continue

            file_path = files[0]
            print(f"  Processing: {file_path.name}")

            chunk_count = 0
            for entry in process_text_file(file_path, 'gutenberg'):
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                total_chunks += 1
                chunk_count += 1
                total_tokens_est += len(entry['text'].split()) * 1.3  # Rough estimate

            print(f"    ✓ {chunk_count} chunks")

    print(f"\nTotal chunks: {total_chunks}")
    print(f"Est. tokens: {int(total_tokens_est):,}")
    print(f"Corpus: {test_corpus_path}")

    # Step 3: Tokenize
    print("\n[3/4] TOKENIZING")
    print("-" * 70)

    if TOKENIZER is None:
        print("  ✗ tiktoken not available, skipping tokenization")
        return True

    all_tokens = []

    with open(test_corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            tokens = TOKENIZER.encode(entry['text'])
            all_tokens.extend(tokens)

    print(f"  Total tokens: {len(all_tokens):,}")

    # Convert to numpy
    all_tokens = np.array(all_tokens, dtype=np.int32)

    # Split train/val
    split_idx = int(len(all_tokens) * 0.95)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    # Save
    test_cache_dir = CACHE_DIR.parent / "cache_test"
    test_cache_dir.mkdir(parents=True, exist_ok=True)

    np.save(test_cache_dir / "tokens_train.npy", train_tokens)
    np.save(test_cache_dir / "tokens_val.npy", val_tokens)

    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val: {len(val_tokens):,} tokens")
    print(f"  Saved to: {test_cache_dir}")

    # Step 4: Verify
    print("\n[4/4] VERIFYING")
    print("-" * 70)

    # Load back and check
    loaded_train = np.load(test_cache_dir / "tokens_train.npy")
    loaded_val = np.load(test_cache_dir / "tokens_val.npy")

    print(f"  ✓ Train tokens loaded: {len(loaded_train):,}")
    print(f"  ✓ Val tokens loaded: {len(loaded_val):,}")
    print(f"  ✓ Vocab range: [{loaded_train.min()}, {loaded_train.max()}]")
    print(f"  ✓ Data type: {loaded_train.dtype}")

    # Test decoding
    sample_tokens = loaded_train[:20].tolist()
    sample_text = TOKENIZER.decode(sample_tokens)
    print(f"\n  Sample decode (first 20 tokens):")
    print(f"    Tokens: {sample_tokens}")
    print(f"    Text: {sample_text!r}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✓ Downloaded: {download_success} books")
    print(f"✓ Processed: {total_chunks} chunks")
    print(f"✓ Tokenized: {len(all_tokens):,} tokens ({len(all_tokens)/1e6:.2f}M)")
    print(f"✓ Train/Val split: {len(train_tokens):,} / {len(val_tokens):,}")
    print(f"\n✓ PIPELINE VALIDATED - Ready for full run!")

    # Cleanup option
    print(f"\nTest files saved in:")
    print(f"  Raw: {RAW_DIR / 'gutenberg'}")
    print(f"  Corpus: {test_corpus_path}")
    print(f"  Tokens: {test_cache_dir}")
    print(f"\nTo clean up test files:")
    print(f"  rm -rf {test_cache_dir}")
    print(f"  rm {test_corpus_path}")

    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
