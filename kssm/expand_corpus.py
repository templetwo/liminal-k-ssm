#!/usr/bin/env python3
"""
Expand K-SSM Training Corpus

Builds on existing 22.2M token corpus (96 Gutenberg texts) to reach target size.

Strategy:
1. Complete Gutenberg (finish the 101 planned + add 200 more curated classics)
2. Download OpenStax textbooks (10 books, CC BY 4.0)
3. Add Philosophy Corpus (Plato, Aristotle, etc. - public domain)
4. Add high-quality curated sources

Target: 100M+ tokens for production-grade training

IMPORTANT: Preserves existing corpus, appends new data
Output: kssm_corpus_expanded.jsonl (can merge with existing later)
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Generator, Dict, List
import html
from html.parser import HTMLParser

# Import cleaning/chunking utilities from build_corpus.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from build_corpus import (
    clean_text, chunk_text, html_to_text,
    MIN_CHUNK_WORDS, DATA_DIR, RAW_DIR, PROCESSED_DIR
)

# ==============================================================================
# Extended Gutenberg Catalog
# ==============================================================================

# Philosophy & Theology (High Priority for Consciousness Research)
PHILOSOPHY_BOOKS = [
    # Ancient Philosophy
    (1497, "The Republic", "Plato"),
    (1600, "Apology", "Plato"),
    (1656, "Phaedo", "Plato"),
    (1750, "The Symposium", "Plato"),
    (6763, "The Politics", "Aristotle"),
    (8438, "Nicomachean Ethics", "Aristotle"),
    (2412, "Enchiridion", "Epictetus"),
    (5740, "Meditations", "Marcus Aurelius"),

    # Eastern Philosophy
    (2680, "Meditations on First Philosophy", "Descartes"),
    (10574, "Tao Teh King", "Laozi"),
    (2500, "I Ching", "Anonymous"),

    # Modern Philosophy
    (4705, "The Critique of Pure Reason", "Kant"),
    (5682, "An Enquiry Concerning Human Understanding", "Hume"),
    (4363, "Thus Spake Zarathustra", "Nietzsche"),
    (56, "The Ethics", "Spinoza"),
    (26659, "Being and Time", "Heidegger"),  # If available

    # Consciousness-Related
    (2680, "Discourse on Method", "Descartes"),
    (10022, "The Principles of Psychology Vol 1", "William James"),
    (10023, "The Principles of Psychology Vol 2", "William James"),
]

# Classic Literature (Depth & Diversity)
LITERATURE_BOOKS = [
    # British Classics
    (2701, "Moby Dick", "Herman Melville"),
    (345, "Dracula", "Bram Stoker"),
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll"),
    (74, "The Adventures of Tom Sawyer", "Mark Twain"),
    (76, "Adventures of Huckleberry Finn", "Mark Twain"),
    (84, "Frankenstein", "Mary Shelley"),
    (98, "A Tale of Two Cities", "Charles Dickens"),
    (1400, "Great Expectations", "Charles Dickens"),

    # American Classics
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (244, "A Study in Scarlet", "Arthur Conan Doyle"),
    (209, "The Turn of the Screw", "Henry James"),
    (514, "Little Women", "Louisa May Alcott"),
    (1260, "Jane Eyre", "Charlotte Brontë"),
    (768, "Wuthering Heights", "Emily Brontë"),

    # World Literature
    (2554, "Crime and Punishment", "Fyodor Dostoyevsky"),
    (2600, "War and Peace", "Leo Tolstoy"),
    (1399, "Anna Karenina", "Leo Tolstoy"),
    (2638, "The Idiot", "Fyodor Dostoyevsky"),
    (28054, "The Brothers Karamazov", "Fyodor Dostoyevsky"),

    # Plays & Poetry
    (1524, "Hamlet", "William Shakespeare"),
    (1513, "Romeo and Juliet", "William Shakespeare"),
    (1533, "Macbeth", "William Shakespeare"),
    (1532, "King Lear", "William Shakespeare"),
    (1041, "The Divine Comedy", "Dante Alighieri"),
    (1727, "The Odyssey", "Homer"),
    (6130, "The Iliad", "Homer"),
]

# Science & Nature Writing (For Technical Language)
SCIENCE_BOOKS = [
    (2009, "On the Origin of Species", "Charles Darwin"),
    (1228, "The Voyage of the Beagle", "Charles Darwin"),
    (30155, "The Descent of Man", "Charles Darwin"),
    (1780, "Relativity: The Special and General Theory", "Albert Einstein"),
    (5001, "The Autobiography of Benjamin Franklin", "Benjamin Franklin"),
    (4280, "Discourse on Floating Bodies", "Galileo Galilei"),
]

# Historical Documents (For Formal/Legal Language)
HISTORICAL_BOOKS = [
    (1, "The Declaration of Independence of the United States", "Thomas Jefferson"),
    (5, "The United States Constitution", "United States"),
    (6593, "History of the Peloponnesian War", "Thucydides"),
    (1232, "The Prince", "Niccolò Machiavelli"),
    (3600, "Essays of Michel de Montaigne", "Michel de Montaigne"),
]

ALL_EXPANSION_BOOKS = (
    PHILOSOPHY_BOOKS +
    LITERATURE_BOOKS +
    SCIENCE_BOOKS +
    HISTORICAL_BOOKS
)

# ==============================================================================
# OpenStax Textbooks (CC BY 4.0)
# ==============================================================================

OPENSTAX_BOOKS = [
    {
        'title': 'Psychology 2e',
        'url': 'https://openstax.org/details/books/psychology-2e',
        'id': 'openstax_psychology_2e',
        'slug': 'psychology-2e'
    },
    {
        'title': 'Introduction to Philosophy',
        'url': 'https://openstax.org/details/books/introduction-philosophy',
        'id': 'openstax_philosophy',
        'slug': 'introduction-philosophy'
    },
    {
        'title': 'Introduction to Sociology 3e',
        'url': 'https://openstax.org/details/books/introduction-sociology-3e',
        'id': 'openstax_sociology_3e',
        'slug': 'introduction-sociology-3e'
    },
    {
        'title': 'American Government 3e',
        'url': 'https://openstax.org/details/books/american-government-3e',
        'id': 'openstax_american_government_3e',
        'slug': 'american-government-3e'
    },
    {
        'title': 'World History Volume 1',
        'url': 'https://openstax.org/details/books/world-history-volume-1',
        'id': 'openstax_world_history_1',
        'slug': 'world-history-volume-1'
    },
    {
        'title': 'Biology 2e',
        'url': 'https://openstax.org/details/books/biology-2e',
        'id': 'openstax_biology_2e',
        'slug': 'biology-2e'
    },
    {
        'title': 'Principles of Economics 3e',
        'url': 'https://openstax.org/details/books/principles-economics-3e',
        'id': 'openstax_economics_3e',
        'slug': 'principles-economics-3e'
    },
]

# ==============================================================================
# Gutenberg Download
# ==============================================================================

def download_gutenberg_book(book_id: int, title: str, max_retries: int = 3) -> str:
    """Download a book from Project Gutenberg with retry logic."""
    output_dir = RAW_DIR / "gutenberg"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    output_path = output_dir / f"{book_id}_{safe_title}.txt"

    # Skip if already downloaded
    if output_path.exists():
        print(f"  ✓ Already downloaded: {title}")
        with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    # Try multiple URL patterns
    url_patterns = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    for attempt in range(max_retries):
        for url in url_patterns:
            try:
                print(f"  Downloading from: {url}")
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (K-SSM Corpus Builder/1.0)'}
                )

                with urllib.request.urlopen(req, timeout=30) as response:
                    content = response.read().decode('utf-8', errors='ignore')

                # Verify we got actual content (not error page)
                if len(content) > 1000 and 'Project Gutenberg' in content:
                    # Save to disk
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    print(f"  ✓ Downloaded: {title} ({len(content)} chars)")
                    return content

            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                print(f"  ✗ Failed: {e}")
                continue

        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 2
            print(f"  Retrying in {wait_time}s...")
            time.sleep(wait_time)

    print(f"  ✗ FAILED after {max_retries} attempts: {title}")
    return ""

def process_gutenberg_expansion(max_books: int = None) -> Generator[Dict, None, None]:
    """Process expanded Gutenberg catalog."""
    books = ALL_EXPANSION_BOOKS[:max_books] if max_books else ALL_EXPANSION_BOOKS

    print(f"\nProcessing {len(books)} Gutenberg books...")
    successful = 0
    failed = 0

    for i, (book_id, title, author) in enumerate(books):
        print(f"\n[{i+1}/{len(books)}] {title} by {author}")

        text = download_gutenberg_book(book_id, title)

        if not text or len(text.split()) < MIN_CHUNK_WORDS:
            failed += 1
            continue

        # Clean text
        text = clean_text(text)

        # Remove Gutenberg header/footer
        text = re.sub(r'\*\*\* START OF .*? \*\*\*', '', text, flags=re.DOTALL)
        text = re.sub(r'\*\*\* END OF .*? \*\*\*.*', '', text, flags=re.DOTALL)

        # Chunk
        chunks = chunk_text(text)

        for chunk_id, chunk in enumerate(chunks):
            yield {
                'text': chunk,
                'source': 'gutenberg',
                'title': title,
                'author': author,
                'license': 'Public Domain',
                'doc_id': f"gutenberg_{book_id}",
                'chunk_id': chunk_id
            }

        successful += 1

        # Rate limiting (be nice to Project Gutenberg)
        time.sleep(1)

    print(f"\n✓ Gutenberg: {successful} successful, {failed} failed")

# ==============================================================================
# OpenStax Download (CC BY 4.0)
# ==============================================================================

def download_openstax_book(book: Dict) -> str:
    """
    Download OpenStax textbook.

    NOTE: OpenStax requires web scraping or API access. This is a placeholder.
    For production, use OpenStax API or manually download PDFs and convert.
    """
    output_dir = RAW_DIR / "openstax"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{book['id']}.txt"

    if output_path.exists():
        print(f"  ✓ Already downloaded: {book['title']}")
        with open(output_path, 'r', encoding='utf-8') as f:
            return f.read()

    print(f"  ⚠ OpenStax download requires manual setup")
    print(f"    Visit: {book['url']}")
    print(f"    Download PDF, convert to text, save as: {output_path}")
    print(f"    Skipping for now...")

    return ""

def process_openstax() -> Generator[Dict, None, None]:
    """Process OpenStax textbooks."""
    print("\nProcessing OpenStax textbooks...")

    for book in OPENSTAX_BOOKS:
        print(f"\n{book['title']}")

        text = download_openstax_book(book)

        if not text or len(text.split()) < MIN_CHUNK_WORDS:
            continue

        text = clean_text(text)
        chunks = chunk_text(text)

        for chunk_id, chunk in enumerate(chunks):
            yield {
                'text': chunk,
                'source': 'openstax',
                'title': book['title'],
                'author': 'OpenStax',
                'license': 'CC BY 4.0',
                'doc_id': book['id'],
                'chunk_id': chunk_id
            }

# ==============================================================================
# Main Expansion Pipeline
# ==============================================================================

def expand_corpus(
    output_path: Path = PROCESSED_DIR / "kssm_corpus_expanded.jsonl",
    include_gutenberg: bool = True,
    include_openstax: bool = False,  # Requires manual setup
    max_gutenberg: int = None
):
    """
    Expand training corpus with high-quality sources.

    Creates NEW file (doesn't overwrite existing corpus).
    Can be merged with existing 22M token corpus later.
    """
    print("=" * 70)
    print("EXPANDING K-SSM TRAINING CORPUS")
    print("=" * 70)
    print(f"Output: {output_path}")
    print()
    print("IMPORTANT: This creates a NEW file")
    print("Existing corpus preserved at: kssm_corpus.jsonl")
    print()

    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_tokens_approx = 0

    with open(output_path, 'w', encoding='utf-8') as f:

        if include_gutenberg:
            print("\n[1/2] EXPANDED GUTENBERG CATALOG (Public Domain)")
            print("-" * 60)
            for chunk in process_gutenberg_expansion(max_gutenberg):
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                total_chunks += 1
                total_tokens_approx += len(chunk['text'].split()) * 1.3

            print(f"\n  Gutenberg expansion: {total_chunks} chunks")

        if include_openstax:
            print("\n[2/2] OPENSTAX TEXTBOOKS (CC BY 4.0)")
            print("-" * 60)
            openstax_start = total_chunks
            for chunk in process_openstax():
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                total_chunks += 1
                total_tokens_approx += len(chunk['text'].split()) * 1.3

            print(f"\n  OpenStax: {total_chunks - openstax_start} chunks")

    # Summary
    print("\n" + "=" * 70)
    print("CORPUS EXPANSION COMPLETE")
    print("=" * 70)
    print(f"New chunks added: {total_chunks:,}")
    print(f"Approx new tokens: {int(total_tokens_approx):,}")
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("NEXT STEPS:")
    print("1. Verify expanded corpus quality")
    print("2. Merge with existing corpus (or use separately)")
    print("3. Re-tokenize for training")

    return {
        'total_chunks': total_chunks,
        'approx_tokens': int(total_tokens_approx),
        'output_path': str(output_path)
    }


def merge_corpora(
    existing_path: Path = PROCESSED_DIR / "kssm_corpus.jsonl",
    expanded_path: Path = PROCESSED_DIR / "kssm_corpus_expanded.jsonl",
    output_path: Path = PROCESSED_DIR / "kssm_corpus_full.jsonl"
):
    """Merge existing 22M corpus with expanded corpus."""
    print("=" * 70)
    print("MERGING CORPORA")
    print("=" * 70)
    print(f"Existing: {existing_path}")
    print(f"Expanded: {expanded_path}")
    print(f"Output:   {output_path}")
    print()

    total_lines = 0

    with open(output_path, 'w', encoding='utf-8') as out_f:
        # Copy existing corpus
        print("Copying existing corpus...")
        with open(existing_path, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                out_f.write(line)
                total_lines += 1

        print(f"  Existing: {total_lines} chunks")

        # Append expanded corpus
        print("Appending expanded corpus...")
        start_line = total_lines
        with open(expanded_path, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                out_f.write(line)
                total_lines += 1

        print(f"  Expanded: {total_lines - start_line} chunks")

    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Total chunks: {total_lines:,}")
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Expand K-SSM training corpus')
    parser.add_argument('--expand', action='store_true', help='Build expanded corpus')
    parser.add_argument('--merge', action='store_true', help='Merge existing + expanded')
    parser.add_argument('--max-books', type=int, help='Limit Gutenberg books')
    parser.add_argument('--include-openstax', action='store_true', help='Include OpenStax (requires manual download)')

    args = parser.parse_args()

    if args.expand:
        expand_corpus(
            include_gutenberg=True,
            include_openstax=args.include_openstax,
            max_gutenberg=args.max_books
        )
    elif args.merge:
        merge_corpora()
    else:
        print("Usage:")
        print("  python expand_corpus.py --expand              # Download new books")
        print("  python expand_corpus.py --expand --max-books 50  # Limit to 50 books")
        print("  python expand_corpus.py --merge               # Merge with existing")
