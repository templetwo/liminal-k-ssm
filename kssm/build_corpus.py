#!/usr/bin/env python3
"""
Build K-SSM Training Corpus

Sources (all clean licenses):
1. Standard Ebooks (CC0 - public domain) - ~150M tokens of curated novels
2. OpenStax (CC BY 4.0) - ~50M tokens of textbooks

Output: JSONL with fields {text, source, title, author, license, doc_id}
Target: ~200M tokens total

EOS Collapse Prevention:
- Minimum chunk length enforced
- Include dialogue-heavy texts (plays, novels with conversation)
- Chunks overlap to maintain continuity
"""

import json
import os
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Generator, Dict, List
import urllib.request
import html
from html.parser import HTMLParser


# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CHUNK_SIZE = 1024  # tokens (approximate via words * 1.3)
CHUNK_OVERLAP = 128  # overlap between chunks
MIN_CHUNK_WORDS = 100  # minimum words per chunk (EOS collapse prevention)


# ==============================================================================
# Text Cleaning
# ==============================================================================

class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML."""
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'head', 'meta', 'link'}
        self.current_skip = False

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.current_skip = True
        elif tag in {'p', 'br', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}:
            self.text_parts.append('\n')

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.current_skip = False
        elif tag in {'p', 'div'}:
            self.text_parts.append('\n')

    def handle_data(self, data):
        if not self.current_skip:
            self.text_parts.append(data)

    def get_text(self):
        return ''.join(self.text_parts)


def html_to_text(html_content: str) -> str:
    """Convert HTML to plain text."""
    parser = HTMLTextExtractor()
    parser.feed(html_content)
    text = parser.get_text()
    text = html.unescape(text)
    return text


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Normalize unicode
    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove control characters (keep newlines)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP,
               min_words: int = MIN_CHUNK_WORDS) -> List[str]:
    """
    Split text into overlapping chunks.

    EOS Collapse Prevention: enforce minimum chunk length.
    """
    words = text.split()

    if len(words) < min_words:
        return []  # Skip very short texts

    # Approximate tokens as words * 1.3
    words_per_chunk = int(chunk_size / 1.3)
    overlap_words = int(overlap / 1.3)

    chunks = []
    start = 0

    while start < len(words):
        end = start + words_per_chunk
        chunk_words = words[start:end]

        if len(chunk_words) >= min_words:
            chunks.append(' '.join(chunk_words))

        start = end - overlap_words

        # Prevent infinite loop
        if start >= len(words) - overlap_words:
            break

    return chunks


# ==============================================================================
# Standard Ebooks (CC0)
# ==============================================================================

STANDARD_EBOOKS_CATALOG = "https://standardebooks.org/opds/all"

def download_standard_ebooks_catalog() -> List[Dict]:
    """
    Download Standard Ebooks catalog.

    Returns list of {title, author, url, id}
    """
    print("Fetching Standard Ebooks catalog...")

    catalog_path = RAW_DIR / "standard_ebooks_catalog.xml"

    if not catalog_path.exists():
        # Download OPDS catalog
        urllib.request.urlretrieve(STANDARD_EBOOKS_CATALOG, catalog_path)

    # Parse OPDS (simplified - just extract epub URLs)
    with open(catalog_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract entries using regex (OPDS is XML but we keep it simple)
    entries = []

    # Find all entry blocks
    entry_pattern = r'<entry>(.*?)</entry>'
    title_pattern = r'<title>([^<]+)</title>'
    author_pattern = r'<name>([^<]+)</name>'
    id_pattern = r'<id>([^<]+)</id>'
    epub_pattern = r'href="([^"]+\.epub)"'

    for entry_match in re.finditer(entry_pattern, content, re.DOTALL):
        entry_content = entry_match.group(1)

        title_match = re.search(title_pattern, entry_content)
        author_match = re.search(author_pattern, entry_content)
        id_match = re.search(id_pattern, entry_content)
        epub_match = re.search(epub_pattern, entry_content)

        if title_match and epub_match:
            entries.append({
                'title': html.unescape(title_match.group(1)),
                'author': html.unescape(author_match.group(1)) if author_match else 'Unknown',
                'id': id_match.group(1) if id_match else f"se_{len(entries)}",
                'url': epub_match.group(1),
                'source': 'standard_ebooks',
                'license': 'CC0'
            })

    print(f"Found {len(entries)} Standard Ebooks entries")
    return entries


def extract_epub_text(epub_path: Path) -> str:
    """Extract plain text from EPUB file."""
    text_parts = []

    try:
        with zipfile.ZipFile(epub_path, 'r') as zf:
            for filename in zf.namelist():
                if filename.endswith(('.xhtml', '.html', '.htm')):
                    try:
                        content = zf.read(filename).decode('utf-8', errors='ignore')
                        text = html_to_text(content)
                        if text.strip():
                            text_parts.append(text)
                    except Exception as e:
                        continue
    except Exception as e:
        print(f"Error extracting {epub_path}: {e}")
        return ""

    return '\n\n'.join(text_parts)


def process_standard_ebooks(max_books: int = None) -> Generator[Dict, None, None]:
    """
    Download and process Standard Ebooks.

    Yields chunks as {text, source, title, author, license, doc_id, chunk_id}
    """
    entries = download_standard_ebooks_catalog()

    if max_books:
        entries = entries[:max_books]

    ebooks_dir = RAW_DIR / "standard_ebooks"
    ebooks_dir.mkdir(exist_ok=True)

    for i, entry in enumerate(entries):
        print(f"Processing [{i+1}/{len(entries)}] {entry['title'][:50]}...")

        # Download EPUB if not cached
        safe_title = re.sub(r'[^\w\-]', '_', entry['title'])[:50]
        epub_path = ebooks_dir / f"{safe_title}.epub"

        if not epub_path.exists():
            try:
                urllib.request.urlretrieve(entry['url'], epub_path)
            except Exception as e:
                print(f"  Failed to download: {e}")
                continue

        # Extract text
        text = extract_epub_text(epub_path)
        text = clean_text(text)

        if not text or len(text.split()) < MIN_CHUNK_WORDS:
            continue

        # Chunk
        chunks = chunk_text(text)

        for chunk_id, chunk in enumerate(chunks):
            yield {
                'text': chunk,
                'source': 'standard_ebooks',
                'title': entry['title'],
                'author': entry['author'],
                'license': 'CC0',
                'doc_id': entry['id'],
                'chunk_id': chunk_id
            }


# ==============================================================================
# OpenStax (CC BY 4.0)
# ==============================================================================

# Manually curated list of OpenStax textbooks with direct download URLs
OPENSTAX_BOOKS = [
    {
        'title': 'Psychology 2e',
        'url': 'https://openstax.org/details/books/psychology-2e',
        'id': 'openstax_psychology_2e'
    },
    {
        'title': 'American Government 3e',
        'url': 'https://openstax.org/details/books/american-government-3e',
        'id': 'openstax_american_government_3e'
    },
    {
        'title': 'Introduction to Sociology 3e',
        'url': 'https://openstax.org/details/books/introduction-sociology-3e',
        'id': 'openstax_sociology_3e'
    },
    {
        'title': 'US History',
        'url': 'https://openstax.org/details/books/us-history',
        'id': 'openstax_us_history'
    },
    {
        'title': 'World History Volume 1',
        'url': 'https://openstax.org/details/books/world-history-volume-1',
        'id': 'openstax_world_history_1'
    },
    {
        'title': 'Introduction to Philosophy',
        'url': 'https://openstax.org/details/books/introduction-philosophy',
        'id': 'openstax_philosophy'
    },
    {
        'title': 'Writing Guide with Handbook',
        'url': 'https://openstax.org/details/books/writing-guide',
        'id': 'openstax_writing_guide'
    },
    {
        'title': 'Principles of Economics 3e',
        'url': 'https://openstax.org/details/books/principles-economics-3e',
        'id': 'openstax_economics_3e'
    },
    {
        'title': 'Biology 2e',
        'url': 'https://openstax.org/details/books/biology-2e',
        'id': 'openstax_biology_2e'
    },
    {
        'title': 'Astronomy 2e',
        'url': 'https://openstax.org/details/books/astronomy-2e',
        'id': 'openstax_astronomy_2e'
    },
]


def process_openstax(max_books: int = None) -> Generator[Dict, None, None]:
    """
    Process OpenStax textbooks.

    Note: OpenStax requires manual download of PDFs or web scraping.
    For now, we'll create placeholder entries and document the manual process.
    """
    print("\n" + "=" * 60)
    print("OPENSTAX TEXTBOOKS (CC BY 4.0)")
    print("=" * 60)
    print("\nOpenStax books require manual download from:")
    print("  https://openstax.org/subjects")
    print("\nDownload the 'Offline PDF' or 'Web View' and save to:")
    print(f"  {RAW_DIR / 'openstax/'}")
    print("\nBooks to download:")

    for book in OPENSTAX_BOOKS[:max_books] if max_books else OPENSTAX_BOOKS:
        print(f"  - {book['title']}")
        print(f"    {book['url']}")

    # Check if any OpenStax files exist
    openstax_dir = RAW_DIR / "openstax"
    openstax_dir.mkdir(exist_ok=True)

    # Process any text files that have been manually added
    for txt_file in openstax_dir.glob("*.txt"):
        print(f"\nProcessing {txt_file.name}...")

        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()

        text = clean_text(text)
        chunks = chunk_text(text)

        for chunk_id, chunk in enumerate(chunks):
            yield {
                'text': chunk,
                'source': 'openstax',
                'title': txt_file.stem,
                'author': 'OpenStax',
                'license': 'CC BY 4.0',
                'doc_id': f"openstax_{txt_file.stem}",
                'chunk_id': chunk_id
            }


# ==============================================================================
# Project Gutenberg Fallback (also public domain)
# ==============================================================================

GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub"

# Classic English literature (all public domain)
GUTENBERG_BOOKS = [
    # Classic English Novels
    (1342, "Pride and Prejudice", "Jane Austen"),
    (158, "Emma", "Jane Austen"),
    (161, "Sense and Sensibility", "Jane Austen"),
    (84, "Frankenstein", "Mary Shelley"),
    (1661, "Sherlock Holmes Adventures", "Arthur Conan Doyle"),
    (2852, "The Hound of the Baskervilles", "Arthur Conan Doyle"),
    (98, "A Tale of Two Cities", "Charles Dickens"),
    (730, "Oliver Twist", "Charles Dickens"),
    (1400, "Great Expectations", "Charles Dickens"),
    (766, "David Copperfield", "Charles Dickens"),
    (2701, "Moby Dick", "Herman Melville"),
    (11, "Alice in Wonderland", "Lewis Carroll"),
    (12, "Through the Looking Glass", "Lewis Carroll"),
    (174, "The Picture of Dorian Gray", "Oscar Wilde"),
    (768, "Wuthering Heights", "Emily Brontë"),
    (1260, "Jane Eyre", "Charlotte Brontë"),
    (145, "Middlemarch", "George Eliot"),

    # American Literature
    (76, "Adventures of Huckleberry Finn", "Mark Twain"),
    (74, "Adventures of Tom Sawyer", "Mark Twain"),
    (1952, "The Yellow Wallpaper", "Charlotte Perkins Gilman"),
    (514, "Little Women", "Louisa May Alcott"),
    (209, "The Turn of the Screw", "Henry James"),
    (36, "The War of the Worlds", "H.G. Wells"),
    (35, "The Time Machine", "H.G. Wells"),
    (159, "The Island of Doctor Moreau", "H.G. Wells"),

    # Philosophy & Essays
    (1232, "The Prince", "Niccolò Machiavelli"),
    (1080, "A Modest Proposal", "Jonathan Swift"),
    (829, "Gulliver's Travels", "Jonathan Swift"),
    (4705, "A Room of One's Own", "Virginia Woolf"),

    # Drama (dialogue-heavy for EOS prevention)
    (100, "Complete Works of Shakespeare", "William Shakespeare"),
    (1513, "Romeo and Juliet", "William Shakespeare"),
    (1524, "Hamlet", "William Shakespeare"),
    (2267, "Macbeth", "William Shakespeare"),
    (1532, "The Tempest", "William Shakespeare"),

    # Russian Literature (English translations)
    (28054, "The Brothers Karamazov", "Fyodor Dostoevsky"),
    (2554, "Crime and Punishment", "Fyodor Dostoevsky"),
    (600, "Notes from Underground", "Fyodor Dostoevsky"),
    (2600, "War and Peace", "Leo Tolstoy"),
    (1399, "Anna Karenina", "Leo Tolstoy"),

    # More Classics
    (4300, "Ulysses", "James Joyce"),
    (2814, "Dubliners", "James Joyce"),
    (5200, "Metamorphosis", "Franz Kafka"),
    (7849, "The Trial", "Franz Kafka"),
    (16328, "Beowulf", "Anonymous"),
    (120, "Treasure Island", "Robert Louis Stevenson"),
    (43, "The Strange Case of Dr Jekyll and Mr Hyde", "Robert Louis Stevenson"),
    (345, "Dracula", "Bram Stoker"),
    (215, "The Call of the Wild", "Jack London"),
    (1184, "The Count of Monte Cristo", "Alexandre Dumas"),
    (2413, "Meditations", "Marcus Aurelius"),

    # Ancient Religious & Philosophical Texts (Public Domain translations)
    (10, "The King James Bible", "Various"),  # Complete KJV
    (8300, "The Koran (Quran)", "Mohammed (Rodwell translation)"),
    (2680, "The Dhammapada", "Buddha (Müller translation)"),
    (2500, "Siddhartha", "Hermann Hesse"),
    (4363, "The Bhagavad Gita", "Vyasa (Arnold translation)"),
    (7700, "The Tao Te Ching", "Lao Tzu (Legge translation)"),
    (7698, "Analects of Confucius", "Confucius (Legge translation)"),
    (17, "The Book of Mormon", "Joseph Smith"),
    (3296, "The Confessions of St. Augustine", "Augustine"),
    (45626, "The Imitation of Christ", "Thomas à Kempis"),
    (1499, "Thus Spake Zarathustra", "Friedrich Nietzsche"),
    (5827, "The Enchiridion", "Epictetus"),
    (1656, "The Apology of Socrates", "Plato"),
    (1497, "The Republic", "Plato"),
    (1657, "Symposium", "Plato"),
    (6762, "Phaedo", "Plato"),
    (2381, "Poetics", "Aristotle"),
    (8438, "Nicomachean Ethics", "Aristotle"),
    (59, "The Book of Enoch", "Enoch (Charles translation)"),
    (10897, "The Upanishads", "Various (Müller translation)"),

    # Additional Philosophy (Philosophia expansion)
    (1998, "Phaedrus", "Plato"),
    (1600, "Timaeus", "Plato"),
    (1637, "Meno", "Plato"),
    (1658, "Crito", "Plato"),
    (6763, "Laws", "Plato"),
    (1974, "Politics", "Aristotle"),
    (6867, "Metaphysics", "Aristotle"),
    (8932, "On the Soul (De Anima)", "Aristotle"),
    (36887, "Categories", "Aristotle"),
    (69458, "Physics", "Aristotle"),
    (7846, "Critique of Pure Reason", "Immanuel Kant"),
    (4280, "An Enquiry Concerning Human Understanding", "David Hume"),
    (4583, "A Treatise of Human Nature", "David Hume"),
    (10615, "An Essay Concerning Human Understanding", "John Locke"),
    (5669, "Leviathan", "Thomas Hobbes"),
    (37090, "Two Treatises of Government", "John Locke"),
    (7370, "The Ethics", "Baruch Spinoza"),
    (4300, "Beyond Good and Evil", "Friedrich Nietzsche"),
    (52915, "The Birth of Tragedy", "Friedrich Nietzsche"),
    (25717, "On the Genealogy of Morals", "Friedrich Nietzsche"),
    (4363, "The World as Will and Idea", "Arthur Schopenhauer"),
    (10643, "Monadology", "Gottfried Wilhelm Leibniz"),
    (1497, "Being and Time excerpts via Parmenides", "Martin Heidegger via Parmenides"),
    (5001, "Utilitarianism", "John Stuart Mill"),
    (34901, "On Liberty", "John Stuart Mill"),
    (39270, "The Phenomenology of Spirit excerpts", "G.W.F. Hegel via Parmenides"),
    (10662, "Discourse on Method", "René Descartes"),
    (59, "Meditations on First Philosophy", "René Descartes"),
    (55148, "Pensées", "Blaise Pascal"),
    (8578, "The Social Contract", "Jean-Jacques Rousseau"),
    (46333, "Emile", "Jean-Jacques Rousseau"),
]


def download_gutenberg_book(book_id: int, title: str) -> str:
    """Download a book from Project Gutenberg."""
    gutenberg_dir = RAW_DIR / "gutenberg"
    gutenberg_dir.mkdir(exist_ok=True)

    txt_path = gutenberg_dir / f"{book_id}.txt"

    if not txt_path.exists():
        # Try plain text URL
        url = f"{GUTENBERG_MIRROR}/{book_id}/pg{book_id}.txt"
        try:
            urllib.request.urlretrieve(url, txt_path)
        except Exception as e:
            # Try alternate URL format
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            try:
                urllib.request.urlretrieve(url, txt_path)
            except Exception as e2:
                print(f"  Failed to download {title}: {e2}")
                return ""

    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Remove Gutenberg header/footer
    start_markers = ["*** START OF", "***START OF", "*END*THE SMALL PRINT"]
    end_markers = ["*** END OF", "***END OF", "End of Project Gutenberg"]

    for marker in start_markers:
        if marker in text:
            text = text.split(marker, 1)[-1]
            break

    for marker in end_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break

    return clean_text(text)


def process_gutenberg(max_books: int = None) -> Generator[Dict, None, None]:
    """Process Project Gutenberg books."""
    books = GUTENBERG_BOOKS[:max_books] if max_books else GUTENBERG_BOOKS

    for i, (book_id, title, author) in enumerate(books):
        print(f"Processing Gutenberg [{i+1}/{len(books)}] {title}...")

        text = download_gutenberg_book(book_id, title)

        if not text or len(text.split()) < MIN_CHUNK_WORDS:
            continue

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


# ==============================================================================
# Main Build Pipeline
# ==============================================================================

def build_corpus(
    output_path: Path = PROCESSED_DIR / "kssm_corpus.jsonl",
    max_gutenberg: int = None,
    include_openstax: bool = True
):
    """
    Build the complete K-SSM training corpus.

    Primary source: Project Gutenberg (Public Domain)
    - ~20 classic English novels = ~10M tokens
    - No authentication required
    - Clean, high-quality text

    Secondary: OpenStax textbooks (if manually downloaded)
    """
    print("=" * 60)
    print("BUILDING K-SSM TRAINING CORPUS")
    print("=" * 60)
    print(f"Output: {output_path}")
    print()

    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_tokens_approx = 0

    with open(output_path, 'w', encoding='utf-8') as f:

        # Project Gutenberg (Public Domain) - PRIMARY SOURCE
        print("\n[1/2] PROJECT GUTENBERG (Public Domain)")
        print("-" * 40)
        for chunk in process_gutenberg(max_gutenberg):
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            total_chunks += 1
            total_tokens_approx += len(chunk['text'].split()) * 1.3

        print(f"  Gutenberg: {total_chunks} chunks")

        # OpenStax (CC BY 4.0) - if available
        if include_openstax:
            print("\n[2/2] OPENSTAX (CC BY 4.0)")
            print("-" * 40)
            openstax_start = total_chunks
            for chunk in process_openstax():
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                total_chunks += 1
                total_tokens_approx += len(chunk['text'].split()) * 1.3

            print(f"  OpenStax: {total_chunks - openstax_start} chunks")

    # Summary
    print("\n" + "=" * 60)
    print("CORPUS BUILD COMPLETE")
    print("=" * 60)
    print(f"Total chunks: {total_chunks:,}")
    print(f"Approx tokens: {int(total_tokens_approx):,}")
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    return {
        'total_chunks': total_chunks,
        'approx_tokens': int(total_tokens_approx),
        'output_path': str(output_path)
    }


def verify_corpus(corpus_path: Path = PROCESSED_DIR / "kssm_corpus.jsonl"):
    """Verify and print corpus statistics."""
    print("\n" + "=" * 60)
    print("CORPUS VERIFICATION")
    print("=" * 60)

    sources = {}
    total_words = 0
    total_chunks = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            source = chunk['source']
            sources[source] = sources.get(source, 0) + 1
            total_words += len(chunk['text'].split())
            total_chunks += 1

    print(f"\nTotal chunks: {total_chunks:,}")
    print(f"Total words: {total_words:,}")
    print(f"Approx tokens: {int(total_words * 1.3):,}")

    print("\nBy source:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count:,} chunks ({count/total_chunks*100:.1f}%)")

    # Sample
    print("\nSample chunk:")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())
        print(f"  Source: {sample['source']}")
        print(f"  Title: {sample['title']}")
        print(f"  Text: {sample['text'][:200]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build K-SSM training corpus")
    parser.add_argument("--max-gutenberg", type=int, default=None,
                        help="Max Gutenberg books to process")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing corpus")
    args = parser.parse_args()

    if args.verify_only:
        verify_corpus()
    else:
        stats = build_corpus(
            max_gutenberg=args.max_gutenberg
        )
        print("\n✅ Corpus ready for K-SSM training")
        verify_corpus()
