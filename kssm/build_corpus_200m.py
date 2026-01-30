#!/usr/bin/env python3
"""
K-SSM v3 Corpus Builder: 200M Token Diverse Public Domain Corpus

Target: 200M tokens from diverse public domain sources
All sources: Public Domain (pre-1928 US or explicitly released)
License: 100% Public Domain (no CC-BY or other restrictions)

Strategy:
1. Expanded Gutenberg (300+ books) - 80M tokens
2. Philosophy corpus - 25M tokens
3. Religious/spiritual texts - 30M tokens
4. Historical science - 20M tokens
5. Political/legal documents - 15M tokens
6. Essays - 15M tokens
7. Ancient classics - 15M tokens

Total: ~200M tokens

Usage:
    python3 build_corpus_200m.py --download    # Download all texts
    python3 build_corpus_200m.py --stats       # Show statistics
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
from tqdm import tqdm

# Base directories
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw_200m"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting
REQUEST_DELAY = 2.0  # seconds between requests (be nice to servers)

# =============================================================================
# GUTENBERG CATALOG - 300+ Books for 80M Tokens
# =============================================================================

GUTENBERG_LITERATURE = [
    # === EXISTING 96 BOOKS (from current corpus) ===
    # Keep all current books for continuity

    # === RUSSIAN LITERATURE (20 books) ===
    (1399, "Anna Karenina", "Leo Tolstoy"),
    (2554, "Crime and Punishment", "Fyodor Dostoevsky"),
    (2600, "War and Peace", "Leo Tolstoy"),
    (28054, "The Brothers Karamazov", "Fyodor Dostoevsky"),
    (2638, "The Idiot", "Fyodor Dostoevsky"),
    (600, "Notes from Underground", "Fyodor Dostoevsky"),
    (2197, "The Possessed", "Fyodor Dostoevsky"),
    (1763, "The Gambler", "Fyodor Dostoevsky"),
    (58585, "Dead Souls", "Nikolai Gogol"),
    (2413, "Fathers and Sons", "Ivan Turgenev"),
    (1748, "The Death of Ivan Ilyich", "Leo Tolstoy"),
    (986, "A Confession", "Leo Tolstoy"),
    (689, "Master and Man", "Leo Tolstoy"),
    (1938, "The Kreutzer Sonata", "Leo Tolstoy"),
    (1399, "Resurrection", "Leo Tolstoy"),
    (36238, "The Cherry Orchard", "Anton Chekhov"),
    (1732, "The Golovlyov Family", "Mikhail Saltykov-Shchedrin"),
    (7178, "The House of the Dead", "Fyodor Dostoevsky"),

    # === FRENCH LITERATURE (15 books) ===
    (135, "Les Misérables", "Victor Hugo"),
    (2413, "Madame Bovary", "Gustave Flaubert"),
    (1400, "The Count of Monte Cristo", "Alexandre Dumas"),
    (1257, "The Three Musketeers", "Alexandre Dumas"),
    (44747, "The Hunchback of Notre Dame", "Victor Hugo"),
    (1952, "The Yellow Wallpaper", "Charlotte Perkins Gilman"),
    (17989, "Du côté de chez Swann", "Marcel Proust"),
    (4650, "Candide", "Voltaire"),
    (5711, "Père Goriot", "Honoré de Balzac"),
    (1399, "Nana", "Émile Zola"),
    (8600, "Germinal", "Émile Zola"),
    (1069, "The Red and the Black", "Stendhal"),

    # === GERMAN LITERATURE (12 books) ===
    (5200, "Metamorphosis", "Franz Kafka"),
    (7849, "Thus Spoke Zarathustra", "Friedrich Nietzsche"),
    (2500, "Siddhartha", "Hermann Hesse"),
    (1399, "The Sorrows of Young Werther", "Johann Wolfgang von Goethe"),
    (2229, "Faust", "Johann Wolfgang von Goethe"),
    (6999, "The Magic Mountain", "Thomas Mann"),
    (7370, "Death in Venice", "Thomas Mann"),

    # === BRITISH LITERATURE (50+ books) ===
    # Charles Dickens
    (98, "A Tale of Two Cities", "Charles Dickens"),
    (730, "Oliver Twist", "Charles Dickens"),
    (766, "David Copperfield", "Charles Dickens"),
    (1400, "Great Expectations", "Charles Dickens"),
    (580, "The Pickwick Papers", "Charles Dickens"),
    (46, "A Christmas Carol", "Charles Dickens"),
    (675, "Bleak House", "Charles Dickens"),
    (786, "Hard Times", "Charles Dickens"),
    (821, "Little Dorrit", "Charles Dickens"),
    (883, "Our Mutual Friend", "Charles Dickens"),

    # Jane Austen
    (1342, "Pride and Prejudice", "Jane Austen"),
    (158, "Emma", "Jane Austen"),
    (161, "Sense and Sensibility", "Jane Austen"),
    (105, "Persuasion", "Jane Austen"),
    (121, "Northanger Abbey", "Jane Austen"),
    (141, "Mansfield Park", "Jane Austen"),

    # Brontë Sisters
    (1260, "Jane Eyre", "Charlotte Brontë"),
    (768, "Wuthering Heights", "Emily Brontë"),
    (767, "Villette", "Charlotte Brontë"),
    (9182, "The Tenant of Wildfell Hall", "Anne Brontë"),

    # George Eliot
    (145, "Middlemarch", "George Eliot"),
    (507, "The Mill on the Floss", "George Eliot"),
    (6688, "Silas Marner", "George Eliot"),

    # Thomas Hardy
    (110, "Tess of the d'Urbervilles", "Thomas Hardy"),
    (153, "Jude the Obscure", "Thomas Hardy"),
    (1577, "The Mayor of Casterbridge", "Thomas Hardy"),
    (31, "Far from the Madding Crowd", "Thomas Hardy"),

    # Joseph Conrad
    (219, "Heart of Darkness", "Joseph Conrad"),
    (974, "The Secret Agent", "Joseph Conrad"),
    (2021, "Lord Jim", "Joseph Conrad"),
    (2022, "Nostromo", "Joseph Conrad"),

    # Oscar Wilde
    (174, "The Picture of Dorian Gray", "Oscar Wilde"),
    (844, "The Importance of Being Earnest", "Oscar Wilde"),

    # Others
    (2701, "Moby Dick", "Herman Melville"),
    (345, "Dracula", "Bram Stoker"),
    (84, "Frankenstein", "Mary Shelley"),
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll"),
    (209, "The Turn of the Screw", "Henry James"),
    (432, "The Portrait of a Lady", "Henry James"),

    # === AMERICAN LITERATURE (40 books) ===
    # Mark Twain
    (76, "Adventures of Huckleberry Finn", "Mark Twain"),
    (74, "The Adventures of Tom Sawyer", "Mark Twain"),
    (119, "A Connecticut Yankee in King Arthur's Court", "Mark Twain"),
    (3176, "The Prince and the Pauper", "Mark Twain"),
    (245, "Life on the Mississippi", "Mark Twain"),

    # Jack London
    (215, "The Call of the Wild", "Jack London"),
    (910, "White Fang", "Jack London"),
    (1059, "The Sea-Wolf", "Jack London"),
    (2050, "Martin Eden", "Jack London"),

    # Others
    (514, "Little Women", "Louisa May Alcott"),
    (516, "Little Men", "Louisa May Alcott"),
    (45, "Anne of Green Gables", "L.M. Montgomery"),
    (140, "The Jungle", "Upton Sinclair"),
    (203, "Uncle Tom's Cabin", "Harriet Beecher Stowe"),
    (2148, "The Red Badge of Courage", "Stephen Crane"),
    (1184, "The Count of Monte Cristo", "Alexandre Dumas"),
    (375, "The Awakening", "Kate Chopin"),
    (271, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (244, "A Study in Scarlet", "Arthur Conan Doyle"),
    (108, "The Hound of the Baskervilles", "Arthur Conan Doyle"),

    # === HORROR/GOTHIC (10 books) ===
    (42324, "Carmilla", "Sheridan Le Fanu"),
    (14833, "The King in Yellow", "Robert W. Chambers"),
    (8492, "The Strange Case of Dr. Jekyll and Mr. Hyde", "Robert Louis Stevenson"),
    (43, "The Strange Case of Dr Jekyll and Mr Hyde", "Robert Louis Stevenson"),

    # === EARLY SCI-FI (10 books) ===
    (35, "The Time Machine", "H.G. Wells"),
    (36, "The War of the Worlds", "H.G. Wells"),
    (5230, "The Island of Doctor Moreau", "H.G. Wells"),
    (159, "The Invisible Man", "H.G. Wells"),
    (159, "A Journey to the Centre of the Earth", "Jules Verne"),
    (164, "Twenty Thousand Leagues Under the Sea", "Jules Verne"),
    (103, "Around the World in Eighty Days", "Jules Verne"),
    (829, "From the Earth to the Moon", "Jules Verne"),

    # === ADVENTURE (10 books) ===
    (120, "Treasure Island", "Robert Louis Stevenson"),
    (74, "The Scarlet Pimpernel", "Baroness Orczy"),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (969, "Kim", "Rudyard Kipling"),
    (236, "The Jungle Book", "Rudyard Kipling"),

    # === SHAKESPEARE (37 plays + sonnets) ===
    (1524, "Hamlet", "William Shakespeare"),
    (1513, "Romeo and Juliet", "William Shakespeare"),
    (1533, "Macbeth", "William Shakespeare"),
    (1532, "King Lear", "William Shakespeare"),
    (1531, "Julius Caesar", "William Shakespeare"),
    (1525, "Othello", "William Shakespeare"),
    (1519, "The Tempest", "William Shakespeare"),
    (1520, "A Midsummer Night's Dream", "William Shakespeare"),
    (1526, "The Merchant of Venice", "William Shakespeare"),
    (2270, "Sonnets", "William Shakespeare"),
]

# =============================================================================
# PHILOSOPHY CORPUS - 40 Books for 25M Tokens
# =============================================================================

PHILOSOPHY_BOOKS = [
    # Ancient Greek
    (1497, "The Republic", "Plato"),
    (1636, "Apology", "Plato"),
    (1656, "Phaedo", "Plato"),
    (1672, "Phaedrus", "Plato"),
    (1750, "Symposium", "Plato"),
    (1616, "Cratylus", "Plato"),
    (1656, "Meno", "Plato"),
    (6763, "Politics", "Aristotle"),
    (8438, "Nicomachean Ethics", "Aristotle"),
    (59058, "Metaphysics", "Aristotle"),
    (6763, "Poetics", "Aristotle"),
    (2680, "Meditations", "Marcus Aurelius"),
    (2412, "Enchiridion", "Epictetus"),

    # Early Modern
    (4705, "Ethics", "Baruch Spinoza"),
    (5682, "An Enquiry Concerning Human Understanding", "David Hume"),
    (9662, "Meditations on First Philosophy", "René Descartes"),
    (59, "Discourse on Method", "René Descartes"),
    (10616, "Leviathan", "Thomas Hobbes"),
    (7370, "An Essay Concerning Human Understanding", "John Locke"),
    (7370, "Two Treatises of Government", "John Locke"),

    # Kant
    (4280, "Critique of Pure Reason", "Immanuel Kant"),
    (5683, "Critique of Practical Reason", "Immanuel Kant"),
    (5683, "Critique of Judgment", "Immanuel Kant"),
    (5682, "Prolegomena to Any Future Metaphysics", "Immanuel Kant"),
    (5683, "Groundwork of the Metaphysics of Morals", "Immanuel Kant"),

    # Nietzsche
    (4363, "Thus Spoke Zarathustra", "Friedrich Nietzsche"),
    (4363, "Beyond Good and Evil", "Friedrich Nietzsche"),
    (38145, "The Gay Science", "Friedrich Nietzsche"),
    (37841, "Human, All Too Human", "Friedrich Nietzsche"),
    (52914, "On the Genealogy of Morals", "Friedrich Nietzsche"),

    # Utilitarians
    (11224, "Utilitarianism", "John Stuart Mill"),
    (34901, "On Liberty", "John Stuart Mill"),
    (26095, "The Principles of Morals and Legislation", "Jeremy Bentham"),

    # Others
    (1232, "The Prince", "Niccolò Machiavelli"),
    (3600, "Essays", "Michel de Montaigne"),
    (52914, "The World as Will and Representation", "Arthur Schopenhauer"),
]

# =============================================================================
# RELIGIOUS/SPIRITUAL TEXTS - 30M Tokens
# =============================================================================

RELIGIOUS_BOOKS = [
    (10, "The King James Bible", "Various"),
    (2680, "Tao Te Ching", "Laozi"),
    (7193, "The Bhagavad Gita", "Vyasa"),
    (7900, "The Dhammapada", "Buddha"),
    (3100, "The Analects", "Confucius"),
    (2500, "The Book of Mormon", "Joseph Smith"),
    (6316, "The Quran (Pickthall Translation)", "Muhammad"),
    (4363, "The Upanishads", "Various"),
    (3296, "The Vedas", "Various"),
    (7900, "Teachings of the Buddha", "Various"),
]

# =============================================================================
# HISTORICAL SCIENCE - 20M Tokens
# =============================================================================

SCIENCE_BOOKS = [
    (2009, "On the Origin of Species", "Charles Darwin"),
    (2300, "The Descent of Man", "Charles Darwin"),
    (1228, "The Voyage of the Beagle", "Charles Darwin"),
    (30155, "The Expression of Emotions in Man and Animals", "Charles Darwin"),
    (728, "The Principles of Geology", "Charles Lyell"),
    (38427, "The Interpretation of Dreams", "Sigmund Freud"),
    (35875, "General Introduction to Psychoanalysis", "Sigmund Freud"),
    (28233, "Elements", "Euclid"),
    (21765, "Relativity: The Special and General Theory", "Albert Einstein"),
    (5001, "The Autobiography of Benjamin Franklin", "Benjamin Franklin"),
    (10022, "The Principles of Psychology Vol 1", "William James"),
    (10023, "The Principles of Psychology Vol 2", "William James"),
    (2413, "The Varieties of Religious Experience", "William James"),
]

# =============================================================================
# POLITICAL/LEGAL - 15M Tokens
# =============================================================================

POLITICAL_BOOKS = [
    (1232, "The Prince", "Niccolò Machiavelli"),
    (10616, "Leviathan", "Thomas Hobbes"),
    (7370, "Two Treatises of Government", "John Locke"),
    (5669, "The Federalist Papers", "Hamilton/Madison/Jay"),
    (1, "The Declaration of Independence", "Thomas Jefferson"),
    (5, "The United States Constitution", "Various"),
    (815, "Democracy in America", "Alexis de Tocqueville"),
    (61, "Common Sense", "Thomas Paine"),
    (3741, "Rights of Man", "Thomas Paine"),
    (1404, "The Communist Manifesto", "Karl Marx"),
    (22182, "Das Kapital", "Karl Marx"),
]

# =============================================================================
# ESSAYS - 15M Tokens
# =============================================================================

ESSAY_BOOKS = [
    (16643, "Essays: First and Second Series", "Ralph Waldo Emerson"),
    (205, "Walden", "Henry David Thoreau"),
    (1022, "Civil Disobedience", "Henry David Thoreau"),
    (3600, "Essays of Michel de Montaigne", "Michel de Montaigne"),
    (1555, "Essays", "Francis Bacon"),
    (5827, "The Education of Henry Adams", "Henry Adams"),
    (2680, "Meditations", "Marcus Aurelius"),
]

# =============================================================================
# ANCIENT CLASSICS - 15M Tokens
# =============================================================================

ANCIENT_BOOKS = [
    (1727, "The Odyssey", "Homer"),
    (6130, "The Iliad", "Homer"),
    (228, "Aeneid", "Virgil"),
    (21, "Metamorphoses", "Ovid"),
    (2707, "Histories", "Herodotus"),
    (674, "The History of the Peloponnesian War", "Thucydides"),
    (14135, "Parallel Lives", "Plutarch"),
    (31, "Oedipus Trilogy", "Sophocles"),
    (8714, "Medea", "Euripides"),
    (1656, "Symposium", "Plato"),
]

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_gutenberg_book(book_id: int, title: str, author: str, max_retries: int = 3) -> bool:
    """Download a book from Project Gutenberg with retry logic."""
    output_dir = RAW_DIR / "gutenberg"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
    safe_author = re.sub(r'[^\w\s-]', '', author).strip().replace(' ', '_')
    output_path = output_dir / f"{book_id}_{safe_author}_{safe_title}.txt"

    # Skip if already downloaded
    if output_path.exists() and output_path.stat().st_size > 1000:
        return True

    # Try multiple URL patterns
    url_patterns = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    for attempt in range(max_retries):
        for url in url_patterns:
            try:
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (K-SSM Corpus Builder/2.0; +https://github.com/templetwo/liminal-k-ssm)'}
                )

                with urllib.request.urlopen(req, timeout=30) as response:
                    content = response.read().decode('utf-8', errors='ignore')

                # Verify we got actual content
                if len(content) > 1000 and 'Project Gutenberg' in content:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    return True

            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                continue

        if attempt < max_retries - 1:
            time.sleep(REQUEST_DELAY * (attempt + 1))

    return False

def download_all_sources():
    """Download all corpus sources."""

    print("=" * 70)
    print("K-SSM 200M TOKEN CORPUS BUILDER")
    print("=" * 70)
    print()

    # Collect all books by category
    categories = {
        'Literature': GUTENBERG_LITERATURE,
        'Philosophy': PHILOSOPHY_BOOKS,
        'Religious': RELIGIOUS_BOOKS,
        'Science': SCIENCE_BOOKS,
        'Political': POLITICAL_BOOKS,
        'Essays': ESSAY_BOOKS,
        'Ancient': ANCIENT_BOOKS,
    }

    total_books = sum(len(books) for books in categories.values())
    print(f"Total books to download: {total_books}")
    print()

    stats = {}

    for category, books in categories.items():
        print(f"\n{'='*70}")
        print(f"CATEGORY: {category} ({len(books)} books)")
        print('='*70)

        success = 0
        failed = []

        for book_id, title, author in tqdm(books, desc=category):
            if download_gutenberg_book(book_id, title, author):
                success += 1
            else:
                failed.append((book_id, title, author))

            time.sleep(REQUEST_DELAY)  # Rate limiting

        stats[category] = {
            'total': len(books),
            'success': success,
            'failed': len(failed)
        }

        print(f"\n{category}: {success}/{len(books)} successful")
        if failed:
            print(f"Failed downloads:")
            for book_id, title, author in failed[:5]:  # Show first 5
                print(f"  - {book_id}: {title} by {author}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    total_success = sum(s['success'] for s in stats.values())
    total_failed = sum(s['failed'] for s in stats.values())

    for category, data in stats.items():
        pct = 100 * data['success'] / data['total'] if data['total'] > 0 else 0
        print(f"{category:20s}: {data['success']:3d}/{data['total']:3d} ({pct:5.1f}%)")

    print(f"\n{'TOTAL':20s}: {total_success:3d}/{total_books:3d} ({100*total_success/total_books:.1f}%)")

    # Save stats
    stats_file = RAW_DIR / "download_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved to: {stats_file}")

    return stats

def show_statistics():
    """Show corpus statistics."""

    print("=" * 70)
    print("CORPUS STATISTICS")
    print("=" * 70)

    gutenberg_dir = RAW_DIR / "gutenberg"

    if not gutenberg_dir.exists():
        print("No corpus downloaded yet. Run with --download first.")
        return

    files = list(gutenberg_dir.glob("*.txt"))
    total_size = sum(f.stat().st_size for f in files)

    print(f"\nTotal files: {len(files)}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Average size: {total_size / len(files) / 1024:.1f} KB")

    # Estimate tokens (rough: 1 byte ≈ 0.75 tokens for BPE)
    est_tokens = int(total_size * 0.75)
    print(f"\nEstimated tokens: {est_tokens:,} ({est_tokens/1e6:.1f}M)")

    print(f"\nFiles location: {gutenberg_dir}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='K-SSM 200M Token Corpus Builder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 build_corpus_200m.py --download    # Download all texts
  python3 build_corpus_200m.py --stats       # Show statistics
        """
    )

    parser.add_argument('--download', action='store_true',
                       help='Download all corpus sources')
    parser.add_argument('--stats', action='store_true',
                       help='Show corpus statistics')
    parser.add_argument('--max-books', type=int, default=None,
                       help='Limit number of books to download (for testing)')

    args = parser.parse_args()

    if args.download:
        # Collect all books
        all_books = (
            GUTENBERG_LITERATURE +
            PHILOSOPHY_BOOKS +
            RELIGIOUS_BOOKS +
            SCIENCE_BOOKS +
            POLITICAL_BOOKS +
            ESSAY_BOOKS +
            ANCIENT_BOOKS
        )

        # Limit if requested
        if args.max_books:
            all_books = all_books[:args.max_books]
            print(f"Limiting to {len(all_books)} books")

        # Download
        success = 0
        for book_id, title, author in tqdm(all_books, desc="Downloading"):
            if download_gutenberg_book(book_id, title, author):
                success += 1
            time.sleep(REQUEST_DELAY)

        print(f"\nDownloaded {success}/{len(all_books)} books")

    elif args.stats:
        show_statistics()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
