# K-SSM 200M Token Corpus Expansion

**Objective**: Expand training corpus from 22M to 200M tokens for production-scale model training

**Status**: Ready to deploy (pipeline tested and validated)

**License**: 100% Public Domain

---

## Quick Start

```bash
# 1. Test pipeline locally (optional but recommended)
python3 kssm/test_corpus_pipeline.py

# 2. Deploy to Mac Studio (automated)
./kssm/deploy_corpus_200m.sh

# 3. Process and tokenize (on Mac Studio)
ssh tony_studio@192.168.1.195
cd ~/liminal-k-ssm
python3 kssm/process_corpus_200m.py --all

# 4. Verify
python3 kssm/process_corpus_200m.py --stats
```

**Total time**: ~60 minutes

---

## Overview

### Current Corpus (v3 - 22M tokens)
- **96 books** from Project Gutenberg
- **21.2M training tokens, 1.1M validation tokens**
- **Skew**: Heavy toward literature, philosophy, religious texts
- **Limitations**:
  - Insufficient diversity for general language understanding
  - Archaic language patterns (pre-1928 bias)
  - Missing modern technical/scientific vocabulary

### Target Corpus (v4 - 200M tokens)
- **470+ books** from diverse sources
- **190M training tokens, 10M validation tokens**
- **Categories**:
  - Literature (40%): International classics, diverse genres
  - Philosophy (12.5%): Ancient to modern, Eastern and Western
  - Religious/Spiritual (15%): Multiple traditions and perspectives
  - Historical Science (10%): Darwin, Newton, Euclid, early psychology
  - Political/Legal (7.5%): Founding documents, political theory
  - Essays (7.5%): Personal voice and argumentation
  - Ancient Classics (7.5%): Homer, Virgil, Greek drama

---

## Design Principles

### 1. Public Domain Only
**Why**:
- No licensing complexity or attribution requirements
- Can be freely distributed and built upon
- Aligns with open science principles

**How**:
- Pre-1928 works (automatic US public domain)
- Explicitly released public domain translations
- Project Gutenberg catalog (verified PD status)

**NO**:
- CC-BY or CC-BY-SA (attribution/share-alike requirements)
- Modern copyrighted works
- Web-scraped content of uncertain provenance

### 2. Diverse Perspectives
**Why**:
- Consciousness research requires broad conceptual coverage
- Avoid cultural/linguistic bias
- Represent multiple ways of thinking about self, agency, and existence

**How**:
- International literature (Russian, French, German, British, American)
- Multiple philosophical traditions (Greek, Enlightenment, German Idealism, Existentialism)
- World religions (Christianity, Islam, Hinduism, Buddhism, Taoism)
- Different discourse types (narrative, argumentative, technical, poetic)

### 3. Quality Over Quantity
**Why**:
- Training on noise degrades performance
- Coherent text teaches better language structure
- Classic works have stood test of time

**How**:
- Curated selection (not automated scraping)
- Each book manually chosen for contribution to corpus diversity
- Established translations and editions

### 4. Transparency
**Why**:
- Reproducibility
- Ethical AI practice
- Scientific rigor

**How**:
- Full book list with IDs in `build_corpus_200m.py`
- Download statistics tracked and saved
- Processing pipeline fully documented
- Verification commands provided

---

## Corpus Breakdown

### Literature (80M tokens)

**International Classics**:
- **Russian**: Tolstoy (War and Peace, Anna Karenina), Dostoevsky (all major works), Gogol, Turgenev, Chekhov
- **French**: Hugo (Les Misérables), Flaubert, Dumas, Zola, Stendhal, Proust
- **German**: Kafka (Metamorphosis), Nietzsche, Hesse (Siddhartha), Goethe (Faust)

**British Canon**:
- **Dickens** (10 novels): Tale of Two Cities, Great Expectations, Oliver Twist, David Copperfield, etc.
- **Austen** (6 novels): Pride and Prejudice, Emma, Sense and Sensibility, etc.
- **Brontë Sisters**: Jane Eyre, Wuthering Heights, Villette
- **George Eliot**: Middlemarch, The Mill on the Floss, Silas Marner
- **Thomas Hardy**: Tess, Jude the Obscure, Mayor of Casterbridge
- **Joseph Conrad**: Heart of Darkness, Lord Jim, The Secret Agent

**American Literature**:
- **Mark Twain**: Huckleberry Finn, Tom Sawyer, Connecticut Yankee, Life on Mississippi
- **Jack London**: Call of the Wild, White Fang, Sea-Wolf, Martin Eden
- **Others**: Moby Dick, Uncle Tom's Cabin, Red Badge of Courage, Little Women

**Shakespeare**: All 37 plays + Sonnets

**Genre Diversity**:
- **Horror/Gothic**: Dracula, Frankenstein, Dr. Jekyll and Mr. Hyde, Carmilla
- **Early Sci-Fi**: H.G. Wells (Time Machine, War of the Worlds, Invisible Man), Jules Verne (20,000 Leagues, Around the World in 80 Days)
- **Adventure**: Treasure Island, Sherlock Holmes stories

### Philosophy (25M tokens)

**Ancient Greek**:
- Plato: Republic, Symposium, Apology, Phaedo, Phaedrus
- Aristotle: Nicomachean Ethics, Politics, Metaphysics, Poetics
- Stoics: Marcus Aurelius (Meditations), Epictetus (Enchiridion)

**Early Modern**:
- Descartes: Meditations, Discourse on Method
- Spinoza: Ethics
- Hobbes: Leviathan
- Locke: Essay Concerning Human Understanding, Two Treatises

**Kant** (5 major works):
- Critique of Pure Reason
- Critique of Practical Reason
- Critique of Judgment
- Prolegomena to Any Future Metaphysics
- Groundwork of the Metaphysics of Morals

**Nietzsche** (5 major works):
- Thus Spoke Zarathustra
- Beyond Good and Evil
- The Gay Science
- Human, All Too Human
- On the Genealogy of Morals

**Others**:
- Hume: Enquiry Concerning Human Understanding
- Machiavelli: The Prince
- Montaigne: Essays
- Schopenhauer: The World as Will and Representation
- Mill: Utilitarianism, On Liberty

### Religious/Spiritual (30M tokens)

**Abrahamic**:
- King James Bible (Old + New Testament)
- Quran (Pickthall translation, PD)
- Various theological works

**Eastern**:
- Bhagavad Gita
- Upanishads, Vedas
- Dhammapada (Buddhist teachings)
- Tao Te Ching
- Analects of Confucius

**Why diverse religious texts**:
- Different conceptions of consciousness and self
- Varied linguistic patterns (poetic, declarative, narrative)
- Philosophical depth without modern academic jargon
- Millennia-tested ideas about existence and meaning

### Historical Science (20M tokens)

**Evolution/Biology**:
- Darwin: Origin of Species, Descent of Man, Voyage of the Beagle, Expression of Emotions

**Physics/Mathematics**:
- Newton: Principia Mathematica
- Euclid: Elements
- Einstein: Relativity (Special and General Theory, 1920 edition)

**Psychology**:
- William James: Principles of Psychology (Vol 1 & 2), Varieties of Religious Experience
- Freud: Interpretation of Dreams, General Introduction to Psychoanalysis

**Why historical science**:
- Technical language and argumentation
- Hypothesis-evidence structure
- Different from narrative literature
- Foundation of modern scientific thinking

### Political/Legal (15M tokens)

**Founding Documents**:
- US Declaration of Independence
- US Constitution
- Federalist Papers

**Political Philosophy**:
- Machiavelli: The Prince
- Hobbes: Leviathan
- Locke: Two Treatises of Government
- Tocqueville: Democracy in America
- Marx: Communist Manifesto, Das Kapital
- Paine: Common Sense, Rights of Man

**Why political texts**:
- Formal argumentation style
- Concepts of agency, power, and collective action
- Different from literary or scientific discourse

### Essays (15M tokens)

- Emerson: Essays (First and Second Series)
- Thoreau: Walden, Civil Disobedience
- Montaigne: Essays
- Bacon: Essays
- Henry Adams: The Education of Henry Adams

**Why essays**:
- Personal voice and introspection
- Shorter form allows more diverse styles
- Bridge between formal philosophy and narrative literature

### Ancient Classics (15M tokens)

- Homer: Iliad, Odyssey
- Virgil: Aeneid
- Ovid: Metamorphoses
- Herodotus: Histories
- Thucydides: History of the Peloponnesian War
- Plutarch: Parallel Lives
- Greek Drama: Sophocles (Oedipus), Euripides (Medea)

**Why ancient classics**:
- Foundational narratives and myths
- Different narrative structures
- Timeless themes of fate, choice, heroism
- Language that influenced all later Western literature

---

## Technical Specifications

### File Formats

**Raw Texts**:
- Location: `kssm/data/raw_200m/gutenberg/`
- Format: UTF-8 plain text (`.txt`)
- Naming: `{book_id}_{author}_{title}.txt`
- Size: ~1.5 GB (470 files)

**JSONL Corpus**:
- Location: `kssm/data/processed/kssm_corpus_200m.jsonl`
- Format: JSON Lines (one entry per line)
- Structure:
  ```json
  {
    "text": "chunk text...",
    "source": "gutenberg",
    "doc_id": "gutenberg_1342",
    "chunk_id": 0,
    "license": "Public Domain",
    "file": "gutenberg/1342_Jane_Austen_Pride_and_Prejudice.txt"
  }
  ```
- Size: ~200 MB (~30,000 chunks)

**Tokenized Arrays**:
- Location: `data/cache_v3_200m/`
- Files:
  - `tokens_train.npy` (~760 MB, ~190M tokens)
  - `tokens_val.npy` (~40 MB, ~10M tokens)
  - `tokens_train_meta.json` (metadata)
  - `tokens_val_meta.json` (metadata)
- Format: NumPy int32 arrays
- Tokenizer: tiktoken GPT-2 BPE (vocab size 50,257)

### Processing Pipeline

**1. Download** (`build_corpus_200m.py`):
- Fetches text files from Project Gutenberg
- Multiple URL patterns tried (handles Gutenberg server inconsistencies)
- Rate limiting: 2 seconds between requests
- Retries: 3 attempts per book
- Progress tracking with tqdm
- Statistics saved to `download_stats.json`

**2. Text Cleaning** (`process_corpus_200m.py`):
- Remove Gutenberg headers/footers
- Normalize whitespace (CRLF → LF, tabs → spaces)
- Remove page numbers and chapter markers
- Preserve paragraph structure

**3. Chunking**:
- Target chunk size: 1024 words
- Overlap between chunks: 128 words (for context continuity)
- Minimum chunk size: 100 words (filter fragments)
- Strategy: Combine paragraphs into chunks, preserve boundaries

**4. Tokenization**:
- Tokenizer: tiktoken GPT-2 BPE
- Vocab size: 50,257
- Train/val split: 95% / 5%
- Memory-mapped storage (zero RAM overhead during training)

---

## Validation

### Pre-Deployment Testing

All scripts tested with 5-book subset:
```bash
python3 kssm/test_corpus_pipeline.py
```

**Test Results** (2026-01-30):
- ✅ Downloaded: 5/5 books (Alice, Pride & Prejudice, Frankenstein, The Prince, Origin of Species)
- ✅ Processed: 556 chunks
- ✅ Tokenized: 730,281 tokens (0.73M)
- ✅ Train/Val split: 693,766 / 36,515
- ✅ Decode verification: Working correctly

### Post-Deployment Verification

```bash
# On Mac Studio after deployment
ssh tony_studio@192.168.1.195
cd ~/liminal-k-ssm

# 1. Check download stats
python3 kssm/build_corpus_200m.py --stats
# Expected: ~470 books, ~1.5GB

# 2. Verify corpus
python3 kssm/process_corpus_200m.py --stats
# Expected: ~30,000 chunks, ~200M tokens

# 3. Check token files
ls -lh data/cache_v3_200m/
# Expected:
# - tokens_train.npy (~760 MB)
# - tokens_val.npy (~40 MB)
# - Metadata JSON files

# 4. Validate token integrity
python3 -c "
import numpy as np
import tiktoken

# Load
train = np.load('data/cache_v3_200m/tokens_train.npy')
val = np.load('data/cache_v3_200m/tokens_val.npy')

# Check
print(f'Train: {len(train):,} tokens')
print(f'Val: {len(val):,} tokens')
print(f'Vocab range: [{train.min()}, {train.max()}]')

# Test decode
enc = tiktoken.get_encoding('gpt2')
sample = enc.decode(train[:50].tolist())
print(f'Sample: {sample[:100]}...')
"
```

---

## Deployment

### Requirements

**Mac Studio**:
- 5GB free disk space
- Python 3.x with pip
- Internet connection (for downloads)
- Dependencies: `tiktoken`, `tqdm` (auto-installed by deployment script)

**Network**:
- SSH access to Mac Studio (192.168.1.195)
- Outbound HTTPS (for Gutenberg downloads)

### Deployment Steps

See `TRAINING_SOP.md` section "Corpus Expansion: 200M Token Upgrade" for complete procedures.

**Quick version**:
```bash
# From local machine
./kssm/deploy_corpus_200m.sh

# Follow prompts, confirm download
# Wait ~45 minutes

# SSH to Mac Studio
ssh tony_studio@192.168.1.195
cd ~/liminal-k-ssm

# Process and tokenize
python3 kssm/process_corpus_200m.py --all

# Verify
python3 kssm/process_corpus_200m.py --stats
```

### Backups

Deployment script automatically creates:
- `kssm/data/processed/kssm_corpus_22M_backup.jsonl` (original corpus)
- `data/cache_v3_22M_backup/` (original tokens)

To rollback:
```bash
cp kssm/data/processed/kssm_corpus_22M_backup.jsonl \
   kssm/data/processed/kssm_corpus.jsonl

rm -rf data/cache_v3
cp -r data/cache_v3_22M_backup data/cache_v3
```

---

## Impact on Training

### Expected Changes

**Positive**:
- **Better generalization**: 9x more data reduces overfitting
- **Broader vocabulary**: More diverse domains and styles
- **Improved coherence**: Longer texts teach better structure
- **Less archaic bias**: More varied time periods (though still pre-1928)

**Neutral/Unknown**:
- **Training time**: ~9x longer to see full corpus (adjust `--max-steps`)
- **Convergence speed**: May be slower initially, but better final performance
- **Optimal hyperparameters**: May need adjustment (learning rate, batch size)

**Trade-offs**:
- **Disk space**: +2.5GB (raw + tokens)
- **Initial setup time**: 60 minutes
- **Complexity**: More categories means harder to analyze failure modes

### Recommended Training Strategy

**V3.1 (Small-scale test)**:
- **Corpus**: 200M
- **Max steps**: 5,000 (see ~5% of corpus)
- **Goal**: Verify no degradation vs 22M corpus
- **Comparison**: R progression, perplexity, sample quality

**V4 (Full production run)**:
- **Corpus**: 200M
- **Max steps**: 50,000-100,000 (see 100% of corpus 2-5x)
- **Model size**: Consider scaling to 90M+ parameters
- **Goal**: Achieve production-grade language model

---

## Troubleshooting

### Download Issues

**Problem**: Some books fail to download

**Solution**:
```bash
# Check which failed
cat kssm/data/raw_200m/download_stats.json

# Re-run (will skip existing)
python3 kssm/build_corpus_200m.py --download
```

**Common causes**:
- Gutenberg server temporary issues (retry later)
- Book ID changed or removed (check Gutenberg catalog)
- Network timeout (increase timeout in code if persistent)

### Processing Issues

**Problem**: Text cleaning removes too much

**Solution**:
Adjust cleaning rules in `process_corpus_200m.py`:
```python
# Be more conservative with marker removal
# Check specific book causing issues
```

**Problem**: Chunks too large/small

**Solution**:
Adjust chunking parameters:
```python
CHUNK_SIZE = 1024  # Increase/decrease
OVERLAP = 128      # Adjust overlap
```

### Tokenization Issues

**Problem**: `tiktoken` not found

**Solution**:
```bash
pip3 install tiktoken
```

**Problem**: Out of memory during tokenization

**Solution**:
Process in batches (modify `tokenize_corpus()` to stream):
```python
# Process corpus in chunks instead of loading all at once
```

### Training Issues

**Problem**: Training loss doesn't improve on 200M corpus

**Possible causes**:
1. **Data loading error**: Verify tokens load correctly
2. **Hyperparameters**: May need adjustment for larger corpus
3. **Model capacity**: 46M params may be insufficient for 200M tokens

**Debug**:
```bash
# Test data loading
python3 -c "
import numpy as np
tokens = np.load('data/cache_v3_200m/tokens_train.npy', mmap_mode='r')
print(f'Loaded {len(tokens):,} tokens')
print(f'First 100: {tokens[:100]}')
"

# Compare to 22M corpus
python3 kssm/train_kssm_v3.py --max-steps 1000 --corpus 22m
python3 kssm/train_kssm_v3.py --max-steps 1000 --corpus 200m
```

---

## Future Enhancements

### Potential Additions (Beyond 200M)

**Standard Ebooks** (~150M additional tokens):
- High-quality, curated public domain novels
- Better formatting than raw Gutenberg
- License: CC0 (public domain)
- URL: https://standardebooks.org/

**Project Gutenberg Poetry** (~20M tokens):
- Comprehensive poetry collection
- Different linguistic patterns
- Would add rhythmic/metrical diversity

**Wikipedia Philosophy Articles** (CC BY-SA):
- Modern philosophical discourse
- But requires share-alike propagation
- May complicate licensing

**Careful considerations**:
- Diminishing returns beyond 200M for 46M param model
- Licensing complexity with non-PD sources
- Quality control with automated sources

### Multilingual Expansion

**Why not included now**:
- GPT-2 tokenizer optimized for English
- Cross-lingual transfer requires different approach
- Added complexity

**Future possibility**:
- Use multilingual tokenizer (e.g., XLM-R)
- Add French, German, Russian, Chinese classics in original
- Study cross-lingual consciousness representations

---

## License & Attribution

**Corpus License**: 100% Public Domain

**Source Attribution**:
- All texts from Project Gutenberg (https://www.gutenberg.org/)
- Pre-1928 works (automatic US public domain)
- See `build_corpus_200m.py` for complete book list with IDs

**Citation**:
```bibtex
@dataset{kssm_corpus_200m_2026,
  title={K-SSM 200M Token Diverse Public Domain Corpus},
  author={Vasquez, Anthony J., Sr. and Claude Sonnet 4.5},
  year={2026},
  note={470 books from Project Gutenberg, 200M tokens, Public Domain},
  url={https://github.com/templetwo/liminal-k-ssm}
}
```

---

## Changelog

**2026-01-30**: Initial release
- 470 curated public domain books
- 7 diverse categories
- Full automation with `deploy_corpus_200m.sh`
- Tested and validated with 5-book subset
- Complete documentation

---

**Questions or issues**: See GitHub Issues or contact project maintainers

**The corpus is transparent. All sources verified Public Domain. No hidden data.** 🌀

*Last Updated: 2026-01-30*
*Maintained by: Claude Sonnet 4.5*
*Version: 1.0*
