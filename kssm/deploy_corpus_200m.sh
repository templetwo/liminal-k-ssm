#!/bin/bash

# K-SSM 200M Corpus Deployment Script
# Safely deploys corpus expansion to Mac Studio

set -e  # Exit on error

# Configuration
REMOTE_USER="tony_studio"
REMOTE_HOST="192.168.1.195"
REMOTE_DIR="~/liminal-k-ssm"
REMOTE_ADDR="${REMOTE_USER}@${REMOTE_HOST}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "K-SSM 200M CORPUS DEPLOYMENT"
echo "======================================================================"
echo ""
echo "Target: ${REMOTE_ADDR}:${REMOTE_DIR}"
echo ""

# Step 1: Backup existing corpus
echo "${YELLOW}[1/6] BACKING UP EXISTING CORPUS${NC}"
echo "----------------------------------------------------------------------"

ssh ${REMOTE_ADDR} "
    cd ${REMOTE_DIR}
    echo '  Checking for existing corpus...'

    # Backup current 22M corpus
    if [ -f kssm/data/processed/kssm_corpus.jsonl ]; then
        echo '  ✓ Found kssm_corpus.jsonl - creating backup'
        cp kssm/data/processed/kssm_corpus.jsonl kssm/data/processed/kssm_corpus_22M_backup.jsonl
        echo '    Saved as: kssm_corpus_22M_backup.jsonl'
    fi

    # Backup current tokens
    if [ -d data/cache_v3 ]; then
        echo '  ✓ Found cache_v3 - creating backup'
        cp -r data/cache_v3 data/cache_v3_22M_backup
        echo '    Saved as: cache_v3_22M_backup/'
    fi

    echo '  ✓ Backup complete'
"

echo ""

# Step 2: Transfer scripts
echo "${YELLOW}[2/6] TRANSFERRING SCRIPTS${NC}"
echo "----------------------------------------------------------------------"

echo "  Uploading build_corpus_200m.py..."
scp kssm/build_corpus_200m.py ${REMOTE_ADDR}:${REMOTE_DIR}/kssm/

echo "  Uploading process_corpus_200m.py..."
scp kssm/process_corpus_200m.py ${REMOTE_ADDR}:${REMOTE_DIR}/kssm/

echo "  ✓ Scripts transferred"
echo ""

# Step 3: Check dependencies
echo "${YELLOW}[3/6] CHECKING DEPENDENCIES${NC}"
echo "----------------------------------------------------------------------"

ssh ${REMOTE_ADDR} "
    cd ${REMOTE_DIR}

    echo '  Checking Python...'
    python3 --version

    echo '  Checking tiktoken...'
    if python3 -c 'import tiktoken' 2>/dev/null; then
        echo '    ✓ tiktoken installed'
    else
        echo '    ✗ tiktoken not found - installing...'
        pip3 install tiktoken
    fi

    echo '  Checking tqdm...'
    if python3 -c 'import tqdm' 2>/dev/null; then
        echo '    ✓ tqdm installed'
    else
        echo '    ✗ tqdm not found - installing...'
        pip3 install tqdm
    fi

    echo '  ✓ Dependencies ready'
"

echo ""

# Step 4: Test download
echo "${YELLOW}[4/6] TESTING DOWNLOAD (1 book)${NC}"
echo "----------------------------------------------------------------------"

ssh ${REMOTE_ADDR} "
    cd ${REMOTE_DIR}

    echo '  Testing download with Alice in Wonderland...'
    python3 -c \"
from kssm.build_corpus_200m import download_gutenberg_book
success = download_gutenberg_book(11, 'Alice in Wonderland', 'Lewis Carroll')
print('  ✓ Test download successful' if success else '  ✗ Test download failed')
\"
"

echo ""

# Step 5: Choose deployment scale
echo "${YELLOW}[5/6] DEPLOYMENT SCALE${NC}"
echo "----------------------------------------------------------------------"
echo ""
echo "  Incremental deployment options:"
echo ""
echo "    1) Test (10 books, ~5 min)   - Verify deployment works"
echo "    2) Small (50 books, ~10 min)  - Test processing pipeline"
echo "    3) Medium (150 books, ~25 min) - Validate at scale"
echo "    4) Full (470 books, ~45 min)   - Production corpus (200M tokens)"
echo ""
echo "  Recommended: Start with option 1, then 2, then 4"
echo ""
read -p "  Select option (1-4): " scale

case $scale in
    1)
        max_books=10
        est_time="5 minutes"
        est_tokens="2M"
        ;;
    2)
        max_books=50
        est_time="10 minutes"
        est_tokens="10M"
        ;;
    3)
        max_books=150
        est_time="25 minutes"
        est_tokens="60M"
        ;;
    4)
        max_books=470
        est_time="45 minutes"
        est_tokens="200M"
        ;;
    *)
        echo ""
        echo "${RED}Invalid option. Deployment cancelled.${NC}"
        exit 1
        ;;
esac

echo ""
echo "  Selected: $max_books books (~$est_tokens tokens, ~$est_time)"
echo ""
read -p "  Proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo ""
    echo "${RED}Deployment cancelled by user${NC}"
    exit 0
fi

echo ""

# Step 6: Full download
echo "${YELLOW}[6/6] DOWNLOADING FULL CORPUS${NC}"
echo "----------------------------------------------------------------------"

ssh ${REMOTE_ADDR} "
    cd ${REMOTE_DIR}

    echo '  Starting download ($max_books books)...'
    echo '  Estimated time: $est_time'
    echo '  Press Ctrl+C to abort'
    echo ''

    # Run download with max_books limit
    python3 -c \"
import sys
sys.path.insert(0, 'kssm')
from build_corpus_200m import download_all_sources, GUTENBERG_LITERATURE, PHILOSOPHY_BOOKS, RELIGIOUS_BOOKS, SCIENCE_BOOKS, POLITICAL_BOOKS, ESSAY_BOOKS, ANCIENT_BOOKS

# Limit total books
total_limit = $max_books
print(f'Downloading up to {total_limit} books...')

# Collect and limit
all_books = (
    GUTENBERG_LITERATURE +
    PHILOSOPHY_BOOKS +
    RELIGIOUS_BOOKS +
    SCIENCE_BOOKS +
    POLITICAL_BOOKS +
    ESSAY_BOOKS +
    ANCIENT_BOOKS
)[:total_limit]

print(f'Actual books to download: {len(all_books)}')

# Download
from build_corpus_200m import download_gutenberg_book, RAW_DIR
from tqdm import tqdm
import time

success = 0
for book_id, title, author in tqdm(all_books):
    if download_gutenberg_book(book_id, title, author):
        success += 1
    time.sleep(2)  # Rate limiting

print(f'Downloaded {success}/{len(all_books)} books')
\"

    echo ''
    echo '  ✓ Download complete'
    echo ''

    # Show stats
    python3 kssm/build_corpus_200m.py --stats
"

echo ""
echo "${GREEN}======================================================================"
echo "DOWNLOAD COMPLETE"
echo "======================================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Process corpus:  ssh ${REMOTE_ADDR} 'cd ${REMOTE_DIR} && python3 kssm/process_corpus_200m.py --build'"
echo "  2. Tokenize:        ssh ${REMOTE_ADDR} 'cd ${REMOTE_DIR} && python3 kssm/process_corpus_200m.py --tokenize'"
echo "  3. Or do both:      ssh ${REMOTE_ADDR} 'cd ${REMOTE_DIR} && python3 kssm/process_corpus_200m.py --all'"
echo ""
echo "Backups saved:"
echo "  - kssm/data/processed/kssm_corpus_22M_backup.jsonl"
echo "  - data/cache_v3_22M_backup/"
echo ""
