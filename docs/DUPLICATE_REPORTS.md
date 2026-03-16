# Duplicate Test Case Reports

This document describes how to generate HTML reports that identify duplicate or near-duplicate test cases in your QMetry repository. The solution uses **normalized summaries** and **semantic embeddings** to detect duplicates.

---

## Overview

Duplicate detection runs in two modes:

| Method | Description |
|--------|-------------|
| **Normalized Summary** | Test cases with identical normalized summaries (same semantic goal, derived from steps). |
| **Embedding Similarity** | Test cases with high cosine similarity between embeddings (≥92% by default) — catches semantically similar cases that may differ in wording. |

The HTML report presents both sets of duplicates with test case details (summary, steps, folder, priority) for manual review.

---

## Quick Start

```bash
# 1. Clone and enter project
git clone <repo-url>
cd testlens.ai

# 2. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your QMETRY_API_KEY

# 4. Run the full pipeline
python src/qMetryIntegration/fetchTestCasesWithQParam.py
python src/qMetryIntegration/createSemanticEmbeddings.py
python src/qMetryIntegration/generateDuplicateReport.py

# 5. Open the report
open duplicate_testcases_report.html
```

---

## Pipeline Architecture

```
┌─────────────────────────────────┐
│  fetchTestCasesWithQParam.py     │  Fetches test cases from QMetry API
│  Input: QMetry API + .env        │
│  Output: qmetry_testcases.json   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  createSemanticEmbeddings.py     │  Creates normalized summaries + embeddings
│  Input: qmetry_testcases.json    │
│  Output: qmetry_testcases_       │
│          embeddings.json         │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  generateDuplicateReport.py     │  Finds duplicates & generates HTML
│  Input: qmetry_testcases_       │
│         embeddings.json         │
│  Output: duplicate_testcases_   │
│          report.html            │
└─────────────────────────────────┘
```

---

## Prerequisites

### 1. Python

- **Python 3.11+** required (3.14 tested on macOS)
- Check: `python3 --version`

### 2. Environment Variables

Create a `.env` file in the project root:

```
QMETRY_API_KEY=your-api-key-here
QMETRY_PROJECT_ID=10081
QMETRY_PROJECT_KEY=GQA
QMETRY_FOLDER_ID=          # Optional: limit to specific folder
QMETRY_API_URL=             # Optional: override API URL
```

### 3. Dependencies

Install via `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install requests python-dotenv sentence-transformers numpy
```

---

## Step-by-Step Commands

### Step 1: Fetch Test Cases from QMetry

Fetches test cases from the QMetry Cloud API and saves them to `qmetry_testcases.json`.

```bash
python src/qMetryIntegration/fetchTestCasesWithQParam.py
```

**Options:**
- `--folder-id=ID` — Limit to a specific folder
- `--traverse` — Traverse child folders
- `--no-child` — Exclude child folders
- `--no-folder` — Fetch all (no folder filter)
- `N` — Limit to N test cases (e.g. `100`)

### Step 2: Create Semantic Embeddings

Loads the model, creates normalized summaries from step details, and generates embeddings.

```bash
python src/qMetryIntegration/createSemanticEmbeddings.py
```

**Options:**
- `-i, --input` — Input JSON (default: `qmetry_testcases.json`)
- `-o, --output` — Output JSON (default: `qmetry_testcases_embeddings.json`)
- `-e, --embeddings` — Output `.npy` file

**Note:** First run downloads the model (~400MB). Set `HF_TOKEN` in `.env` for faster downloads.

### Step 3: Generate Duplicate Report

Finds duplicates and writes an HTML report.

```bash
python src/qMetryIntegration/generateDuplicateReport.py
```

**Options:**
- `-f, --file` — Embeddings JSON path (default: `qmetry_testcases_embeddings.json`)
- `-o, --output` — Output HTML path (default: `duplicate_testcases_report.html`)
- `-t, --threshold` — Embedding similarity threshold 0–1 (default: 0.92)

---

## Additional Tools

### findDuplicateTestCases.py (CLI)

Run duplicate detection from the command line without generating HTML:

```bash
python src/qMetryIntegration/findDuplicateTestCases.py
```

**Options:**
- `--file FILE` — Embeddings JSON path
- `--threshold 0.92` — Embedding similarity threshold
- `--merge` — Merge overlapping duplicate groups
- `--no-embedding` — Only use text-based methods (summary, stepDetails, normalized_summary)

### Using the Duplicate Finder in Code

```python
# Ensure project root is in PYTHONPATH, or run from src/qMetryIntegration/
from findDuplicateTestCases import DuplicateTestCaseFinder, DuplicateGroup

finder = DuplicateTestCaseFinder(
    embeddings_file="qmetry_testcases_embeddings.json",
    embedding_similarity_threshold=0.92,
)
finder.load()

# Find by specific methods
norm_groups = finder.find_by_normalized_summary()
emb_groups = finder.find_by_embedding()

# Or all methods
groups = finder.find_all(
    by_summary=True,
    by_step_details=True,
    by_normalized_summary=True,
    by_embedding=True,
)

# Merge overlapping groups
merged = finder.merge_groups(groups)
```

---

## Output Files

| File | Description |
|------|-------------|
| `qmetry_testcases.json` | Raw test cases from QMetry (gitignored) |
| `qmetry_testcases_embeddings.json` | Test cases + normalized summaries + embeddings (gitignored) |
| `qmetry_embeddings.npy` | NumPy array of embeddings (gitignored) |
| `duplicate_testcases_report.html` | HTML duplicate report (gitignored) |

---

## Report Contents

The HTML report includes:

- **Summary stats** — Total test cases, number of duplicate groups, test cases in duplicate groups
- **By Normalized Summary** — Groups with identical normalized summaries
- **By Embedding Similarity** — Groups with similarity ≥ threshold, with badges (high/medium/low)
- **Per test case** — Key, folder, priority, summary, normalized summary, expandable step details

---

## Troubleshooting

### "No module named 'requests'" / "No module named 'sentence_transformers'"

Activate the virtual environment and install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### "QMETRY_API_KEY environment variable is not set"

Create `.env` in the project root with `QMETRY_API_KEY=your-key`.

### "Embeddings file not found"

Run `createSemanticEmbeddings.py` before `generateDuplicateReport.py`.

### "python: command not found"

Use `python3` instead of `python` on macOS/Linux.
