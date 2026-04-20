# TestLens.ai

TestLens is an AI-powered test intelligence platform that uses NLP embeddings and vector search to enable **semantic search**, **similarity detection**, and **duplicate discovery** across large test repositories, helping QA teams find relevant tests faster, reduce redundancy, and improve regression efficiency.

---

## Features

- **Fetch test cases** from QMetry Cloud API
- **Semantic embeddings** — normalized summaries and vector representations for similarity
- **Duplicate detection** — find duplicates by summary, steps, normalized summary, and embedding similarity
- **HTML reports** — visual duplicate reports for review
- **Qdrant integration** — upload embeddings for semantic search (optional)

---

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd testlens.ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure (copy and edit .env)
cp .env.example .env

# Run full pipeline (fetch -> embeddings -> report)
python index.py

# Open report
open reports/duplicate_testcases_report.html
```

---

## Output Locations

- `reports/duplicate_testcases_report.html` — generated duplicate report
- `embeddings_output/qmetry_testcases.json` — fetched raw QMetry test cases
- `embeddings_output/qmetry_testcases_embeddings.json` — test cases with normalized summaries + embeddings
- `embeddings_output/qmetry_embeddings.npy` — NumPy embedding matrix

If your pipeline generated outputs in the project root, move them with:

```bash
mkdir -p embeddings_output
mv qmetry_testcases.json qmetry_testcases_embeddings.json qmetry_embeddings.npy embeddings_output/
```

---

## Project Structure

```
testlens.ai/
├── index.py                           # End-to-end pipeline runner
├── src/qMetryIntegration/
│   ├── fetchTestCasesWithQParam.py   # Fetch from QMetry API
│   ├── createSemanticEmbeddings.py  # Create embeddings
│   ├── findDuplicateTestCases.py     # Duplicate detection (CLI + class)
│   ├── generateDuplicateReport.py    # HTML report
│   ├── uploadToQdrant.py             # Upload to Qdrant (optional)
│   └── searchQdrant.py               # Search Qdrant (optional)
├── docs/
│   └── DUPLICATE_REPORTS.md         # Full documentation
├── embeddings_output/                # JSON + NPY embedding artifacts
├── reports/
│   └── duplicate_testcases_report.html
├── requirements.txt
├── .env.example
└── README.md
```

---

## Documentation

- **[Duplicate Reports](docs/DUPLICATE_REPORTS.md)** — Full guide: setup, pipeline, CLI options, troubleshooting

---

## Requirements

- Python 3.11+
- QMetry Cloud API key
- ~2GB disk for sentence-transformers model (first run)

---

## License

See repository for license information.
