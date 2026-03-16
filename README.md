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

# Pipeline: fetch → embeddings → duplicate report
python src/qMetryIntegration/fetchTestCasesWithQParam.py
python src/qMetryIntegration/createSemanticEmbeddings.py
python src/qMetryIntegration/generateDuplicateReport.py

# Open report
open duplicate_testcases_report.html
```

---

## Project Structure

```
testlens.ai/
├── src/qMetryIntegration/
│   ├── fetchTestCasesWithQParam.py   # Fetch from QMetry API
│   ├── createSemanticEmbeddings.py  # Create embeddings
│   ├── findDuplicateTestCases.py     # Duplicate detection (CLI + class)
│   ├── generateDuplicateReport.py    # HTML report
│   ├── uploadToQdrant.py             # Upload to Qdrant (optional)
│   └── searchQdrant.py               # Search Qdrant (optional)
├── docs/
│   └── DUPLICATE_REPORTS.md         # Full documentation
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
