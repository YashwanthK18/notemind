

Readme · MD
# NoteMind
 
A local, offline RAG (Retrieval-Augmented Generation) assistant for your study notes. Ask questions in plain English and get answers grounded in your own `.txt`, `.md`, `.pdf`, and `.docx` files — including diagrams and figures inside PDFs.
 
Everything runs locally. No external API calls, no data leaving your machine.
 
## Features
 
- Ingests `.txt`, `.md`, `.pdf`, and `.docx` files
- Extracts and OCRs images/diagrams embedded in PDFs (via Tesseract)
- Semantic search over your notes using sentence embeddings + FAISS
- Hybrid retrieval: semantic search for conceptual questions, exact-match lookup for explicit references like "Figure 3.13" or technical terms like `MPI_Ssend`
- Multimodal answers: for visual questions, sends the actual diagram to a vision model (LLaVA) alongside its OCR text, with automatic fallback to the text model if the vision model is unavailable
- OCR results are cached so re-ingesting unchanged images is fast
## Architecture
 
```
                    User Query
                        |
              Query Classification
             /                    \
    Normal Question          Visual Question
         |                          |
   Query Embedding          Detect Figure Number
         |                          |
     FAISS Search           Direct Metadata Search
         |                          |
  Similarity Ranking          Exact Figure Match
             \                    /
               Retrieved Context
                        |
               Answer Generation
              /                  \
        Text LLM                LLaVA
              \       fallback     /
               \___________________/
                        |
                     Answer
```
 
## Tech stack
 
- **Embeddings:** [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **Vector search:** [FAISS](https://github.com/facebookresearch/faiss) (cosine similarity via normalized inner product)
- **OCR:** [Tesseract](https://github.com/tesseract-ocr/tesseract) via `pytesseract`
- **PDF parsing:** [PyMuPDF](https://pymupdf.readthedocs.io/)
- **LLM / vision model:** [Ollama](https://ollama.com), running `llama3.2` and `llava` locally
## Setup
 
### 1. Clone and enter the project
 
```bash
git clone <your-repo-url>
cd notemind
```
 
### 2. Create and activate a virtual environment
 
**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```
 
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
 
### 3. Install Python dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Install Tesseract OCR (system tool, not a pip package)
 
- **Windows:** install from [UB-Mannheim's Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki) (default install path works out of the box)
- **Mac:** `brew install tesseract`
- **Linux:** `sudo apt install tesseract-ocr`
### 5. Install Ollama and pull the models
 
Install [Ollama](https://ollama.com), then:
 
```bash
ollama pull llama3.2
ollama pull llava
```
 
Make sure the Ollama service is running before using NoteMind — it's called for every answer.
 
## Usage
 
### 1. Add your notes
 
Create a `notes/` folder in the project root (it's gitignored, so it won't be committed) and drop in your `.txt`, `.md`, `.pdf`, and `.docx` files.
 
### 2. Build the index
 
```bash
python ingest.py
```
 
This reads everything in `notes/`, OCRs any images found in PDFs, generates embeddings, and writes `data/faiss.index` and `data/metadata.json`. This can take a few minutes for PDFs with many images.
 
Re-run this any time you add or change files in `notes/`.
 
### 3. Ask questions
 
```bash
python main.py
```
 
You'll get a `You:` prompt. Ask anything about your notes. Type `exit` or `quit`, or press `Ctrl+C`, to stop.
 
## Project structure
 
```
notemind/
├── app/
│   ├── loader.py              # Loads .txt/.md/.pdf/.docx, extracts PDF images
│   ├── docx_loader.py         # Parses .docx (heading styles → markdown)
│   ├── chunker.py             # Splits text into overlapping chunks
│   ├── embeddings.py          # Sentence-transformer embedding generation
│   ├── vector_store.py        # FAISS index wrapper (cosine similarity)
│   ├── retriever.py           # Hybrid semantic + exact-match retrieval
│   ├── image_processor.py     # Runs/caches OCR on extracted images
│   ├── image_extractor.py     # Saves extracted PDF images to disk
│   ├── image_understanding.py # LLaVA-based image description
│   ├── generator.py           # Builds prompts, calls Ollama, formats answers
│   ├── context_builder.py     # Formats retrieved chunks into LLM context
│   └── storage.py             # Metadata JSON read/write
├── notes/                     # Your source documents (gitignored)
├── data/                      # Generated index, metadata, images, OCR cache (gitignored)
├── ingest.py                  # Builds the FAISS index from notes/
├── main.py                    # Interactive Q&A loop
├── test_ocr.py                # Standalone OCR test script
└── requirements.txt
```
 
## Testing
 
Unit tests cover the parts of the pipeline that don't require Ollama or Tesseract to be running: the chunker, the FAISS vector store wrapper, metadata save/load, document loading (including a real, dynamically-built PDF with an embedded image), and the retriever's figure-detection, technical-term-matching, and deduplication logic.
 
```bash
pip install -r requirements-dev.txt
python -m pytest tests/test_all.py
```
 
`test_all.py` stubs out `app.embeddings` with fake, deterministic vectors so tests don't need to download the real sentence-transformer model or hit the network.
 
If plain `pytest` fails with `ModuleNotFoundError: No module named 'app'`, either run `python -m pytest` instead, or make sure `conftest.py` exists at the project root (it does in this repo) — its presence is what lets pytest find the `app/` package.
 
Verified passing on Windows (Python 3.14.3):
 
```
collected 32 items
 
tests\test_all.py ................................             [100%]
 
======================== 32 passed in 1.14s =========================
```
 
## Evaluation (quantitative retrieval metrics)
 
Unit tests check code correctness with fake data. `eval/run_eval.py` checks retrieval *quality* against your real, already-ingested notes — producing actual numbers (hit-rate, MRR, latency), not pass/fail.
 
Requires `python ingest.py` to have already been run (so `data/faiss.index` and `data/metadata.json` exist), and uses the real embedding model.
 
```bash
python eval/run_eval.py
python eval/run_eval.py --top-k 5 --save-results eval/results.json
```
 
Edit `eval/qa_testset.json` to add your own questions. Each entry needs a query and `expected_keywords` (all must appear, case-insensitively, in a retrieved chunk for it to count as a hit), plus optional `expected_source`/`expected_page` to require a specific document/page.
 
### Sample results
 
Run against the real, ingested `BCS702-module-3-pdf.pdf` notes (437 indexed vectors):
 
```
========================================================================
NoteMind Retrieval Evaluation  (top_k=5)
========================================================================
[HIT ] rank=1   latency= 185.9ms  figure_3_13: 'What does Figure 3.13 show?'
[HIT ] rank=1   latency=  40.4ms  mpi_ssend: 'What is MPI_Ssend?'
[HIT ] rank=1   latency=  33.4ms  unsafe_ring_communication: 'What causes unsafe communication in a ring?'
------------------------------------------------------------------------
Questions evaluated : 3
Hit@5              : 3/3  (100.0%)
MRR                 : 1.000
Avg latency         : 86.6 ms/query
========================================================================
```
 
Every question hit its correct chunk at rank 1 — for the exact-figure and technical-term queries, this reflects the hybrid retrieval design (`_find_exact_figure` / `_technical_terms` in `app/retriever.py`) rather than semantic search alone, which is exactly why those paths were added in the first place. Latency includes the embedding + FAISS search + retriever post-processing per query.
 
## Notes
 
- `data/` and `notes/` are gitignored since they're either generated or personal — anyone cloning the repo needs to add their own notes and run `ingest.py` before `main.py` will work.
- If the FAISS index and metadata ever get out of sync (e.g. `ingest.py` was interrupted), delete `data/faiss.index` and `data/metadata.json` and re-run `python ingest.py`.
 
