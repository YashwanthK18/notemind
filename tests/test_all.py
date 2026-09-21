"""
NoteMind - combined test suite.

Run with:
    pip install -r requirements-dev.txt
    pytest tests/test_all.py

Covers: chunker, FAISS vector store wrapper, metadata save/load,
document loading (including a regression test for the page_context
bug), and the retriever's figure-detection / technical-term-matching /
deduplication logic.

NOTE ON THE EMBEDDINGS STUB:
`app/embeddings.py` downloads and loads a real sentence-transformer
model at *import time*, and `app/retriever.py` imports from it. None of
the tests below need real embeddings, so a fake `app.embeddings` module
is installed into sys.modules at the top of this file -- before any
`app.*` import -- so tests run fast and offline, without downloading
any model.
"""

import sys
import types
import zipfile

import numpy as np
import pymupdf
import pytest


# =====================================================================
# Stub app.embeddings BEFORE importing anything from app.*
# =====================================================================

def _fake_create_query_embedding(query):
    rng = np.random.default_rng(abs(hash(query)) % (2**32))
    vector = rng.random(384).astype("float32")
    return vector.reshape(1, -1)


def _fake_create_embeddings(texts):
    rng = np.random.default_rng(0)
    return rng.random((len(texts), 384)).astype("float32")


if "app.embeddings" not in sys.modules:

    fake_embeddings_module = types.ModuleType("app.embeddings")
    fake_embeddings_module.MODEL_NAME = "fake-model-for-tests"
    fake_embeddings_module.create_query_embedding = _fake_create_query_embedding
    fake_embeddings_module.create_embeddings = _fake_create_embeddings

    sys.modules["app.embeddings"] = fake_embeddings_module


from app.chunker import chunk_text                       # noqa: E402
from app.loader import load_documents                    # noqa: E402
from app.retriever import Retriever                       # noqa: E402
from app.storage import load_metadata, save_metadata       # noqa: E402
from app.vector_store import VectorStore                   # noqa: E402


# =====================================================================
# CHUNKER
# =====================================================================

def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_returns_single_chunk():
    text = "This is a short paragraph with fewer words than chunk_size."
    chunks = chunk_text(text, chunk_size=250, overlap=40)

    assert len(chunks) == 1
    assert chunks[0] == " ".join(text.split())


def test_markdown_headings_split_into_separate_sections():
    text = (
        "# Introduction\n"
        "This is the intro paragraph.\n"
        "## Background\n"
        "This is the background paragraph."
    )

    chunks = chunk_text(text, chunk_size=250, overlap=40)

    assert len(chunks) == 2
    assert chunks[0].startswith("# Introduction")
    assert chunks[1].startswith("## Background")


def test_all_caps_heading_starts_a_new_section():
    text = (
        "DISTRIBUTED SYSTEMS\n"
        "A distributed system is a collection of independent computers.\n"
        "COMMUNICATION MODELS\n"
        "Message passing and shared memory are two common models."
    )

    chunks = chunk_text(text, chunk_size=250, overlap=40)

    assert len(chunks) == 2
    assert chunks[0].startswith("DISTRIBUTED SYSTEMS")
    assert chunks[1].startswith("COMMUNICATION MODELS")


def test_large_section_is_split_with_overlap():
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.strip()
        assert len(chunk.split()) <= 100

    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()

    overlap_words = first_chunk_words[-20:]
    assert second_chunk_words[: len(overlap_words)] == overlap_words


def test_no_infinite_loop_when_overlap_smaller_than_chunk_size():
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=10, overlap=5)

    assert len(chunks) > 0


# =====================================================================
# VECTOR STORE
# =====================================================================

def test_search_on_empty_store_returns_empty_list():
    store = VectorStore(dimension=8)
    query = np.random.rand(8).astype("float32")

    assert store.search(query, top_k=5) == []


def test_add_and_search_returns_expected_shape():
    store = VectorStore(dimension=4)

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype="float32",
    )

    store.add(embeddings)

    assert store.size() == 3
    assert store.dimension() == 4

    query = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    results = store.search(query, top_k=2)

    assert len(results) == 2
    for result in results:
        assert "index" in result
        assert "score" in result


def test_identical_vector_scores_close_to_one():
    # Proves the cosine-similarity fix (normalize_L2 on add + search)
    # actually works: a vector searched against itself should score ~1.0.
    store = VectorStore(dimension=4)

    vector = np.array([3.0, 4.0, 0.0, 0.0], dtype="float32")
    store.add(vector.copy())

    results = store.search(vector.copy(), top_k=1)

    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)


def test_dimension_mismatch_raises_value_error():
    store = VectorStore(dimension=4)
    store.add(np.random.rand(4).astype("float32"))

    bad_query = np.random.rand(6).astype("float32")

    with pytest.raises(ValueError):
        store.search(bad_query, top_k=1)


def test_top_k_larger_than_store_size_is_clamped():
    store = VectorStore(dimension=4)
    store.add(np.random.rand(2, 4).astype("float32"))

    results = store.search(np.random.rand(4).astype("float32"), top_k=50)

    assert len(results) <= store.size()


def test_vector_store_save_and_load_round_trip(tmp_path):
    store = VectorStore(dimension=4)
    store.add(np.random.rand(5, 4).astype("float32"))

    index_path = str(tmp_path / "test.index")
    store.save(index_path)

    loaded = VectorStore.load(index_path)

    assert loaded.size() == 5
    assert loaded.dimension() == 4


# =====================================================================
# METADATA STORAGE
# =====================================================================

def test_metadata_save_and_load_round_trip(tmp_path):
    chunks = [
        {"source": "notes.md", "type": ".md", "text": "hello world"},
        {"source": "a.pdf", "type": "image", "page": 3, "image_index": 0},
    ]

    path = str(tmp_path / "metadata.json")

    save_metadata(chunks, path)
    loaded = load_metadata(path)

    assert loaded == chunks


def test_metadata_unicode_content_is_preserved(tmp_path):
    chunks = [{"source": "notes.md", "text": "café résumé — π ≈ 3.14"}]

    path = str(tmp_path / "metadata.json")

    save_metadata(chunks, path)
    loaded = load_metadata(path)

    assert loaded[0]["text"] == "café résumé — π ≈ 3.14"


# =====================================================================
# LOADER
# =====================================================================

def _make_pdf_with_text_and_image(path):
    """Build a minimal real PDF (text + one embedded image)."""
    doc = pymupdf.open()
    page = doc.new_page()

    page_text = "FIGURE 1.1 A tiny test diagram."
    page.insert_text((72, 72), page_text)

    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00"
        b"\x02\x08\x02\x00\x00\x00\xfd\xd4\x9as\x00\x00\x00\x0cIDATx\x9cc"
        b"\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00"
        b"\x00IEND\xaeB`\x82"
    )
    page.insert_image(pymupdf.Rect(72, 100, 172, 200), stream=png_bytes)

    doc.save(str(path))
    doc.close()

    return page_text


def test_load_documents_reads_txt_and_md(tmp_path):
    (tmp_path / "notes.txt").write_text("hello from txt", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# heading\ncontent", encoding="utf-8")
    (tmp_path / "ignored.exe").write_bytes(b"\x00\x01")

    documents = load_documents(tmp_path)

    sources = {doc["source"] for doc in documents}
    assert "notes.txt" in sources
    assert "notes.md" in sources
    assert "ignored.exe" not in sources


def test_empty_text_file_is_skipped(tmp_path):
    (tmp_path / "empty.txt").write_text("   \n  ", encoding="utf-8")

    assert load_documents(tmp_path) == []


def test_pdf_page_text_is_loaded(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    page_text = _make_pdf_with_text_and_image(pdf_path)

    documents = load_documents(tmp_path)

    text_docs = [d for d in documents if d.get("type") == ".pdf"]
    assert len(text_docs) == 1
    assert page_text in text_docs[0]["text"]


def test_pdf_image_chunk_includes_page_context(tmp_path):
    """
    Regression test: app/loader.py used to build image documents
    without a "page_context" key, so ingest.py's
    document.get("page_context", "") always fell back to "" and image
    chunks only ever carried raw (often noisy) OCR text -- losing the
    much more reliable caption text sitting right there on the page.
    """
    pdf_path = tmp_path / "sample.pdf"
    page_text = _make_pdf_with_text_and_image(pdf_path)

    documents = load_documents(tmp_path)

    image_docs = [d for d in documents if d.get("type") == "image"]
    assert len(image_docs) == 1

    image_doc = image_docs[0]
    assert "page_context" in image_doc
    assert image_doc["page_context"].strip() != ""
    assert page_text in image_doc["page_context"]


def test_docx_file_is_loaded(tmp_path):
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>Hello from docx</w:t></w:r></w:p>
      </w:body>
    </w:document>"""

    target = tmp_path / "notes.docx"
    with zipfile.ZipFile(target, "w") as docx:
        docx.writestr("word/document.xml", document_xml)

    documents = load_documents(tmp_path)

    docx_docs = [d for d in documents if d.get("type") == ".docx"]
    assert len(docx_docs) == 1
    assert "Hello from docx" in docx_docs[0]["text"]


# =====================================================================
# RETRIEVER
# =====================================================================

def make_retriever(metadata=None):
    return Retriever(vector_store=None, metadata=metadata or [])


def test_detects_visual_questions():
    retriever = make_retriever()

    assert retriever._is_visual_query("What does Figure 3.13 show?")
    assert retriever._is_visual_query("Explain the diagram on page 5")
    assert retriever._is_visual_query("What is illustrated here?")


def test_does_not_flag_normal_questions_as_visual():
    retriever = make_retriever()

    assert not retriever._is_visual_query("What is MPI_Ssend?")
    assert not retriever._is_visual_query("What causes deadlock in a ring?")


def test_extracts_figure_number_various_formats():
    retriever = make_retriever()

    assert retriever._extract_figure_number("What does Figure 3.13 show?") == "3.13"
    assert retriever._extract_figure_number("explain fig. 2") == "2"
    assert retriever._extract_figure_number("Figure7 is interesting") == "7"


def test_returns_none_when_no_figure_number_present():
    retriever = make_retriever()

    assert retriever._extract_figure_number("What is MPI_Ssend?") is None


def test_detects_mpi_function_names():
    retriever = make_retriever()

    terms = retriever._technical_terms("What is MPI_Ssend used for?")

    assert "mpi_ssend" in terms


def test_technical_terms_includes_figure_reference():
    retriever = make_retriever()

    terms = retriever._technical_terms("What does Figure 3.13 show?")

    assert "figure 3.13" in terms


def test_no_technical_terms_for_generic_question():
    retriever = make_retriever()

    assert retriever._technical_terms("How does deadlock happen?") == []


def test_extract_keywords_strips_question_words():
    retriever = make_retriever()

    keywords = retriever._extract_keywords("What causes unsafe communication in a ring?")

    assert "causes" in keywords
    assert "unsafe" in keywords
    assert "communication" in keywords
    assert "ring" in keywords
    assert "what" not in keywords
    assert "the" not in keywords


def test_find_exact_figure_matches_only_image_chunks():
    metadata = [
        {
            "source": "a.pdf",
            "type": "image",
            "page": 39,
            "image_index": 2,
            "text": "FIGURE 3.13 Safe communication with five processes.",
        },
        {
            "source": "a.pdf",
            "type": ".pdf",
            "page": 39,
            "text": "Figure 3.13 shows a safe ordering that avoids deadlock.",
        },
        {
            "source": "a.pdf",
            "type": "image",
            "page": 10,
            "image_index": 0,
            "text": "FIGURE 3.4 An unsafe ring.",
        },
    ]

    retriever = make_retriever(metadata)

    matches = retriever._find_exact_figure("What does Figure 3.13 show?")

    assert len(matches) == 1
    assert matches[0]["page"] == 39
    assert matches[0]["type"] == "image"


def test_find_exact_figure_returns_empty_when_no_figure_number():
    retriever = make_retriever([{"type": "image", "text": "FIGURE 3.13 ..."}])

    assert retriever._find_exact_figure("What is MPI_Ssend?") == []


def test_remove_duplicates_collapses_identical_image_chunks():
    retriever = make_retriever()

    results = [
        {"source": "a.pdf", "page": 39, "image_index": 2, "type": "image", "text": "x"},
        {"source": "a.pdf", "page": 39, "image_index": 2, "type": "image", "text": "x"},
        {"source": "a.pdf", "page": 39, "image_index": 3, "type": "image", "text": "y"},
    ]

    unique = retriever._remove_duplicates(results)

    assert len(unique) == 2


def test_remove_duplicates_uses_text_prefix_for_text_chunks():
    retriever = make_retriever()

    results = [
        {"source": "a.pdf", "page": 1, "type": ".pdf", "text": "same beginning of text " * 5},
        {"source": "a.pdf", "page": 1, "type": ".pdf", "text": "same beginning of text " * 5},
        {"source": "a.pdf", "page": 2, "type": ".pdf", "text": "different text entirely"},
    ]

    unique = retriever._remove_duplicates(results)

    assert len(unique) == 2


def test_remove_duplicates_skips_falsy_results():
    retriever = make_retriever()

    # None and {} are both falsy, so both get skipped -- this just
    # guards against the method ever raising on falsy/empty input.
    assert retriever._remove_duplicates([None, {}, None]) == []
    assert retriever._remove_duplicates([None, {"source": "a"}]) == [{"source": "a"}]
