from app.loader import load_documents
from app.chunker import chunk_text
from app.embeddings import create_embeddings
from app.vector_store import VectorStore
from app.storage import save_metadata
from app.image_processor import process_image


INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/metadata.json"


def ingest():

    print("Reading documents...", flush=True)

    documents = load_documents("notes")

    print(
        f"Found {len(documents)} document sections.",
        flush=True
    )

    all_chunks = []

    # =========================================================
    # PROCESS DOCUMENTS
    # =========================================================

    for document_number, document in enumerate(
        documents,
        start=1
    ):

        print(
            f"\nProcessing document "
            f"{document_number}/{len(documents)}...",
            flush=True
        )

        # =====================================================
        # NORMAL TEXT
        # =====================================================

        if (
            "text" in document
            and document["text"]
            and document["text"].strip()
        ):

            print(
                "  Chunking normal text...",
                flush=True
            )

            chunks = chunk_text(
                document["text"]
            )

            print(
                f"  Created {len(chunks)} text chunks.",
                flush=True
            )

            for chunk_index, chunk in enumerate(chunks):

                metadata = {
                    "text": chunk,
                    "source": document["source"],
                    "type": document["type"],
                    "chunk_index": chunk_index
                }

                if "page" in document:
                    metadata["page"] = document["page"]

                all_chunks.append(metadata)

        # =====================================================
        # IMAGE
        # =====================================================

        if document.get("type") == "image":

            print(
                f"  Processing image "
                f"{document.get('image_index', '?')} "
                f"on page "
                f"{document.get('page', '?')}...",
                flush=True
            )

            print(
                "  Running OCR...",
                flush=True
            )

            image_result = process_image(
                image_data=document["image_data"],
                source=document["source"],
                page=document["page"],
                image_index=document["image_index"],
                extension=document["image_extension"]
            )

            print(
                "  OCR completed.",
                flush=True
            )

            image_text = image_result["text"]

            # =================================================
            # ADD PAGE CONTEXT
            # =================================================

            page_context = document.get(
                "page_context",
                ""
            )

            combined_text = ""

            if page_context.strip():

                combined_text += (
                    "PAGE CONTEXT:\n"
                    + page_context
                    + "\n\n"
                )

            if image_text.strip():

                combined_text += (
                    "IMAGE OCR:\n"
                    + image_text
                )

            image_text = combined_text

            # =================================================
            # CHUNK IMAGE TEXT
            # =================================================

            if image_text.strip():

                print(
                    "  Chunking OCR text...",
                    flush=True
                )

                chunks = chunk_text(
                    image_text
                )

                print(
                    f"  Created {len(chunks)} image chunks.",
                    flush=True
                )

                for chunk_index, chunk in enumerate(chunks):

                    all_chunks.append({
                        "text": chunk,
                        "source": document["source"],
                        "type": "image",
                        "page": document["page"],
                        "image_index": document["image_index"],
                        "image_path": image_result["image_path"],
                        "chunk_index": chunk_index
                    })

            else:

                print(
                    "  No OCR text found.",
                    flush=True
                )

    # =========================================================
    # CHUNKS
    # =========================================================

    print(
        f"\nCreated {len(all_chunks)} chunks.",
        flush=True
    )

    if not all_chunks:

        print(
            "No chunks were created.",
            flush=True
        )

        return

    # =========================================================
    # EMBEDDINGS
    # =========================================================

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    print(
        "\nGenerating embeddings...",
        flush=True
    )

    print(
        f"Total texts to embed: {len(texts)}",
        flush=True
    )

    embeddings = create_embeddings(
        texts
    )

    print(
        f"Embeddings generated: {embeddings.shape}",
        flush=True
    )

    # =========================================================
    # FAISS
    # =========================================================

    print(
        "\nCreating FAISS index...",
        flush=True
    )

    dimension = embeddings.shape[1]

    vector_store = VectorStore(
        dimension
    )

    print(
        "Adding embeddings to FAISS...",
        flush=True
    )

    vector_store.add(
        embeddings
    )

    print(
        "Saving FAISS index...",
        flush=True
    )

    vector_store.save(
        INDEX_PATH
    )

    print(
        f"FAISS index saved to {INDEX_PATH}",
        flush=True
    )

    # =========================================================
    # METADATA
    # =========================================================

    print(
        "Saving metadata...",
        flush=True
    )

    save_metadata(
        all_chunks,
        METADATA_PATH
    )

    print(
        f"Metadata saved to {METADATA_PATH}",
        flush=True
    )

    # =========================================================
    # COMPLETE
    # =========================================================

    print(
        "\nIngestion completed successfully.",
        flush=True
    )

    print(
        f"Documents: {len(documents)}",
        flush=True
    )

    print(
        f"Chunks: {len(all_chunks)}",
        flush=True
    )

    print(
        f"Index: {INDEX_PATH}",
        flush=True
    )

    print(
        f"Metadata: {METADATA_PATH}",
        flush=True
    )


if __name__ == "__main__":
    ingest()