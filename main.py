from pathlib import Path

from app.vector_store import VectorStore
from app.storage import load_metadata
from app.retriever import Retriever
from app.generator import generate_answer


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/metadata.json"

TOP_K = 5

# IMPORTANT:
# Keep this low while debugging retrieval.
# If Retriever returns nothing because of the threshold,
# lowering this lets us verify that FAISS itself is working.
SIMILARITY_THRESHOLD = 0.20


# ============================================================
# CHECK FILES
# ============================================================

if not Path(INDEX_PATH).exists():
    print("\nERROR: FAISS index not found.")
    print(f"Expected: {INDEX_PATH}")
    print("\nRun:")
    print("python ingest.py")
    raise SystemExit(1)


if not Path(METADATA_PATH).exists():
    print("\nERROR: Metadata file not found.")
    print(f"Expected: {METADATA_PATH}")
    print("\nRun:")
    print("python ingest.py")
    raise SystemExit(1)


# ============================================================
# LOAD VECTOR STORE
# ============================================================

print("Loading NoteMind...")

try:
    vector_store = VectorStore.load(INDEX_PATH)
except Exception as e:
    print("\nERROR loading FAISS index:")
    print(e)
    raise SystemExit(1)


# ============================================================
# LOAD METADATA
# ============================================================

try:
    all_chunks = load_metadata(METADATA_PATH)
except Exception as e:
    print("\nERROR loading metadata:")
    print(e)
    raise SystemExit(1)


print(f"Loaded metadata chunks: {len(all_chunks)}")


# ============================================================
# CHECK INDEX / METADATA SIZE
# ============================================================

try:
    index_size = vector_store.index.ntotal
    print(f"FAISS vectors: {index_size}")
except Exception:
    index_size = None
    print("Could not determine FAISS vector count.")


if index_size is not None:

    if index_size == 0:
        print("\nERROR: FAISS index contains 0 vectors.")
        print("Run:")
        print("python ingest.py")
        raise SystemExit(1)

    if index_size != len(all_chunks):

        print("\nWARNING:")
        print(
            f"FAISS vectors ({index_size}) != "
            f"metadata chunks ({len(all_chunks)})"
        )

        print(
            "\nThis usually means the FAISS index and "
            "metadata.json were generated from different data."
        )

        print(
            "Recommended fix:"
        )

        print("1. Delete data/faiss.index")
        print("2. Delete data/metadata.json")
        print("3. Run: python ingest.py")


# ============================================================
# CREATE RETRIEVER
# ============================================================

try:

    retriever = Retriever(
        vector_store,
        all_chunks,
        similarity_threshold=SIMILARITY_THRESHOLD
    )

except Exception as e:

    print("\nERROR creating Retriever:")
    print(e)

    raise SystemExit(1)


# ============================================================
# START APPLICATION
# ============================================================

print("\n================================")
print("          NoteMind")
print("================================")
print("Your local notes assistant")
print("Type 'exit' to stop.")
print("================================")

print(
    f"\nRetrieval threshold: "
    f"{SIMILARITY_THRESHOLD}"
)

print(
    f"Top-K results: "
    f"{TOP_K}"
)

print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    try:

        query = input("You: ").strip()

    except KeyboardInterrupt:

        print("\n\nGoodbye!")
        break

    except EOFError:

        print("\n\nGoodbye!")
        break


    # --------------------------------------------------------
    # EMPTY QUERY
    # --------------------------------------------------------

    if not query:

        print(
            "\nPlease enter a question.\n"
        )

        continue


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if query.lower() in {
        "exit",
        "quit"
    }:

        print("\nGoodbye!")
        break


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    print(
        "\nSearching your notes..."
    )

    try:

        results = retriever.retrieve(
            query,
            top_k=TOP_K
        )

    except Exception as e:

        print("\nERROR during retrieval:")
        print(e)

        continue


    # --------------------------------------------------------
    # DEBUG RETRIEVAL
    # --------------------------------------------------------

    print(
        f"\nRetriever returned "
        f"{len(results)} result(s)."
    )


    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not results:

        print("\nAI:")

        print(
            "I couldn't find relevant information "
            "in your notes."
        )

        print(
            "\nDEBUG:"
        )

        print(
            f"Similarity threshold = "
            f"{SIMILARITY_THRESHOLD}"
        )

        print(
            "If you know the answer exists in the PDF, "
            "the problem is most likely inside "
            "app/retriever.py or the FAISS index."
        )

        print()

        continue


    # ========================================================
    # DISPLAY RETRIEVED CHUNKS
    # ========================================================

    print(
        "\nRetrieved chunks:"
    )


    for number, result in enumerate(
        results,
        start=1
    ):

        print(
            "\n--------------------------------"
        )

        print(
            f"Chunk {number}"
        )

        print(
            "--------------------------------"
        )


        # SOURCE

        print(
            f"Source: "
            f"{result.get('source', 'Unknown')}"
        )


        # SCORE

        score = result.get(
            "score",
            0
        )

        try:

            print(
                f"Score: "
                f"{float(score):.3f}"
            )

        except (TypeError, ValueError):

            print(
                f"Score: {score}"
            )


        # PAGE

        if "page" in result:

            print(
                f"Page: "
                f"{result['page']}"
            )


        # IMAGE

        if "image_index" in result:

            print(
                f"Image: "
                f"{result['image_index']}"
            )


        # TYPE

        if "type" in result:

            print(
                f"Type: "
                f"{result['type']}"
            )


        # TEXT

        print(
            "\nText:"
        )

        print(
            result.get(
                "text",
                ""
            )
        )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    print(
        "\nGenerating answer..."
    )


    try:

        answer = generate_answer(
            query,
            results
        )

    except Exception as e:

        print(
            "\nERROR generating answer:"
        )

        print(e)

        continue


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    print(
        "\nAI:"
    )

    print(
        answer
    )


    # ========================================================
    # SOURCES
    # ========================================================

    print(
        "\nSources:"
    )


    for number, result in enumerate(
        results,
        start=1
    ):

        source = result.get(
            "source",
            "Unknown"
        )

        score = result.get(
            "score",
            0
        )

        result_type = result.get(
            "type",
            "text"
        )


        # ----------------------------------------------------
        # IMAGE SOURCE
        # ----------------------------------------------------

        if result_type == "image":

            page = result.get(
                "page",
                "?"
            )

            image_index = result.get(
                "image_index",
                "?"
            )

            location = (
                f"{source} | "
                f"Page {page} | "
                f"Image {image_index}"
            )


        # ----------------------------------------------------
        # PAGE SOURCE
        # ----------------------------------------------------

        elif "page" in result:

            location = (
                f"{source} | "
                f"Page {result['page']}"
            )


        # ----------------------------------------------------
        # NORMAL SOURCE
        # ----------------------------------------------------

        else:

            location = source


        print(
            f"{number}. {location}"
        )

        try:

            print(
                f"   Relevance: "
                f"{float(score):.3f}"
            )

        except (TypeError, ValueError):

            print(
                f"   Relevance: "
                f"{score}"
            )


    print()