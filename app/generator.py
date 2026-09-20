import re
import ollama


# =========================================================
# MODELS
# =========================================================

TEXT_MODEL = "llama3.2"
VISION_MODEL = "llava"


# =========================================================
# IMAGE DETECTION
# =========================================================

def is_image_chunk(chunk):
    return (
        chunk.get("type") == "image"
        or bool(chunk.get("image_path"))
        or "image_index" in chunk
    )


# =========================================================
# SOURCE FORMATTING
# =========================================================

def format_source(chunk):

    source = chunk.get(
        "source",
        "Unknown source"
    )

    if is_image_chunk(chunk):

        return (
            f"{source} | "
            f"Page {chunk.get('page', '?')} | "
            f"Image {chunk.get('image_index', '?')}"
        )

    if "page" in chunk:

        return (
            f"{source} | "
            f"Page {chunk.get('page', '?')}"
        )

    return source


# =========================================================
# CLEAN USER QUERY
# =========================================================

def clean_query(query):

    if not query:
        return ""

    query = query.strip()

    # Prevent accidental:
    # "You: What does Figure 3.13 show?"
    query = re.sub(
        r"^\s*you\s*:\s*",
        "",
        query,
        flags=re.IGNORECASE
    )

    return query.strip()


# =========================================================
# VISUAL QUESTION DETECTION
# =========================================================

def is_visual_question(query):

    q = clean_query(query).lower()

    visual_terms = [

        "figure",
        "fig.",
        "diagram",
        "illustration",
        "illustrates",
        "shown in",
        "shows",
        "image",
        "picture",
        "photo",
        "chart",
        "graph",
        "table",
        "flowchart",

        "what does the figure",
        "what does figure",

        "what is shown",
        "what is illustrated",

        "what does the diagram",
        "what does the image",

        "explain the figure",
        "explain figure",

        "explain the diagram",
        "explain the image"
    ]

    return any(
        term in q
        for term in visual_terms
    )


# =========================================================
# FIGURE NUMBER EXTRACTION
# =========================================================

def extract_figure_number(query):

    query = clean_query(query)

    patterns = [

        r"\bfigure\s+(\d+(?:\.\d+)*)\b",

        r"\bfig\.?\s*(\d+(?:\.\d+)*)\b"

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE
        )

        if match:

            return match.group(1)

    return None


# =========================================================
# EXACT FIGURE QUESTION
# =========================================================

def is_figure_description_question(query):

    q = clean_query(query).lower()

    patterns = [

        r"what\s+does\s+(the\s+)?figure",
        r"what\s+does\s+(the\s+)?fig",
        r"what\s+is\s+shown\s+in\s+(the\s+)?figure",
        r"what\s+is\s+shown\s+in\s+(the\s+)?fig",
        r"what\s+is\s+the\s+figure",
        r"explain\s+(the\s+)?figure"

    ]

    return any(
        re.search(
            pattern,
            q
        )
        for pattern in patterns
    )


# =========================================================
# FIND EXACT FIGURE CAPTION
# =========================================================

def find_exact_figure_caption(
    query,
    retrieved_chunks
):

    figure_number = extract_figure_number(
        query
    )

    if not figure_number:
        return None

    figure_pattern = re.compile(
        rf"\b(?:figure|fig\.?)\s*"
        rf"{re.escape(figure_number)}\b",
        flags=re.IGNORECASE
    )

    for chunk in retrieved_chunks:

        text = chunk.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        # -------------------------------------------------
        # Look for an actual caption.
        # -------------------------------------------------

        caption_pattern = re.compile(
            rf"FIGURE\s+{re.escape(figure_number)}"
            rf"\s+(.+?)(?:\n|$)",
            flags=re.IGNORECASE
        )

        match = caption_pattern.search(
            text
        )

        if match:

            caption = match.group(
                0
            ).strip()

            # Remove trailing whitespace.
            caption = re.sub(
                r"\s+",
                " ",
                caption
            )

            return caption

        # -------------------------------------------------
        # Also support "Figure 3.13 ..."
        # -------------------------------------------------

        if figure_pattern.search(text):

            lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip()
            ]

            for line in lines:

                if figure_pattern.search(
                    line
                ):

                    # A short line containing the
                    # figure reference is likely
                    # the caption.
                    if len(line) <= 200:

                        return line

    return None


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(retrieved_chunks):

    parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        image = is_image_chunk(
            chunk
        )

        chunk_type = (
            "image"
            if image
            else "text"
        )

        text = chunk.get(
            "text",
            ""
        ).strip()

        if not text:

            text = (
                "(No OCR/text was extracted "
                "from this image.)"
            )

        parts.append(

            f"[Source {index}]\n"
            f"Location: {format_source(chunk)}\n"
            f"Type: {chunk_type}\n"
            f"Content:\n{text}"

        )

    return "\n\n".join(
        parts
    )


# =========================================================
# TEXT MODEL
# =========================================================

def generate_text_answer(
    query,
    context
):

    query = clean_query(
        query
    )

    prompt = f"""
You are NoteMind, a local notes assistant.

Answer the user's question using ONLY the retrieved notes.

STRICT RULES:

1. Do not use outside knowledge.
2. Read all retrieved information before answering.
3. Give a natural, concise answer.
4. Do not copy large passages from the notes.
5. Do not dump raw PDF text.
6. Do not dump source metadata.
7. Do not mention retrieval, embeddings,
   vectors, chunks, prompts, models,
   FAISS, or internal processing.
8. Do not repeat the question.
9. If the notes directly contain the answer,
   answer it directly.
10. If the notes contain code, use it only
    when necessary to explain the answer.
11. For a cause question, state the cause
    followed by its consequence.
12. If the notes do not contain enough information,
    say that the information is not available
    in the notes.

IMPORTANT:

The retrieved notes may contain several unrelated
pieces of text. Do NOT combine unrelated pieces
just because they were retrieved.

RETRIEVED NOTES:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    try:

        response = ollama.chat(

            model=TEXT_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response[
            "message"
        ][
            "content"
        ].strip()

        return answer

    except Exception as error:

        print(
            f"Text model error: {error}",
            flush=True
        )

        return (
            "I couldn't generate an answer "
            "from the retrieved notes."
        )


# =========================================================
# IMAGE SELECTION
# =========================================================

def select_best_image(
    image_chunks
):

    if not image_chunks:
        return None

    chunks = sorted(

        image_chunks,

        key=lambda x: float(
            x.get(
                "score",
                0.0
            )
        ),

        reverse=True
    )

    for chunk in chunks:

        if chunk.get(
            "image_path"
        ):

            return chunk

    return None


# =========================================================
# VISION MODEL
# =========================================================

def generate_vision_answer(
    query,
    context,
    image_chunk
):

    image_path = image_chunk.get(
        "image_path"
    )

    if not image_path:

        return generate_text_answer(
            query,
            context
        )

    query = clean_query(
        query
    )

    prompt = f"""
You are NoteMind, a local notes assistant.

The user is asking about a visual element
from their notes.

Use BOTH:

1. The supplied image.
2. The retrieved notes.

STRICT RULES:

1. Use only information supported by
   the image and retrieved notes.
2. Do not use outside knowledge.
3. Do not invent labels, arrows,
   process numbers, timings,
   relationships, or sequences.
4. Do not guess unreadable text.
5. If the notes provide an explicit caption,
   use the caption.
6. If the notes provide an explanation
   of the figure, use that explanation.
7. Keep the answer concise.
8. Do not dump OCR text.
9. Do not reproduce code unless specifically
   necessary.
10. Do not mention models, retrieval,
    embeddings, chunks, prompts,
    FAISS, or internal processing.
11. Do not repeat the question.
12. If the exact visual details cannot
    be determined, say so.

For a question such as:

"What does Figure 3.13 show?"

First state the documented caption/purpose.

Then, only if clearly supported by the notes,
briefly explain what the figure represents.

RETRIEVED NOTES:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    try:

        print(
            f"  Using vision model: {VISION_MODEL}",
            flush=True
        )

        print(
            "  Sending 1 image to vision model...",
            flush=True
        )

        response = ollama.chat(

            model=VISION_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [
                        image_path
                    ]
                }
            ]
        )

        print(
            "  Vision analysis completed.",
            flush=True
        )

        return response[
            "message"
        ][
            "content"
        ].strip()

    except Exception as error:

        print(
            f"  Vision analysis unavailable: {error}",
            flush=True
        )

        print(
            "  Using text model fallback.",
            flush=True
        )

        return generate_text_answer(
            query,
            context
        )


# =========================================================
# MAIN ANSWER GENERATOR
# =========================================================

def generate_answer(
    query,
    retrieved_chunks
):

    query = clean_query(
        query
    )

    if not retrieved_chunks:

        return (
            "I couldn't find this information "
            "in your notes."
        )

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = build_context(
        retrieved_chunks
    )

    # -----------------------------------------------------
    # Detect visual question
    # -----------------------------------------------------

    visual_question = is_visual_question(
        query
    )

    # -----------------------------------------------------
    # Exact figure handling
    # -----------------------------------------------------

    if visual_question:

        figure_number = extract_figure_number(
            query
        )

        # -------------------------------------------------
        # If user explicitly asks about a figure,
        # prioritize the exact caption.
        # -------------------------------------------------

        if (
            figure_number
            and is_figure_description_question(
                query
            )
        ):

            caption = find_exact_figure_caption(
                query,
                retrieved_chunks
            )

            if caption:

                print(
                    "Direct figure information found.",
                    flush=True
                )

                print(
                    "Skipping vision model.",
                    flush=True
                )

                # -----------------------------------------
                # Extract only the caption.
                # Never append the entire PDF chunk.
                # -----------------------------------------

                caption = re.sub(
                    r"^FIGURE\s+",
                    "Figure ",
                    caption,
                    flags=re.IGNORECASE
                )

                return caption.strip()

    # -----------------------------------------------------
    # Normal text question
    # -----------------------------------------------------

    if not visual_question:

        print(
            "Text question detected. "
            "Using text model.",
            flush=True
        )

        return generate_text_answer(
            query,
            context
        )

    # -----------------------------------------------------
    # Visual question without exact caption
    # -----------------------------------------------------

    print(
        "Visual question detected.",
        flush=True
    )

    image_chunks = [

        chunk

        for chunk in retrieved_chunks

        if (
            is_image_chunk(chunk)
            and chunk.get("image_path")
        )

    ]

    # -----------------------------------------------------
    # No image available
    # -----------------------------------------------------

    if not image_chunks:

        print(
            "No actual image available. "
            "Using retrieved notes.",
            flush=True
        )

        return generate_text_answer(
            query,
            context
        )

    # -----------------------------------------------------
    # Select best image
    # -----------------------------------------------------

    selected_image = select_best_image(
        image_chunks
    )

    if not selected_image:

        print(
            "No usable image path found. "
            "Using text model.",
            flush=True
        )

        return generate_text_answer(
            query,
            context
        )

    # -----------------------------------------------------
    # Vision
    # -----------------------------------------------------

    print(
        "Selected 1 image for visual analysis.",
        flush=True
    )

    return generate_vision_answer(
        query,
        context,
        selected_image
    )
