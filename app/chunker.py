import re


def chunk_text(
    text,
    chunk_size=250,
    overlap=40
):

    # ==========================================
    # Clean text
    # ==========================================

    text = text.strip()

    if not text:

        return []


    # ==========================================
    # Detect headings
    #
    # Supports:
    # # Heading
    # ## Heading
    # ### Heading
    #
    # Also supports simple ALL CAPS headings.
    # ==========================================

    lines = text.splitlines()

    sections = []

    current_section = []


    for line in lines:

        stripped = line.strip()


        is_markdown_heading = bool(
            re.match(
                r"^#{1,6}\s+.+",
                stripped
            )
        )


        is_caps_heading = (
            stripped
            and len(stripped.split()) <= 10
            and stripped.upper() == stripped
            and any(
                char.isalpha()
                for char in stripped
            )
        )


        is_heading = (
            is_markdown_heading
            or is_caps_heading
        )


        # --------------------------------------
        # New section
        # --------------------------------------

        if is_heading:

            if current_section:

                sections.append(
                    "\n".join(
                        current_section
                    ).strip()
                )

            current_section = [
                stripped
            ]


        else:

            if stripped:

                current_section.append(
                    stripped
                )


    # ==========================================
    # Add final section
    # ==========================================

    if current_section:

        sections.append(
            "\n".join(
                current_section
            ).strip()
        )


    # ==========================================
    # If no headings were detected
    # ==========================================

    if not sections:

        sections = [text]


    # ==========================================
    # Split sections into smaller chunks
    # ==========================================

    final_chunks = []


    for section in sections:

        words = section.split()


        # --------------------------------------
        # Small section
        # --------------------------------------

        if len(words) <= chunk_size:

            final_chunks.append(
                " ".join(words)
            )

            continue


        # --------------------------------------
        # Large section
        # --------------------------------------

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk = " ".join(
                words[start:end]
            )

            if chunk.strip():

                final_chunks.append(
                    chunk.strip()
                )


            # ----------------------------------
            # Move forward with overlap
            # ----------------------------------

            start += (
                chunk_size - overlap
            )


    return final_chunks
