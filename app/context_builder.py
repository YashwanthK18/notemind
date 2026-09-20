def format_source(chunk):

    source = chunk["source"]

    if chunk.get("type") == "image":

        page = chunk.get(
            "page",
            "?"
        )

        image_index = chunk.get(
            "image_index",
            "?"
        )

        return (
            f"{source} | "
            f"Page {page} | "
            f"Image {image_index}"
        )


    if "page" in chunk:

        return (
            f"{source} | "
            f"Page {chunk['page']}"
        )


    return source


def build_context(chunks):

    context_parts = []


    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        source = format_source(
            chunk
        )

        context_parts.append(
            f"""
[Source {index}]

Location:
{source}

Content:
{chunk['text']}
"""
        )


    return "\n".join(
        context_parts
    )
