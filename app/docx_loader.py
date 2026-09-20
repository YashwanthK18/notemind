from zipfile import ZipFile
from xml.etree import ElementTree


WORD_NAMESPACE = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
}


def get_paragraph_style(paragraph):

    style_node = paragraph.find(
        "./w:pPr/w:pStyle",
        WORD_NAMESPACE
    )

    if style_node is None:

        return None

    return style_node.get(
        f"{{{WORD_NAMESPACE['w']}}}val"
    )


def extract_paragraph_text(paragraph):

    texts = []

    for text_node in paragraph.findall(
        ".//w:t",
        WORD_NAMESPACE
    ):

        if text_node.text:

            texts.append(
                text_node.text
            )

    return "".join(texts).strip()


def convert_heading(
    text,
    style
):

    if not style:

        return text


    style_lower = style.lower()


    # ==========================================
    # Word Heading 1
    # ==========================================

    if (
        style_lower == "heading1"
        or "heading1" in style_lower
    ):

        return f"# {text}"


    # ==========================================
    # Word Heading 2
    # ==========================================

    if (
        style_lower == "heading2"
        or "heading2" in style_lower
    ):

        return f"## {text}"


    # ==========================================
    # Word Heading 3
    # ==========================================

    if (
        style_lower == "heading3"
        or "heading3" in style_lower
    ):

        return f"### {text}"


    # ==========================================
    # Word Heading 4
    # ==========================================

    if (
        style_lower == "heading4"
        or "heading4" in style_lower
    ):

        return f"#### {text}"


    return text


def load_docx_file(file_path):

    paragraphs = []


    # ==========================================
    # Open DOCX
    # ==========================================

    with ZipFile(
        file_path,
        "r"
    ) as docx:

        xml_data = docx.read(
            "word/document.xml"
        )


    # ==========================================
    # Parse XML
    # ==========================================

    root = ElementTree.fromstring(
        xml_data
    )


    # ==========================================
    # Extract paragraphs
    # ==========================================

    for paragraph in root.findall(
        ".//w:p",
        WORD_NAMESPACE
    ):

        paragraph_text = extract_paragraph_text(
            paragraph
        )


        if not paragraph_text:

            continue


        style = get_paragraph_style(
            paragraph
        )


        formatted_text = convert_heading(
            paragraph_text,
            style
        )


        paragraphs.append(
            formatted_text
        )


    # ==========================================
    # Return structured text
    # ==========================================

    return "\n\n".join(
        paragraphs
    )
