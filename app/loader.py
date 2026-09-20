from pathlib import Path

import pymupdf

from app.docx_loader import load_docx_file


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx"
}


def load_text_file(file_path):

    return file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

def load_pdf_file(file_path):

    pdf = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(pdf):

        page_data = {
            "page": page_number + 1,
            "text": page.get_text(),
            "images": []
        }

        image_list = page.get_images(
            full=True
        )

        for image_index, image in enumerate(
            image_list
        ):

            xref = image[0]

            image_data = pdf.extract_image(
                xref
            )

            image_bytes = image_data["image"]

            image_ext = image_data["ext"]

            page_data["images"].append({
                "index": image_index,
                "extension": image_ext,
                "data": image_bytes
            })

        pages.append(page_data)

    pdf.close()

    return pages


def load_documents(folder_path):

    documents = []

    folder = Path(folder_path)

    for file_path in folder.rglob("*"):

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        relative_path = str(
            file_path.relative_to(folder)
        )

        # TXT / Markdown
        if extension in {".txt", ".md"}:

            text = load_text_file(
                file_path
            )

            if text.strip():

                documents.append({
                    "source": relative_path,
                    "type": extension,
                    "text": text
                })

        # PDF
        elif extension == ".pdf":

            pages = load_pdf_file(
                file_path
            )

            for page in pages:

                # Add page text
                if page["text"].strip():

                    documents.append({
                        "source": relative_path,
                        "type": ".pdf",
                        "page": page["page"],
                        "text": page["text"]
                    })

                # Add images
                for image in page["images"]:

                    documents.append({
                        "source": relative_path,
                        "type": "image",
                        "page": page["page"],
                        "image_index": image["index"],
                        "image_extension": image["extension"],
                        "image_data": image["data"],

                        # IMPORTANT:
                        # Store the page's text alongside the image so
                        # ingest.py can prepend it as PAGE CONTEXT. Without
                        # this, image chunks only ever have raw OCR text,
                        # even though the real caption usually lives in the
                        # page text.
                        "page_context": page["text"]
                    })

        # DOCX
        elif extension == ".docx":

            text = load_docx_file(
                file_path
            )

            if text.strip():

                documents.append({
                    "source": relative_path,
                    "type": extension,
                    "text": text
                })

    return documents