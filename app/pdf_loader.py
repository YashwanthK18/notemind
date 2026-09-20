import pymupdf


def load_pdf_file(file_path):

    documents = []

    pdf = pymupdf.open(file_path)

    for page_number, page in enumerate(pdf, start=1):

        # ==========================================
        # Extract normal text from the page
        # ==========================================

        page_text = page.get_text().strip()

        if page_text:

            documents.append({
                "text": page_text,
                "source": file_path.name if hasattr(file_path, "name") else str(file_path),
                "type": "pdf",
                "page": page_number
            })


        # ==========================================
        # Extract images from the page
        # ==========================================

        images = page.get_images(full=True)

        for image_index, image in enumerate(images):

            xref = image[0]

            image_info = pdf.extract_image(xref)

            image_data = image_info["image"]
            image_extension = image_info["ext"]


            # ======================================
            # Image + page context
            # ======================================

            documents.append({
                "text": "",
                "source": file_path.name if hasattr(file_path, "name") else str(file_path),
                "type": "image",
                "page": page_number,
                "image_index": image_index,
                "image_data": image_data,
                "image_extension": image_extension,

                # IMPORTANT:
                # Store the complete page text with
                # the image so retrieval has context.
                "page_context": page_text
            })

    pdf.close()

    return documents