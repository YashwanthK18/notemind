from pathlib import Path


def save_extracted_image(
    image_data,
    output_folder,
    source_name,
    page_number,
    image_index,
    extension
):
    output_folder = Path(output_folder)
    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_name = Path(source_name).stem

    filename = (
        f"{safe_name}"
        f"_page_{page_number}"
        f"_image_{image_index}"
        f".{extension}"
    )

    output_path = output_folder / filename

    with open(output_path, "wb") as file:
        file.write(image_data)

    return str(output_path)