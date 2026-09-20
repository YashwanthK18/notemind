from pathlib import Path
import json
import hashlib
import time

from app.image_extractor import save_extracted_image
from app.ocr import extract_text_from_image


# =========================================================
# OCR CACHE DIRECTORY
# =========================================================

CACHE_DIR = Path("data/ocr_cache")

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CREATE CACHE PATH
# =========================================================

def get_cache_path(
    source,
    page,
    image_index
):
    """
    Create a unique cache filename for each image.
    """

    key = (
        f"{source}|"
        f"{page}|"
        f"{image_index}"
    )

    filename = hashlib.md5(
        key.encode("utf-8")
    ).hexdigest()

    return CACHE_DIR / f"{filename}.json"


# =========================================================
# LOAD CACHED RESULT
# =========================================================

def load_cached_result(
    cache_path
):
    """
    Load previously processed OCR result.
    """

    if not cache_path.exists():
        return None

    try:

        with open(
            cache_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as error:

        print(
            f"  Warning: Could not read OCR cache: {error}",
            flush=True
        )

        return None


# =========================================================
# SAVE CACHED RESULT
# =========================================================

def save_cached_result(
    cache_path,
    text
):
    """
    Save OCR text to cache.
    """

    data = {
        "text": text or ""
    }

    try:

        with open(
            cache_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

    except Exception as error:

        print(
            f"  Warning: Could not save OCR cache: {error}",
            flush=True
        )


# =========================================================
# PROCESS IMAGE
# =========================================================

def process_image(
    image_data,
    source,
    page,
    image_index,
    extension
):
    """
    Extract an image, run OCR if necessary,
    cache the OCR result, and return metadata.

    AI image understanding is intentionally NOT
    performed during ingestion.
    """

    start_time = time.time()

    # =====================================================
    # SAVE EXTRACTED IMAGE
    # =====================================================

    image_path = save_extracted_image(
        image_data=image_data,
        output_folder="data/images",
        source_name=source,
        page_number=page,
        image_index=image_index,
        extension=extension
    )

    # =====================================================
    # CACHE PATH
    # =====================================================

    cache_path = get_cache_path(
        source,
        page,
        image_index
    )

    # =====================================================
    # CHECK CACHE
    # =====================================================

    cached = load_cached_result(
        cache_path
    )

    if cached is not None:

        print(
            "  Using cached OCR result.",
            flush=True
        )

        text = cached.get(
            "text",
            ""
        )

    else:

        # =================================================
        # RUN OCR
        # =================================================

        print(
            "  Running OCR...",
            flush=True
        )

        ocr_start = time.time()

        try:

            text = extract_text_from_image(
                image_path
            )

        except Exception as error:

            print(
                f"  OCR failed: {error}",
                flush=True
            )

            text = ""

        ocr_time = time.time() - ocr_start

        print(
            f"  OCR completed in {ocr_time:.1f}s",
            flush=True
        )

        # =================================================
        # SAVE OCR CACHE
        # =================================================

        save_cached_result(
            cache_path,
            text
        )

        print(
            "  OCR result cached.",
            flush=True
        )

    # =====================================================
    # AI IMAGE UNDERSTANDING
    # =====================================================

    # IMPORTANT:
    #
    # We intentionally do NOT call describe_image()
    # here.
    #
    # AI image understanding will be performed later
    # only when the user asks a question that requires
    # visual understanding.
    #
    # This keeps ingestion much faster.

    description = ""

    # =====================================================
    # COMBINE OCR
    # =====================================================

    combined_text = ""

    if text and text.strip():

        combined_text += (
            "IMAGE OCR:\n"
            + text.strip()
        )

    if description and description.strip():

        if combined_text:

            combined_text += "\n\n"

        combined_text += (
            "IMAGE DESCRIPTION:\n"
            + description.strip()
        )

    # =====================================================
    # TOTAL PROCESSING TIME
    # =====================================================

    total_time = time.time() - start_time

    print(
        f"  Image processing completed in "
        f"{total_time:.1f}s",
        flush=True
    )

    # =====================================================
    # RETURN IMAGE METADATA
    # =====================================================

    return {
        "text": combined_text,
        "source": source,
        "type": "image",
        "page": page,
        "image_index": image_index,
        "image_path": image_path
    }