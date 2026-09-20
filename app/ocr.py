import shutil
import sys

import pytesseract


# Only override pytesseract's command on Windows, and only if the default
# install location actually exists. On Linux/Mac (or a different Windows
# install path), leave it alone so pytesseract falls back to whatever
# "tesseract" resolves to on PATH.
_DEFAULT_WINDOWS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if sys.platform.startswith("win") and shutil.which("tesseract") is None:

    from pathlib import Path

    if Path(_DEFAULT_WINDOWS_PATH).exists():

        pytesseract.pytesseract.tesseract_cmd = _DEFAULT_WINDOWS_PATH


def extract_text_from_image(image_path):

    text = pytesseract.image_to_string(
        image_path
    )

    return text.strip()