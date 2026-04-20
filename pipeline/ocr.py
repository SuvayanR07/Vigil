"""
OCR for image and PDF adverse-event documents.

Uses Tesseract (via pytesseract) for text extraction and pdf2image
(via poppler) for PDF page rendering. Both are fully local and free.
"""

from __future__ import annotations

import io
from typing import Optional


class OCRDependencyError(RuntimeError):
    """Raised when a required OCR dependency is missing."""


# --------------------------------------------------------------------------- #
# Lazy imports — keep the rest of the app loadable even if OCR deps are absent
# --------------------------------------------------------------------------- #

def _load_image_deps():
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise OCRDependencyError(
            "OCR requires `pytesseract` and `Pillow`. Install with:\n"
            "  pip install pytesseract Pillow\n"
            "Plus the Tesseract binary:\n"
            "  macOS : brew install tesseract\n"
            "  Ubuntu: sudo apt-get install tesseract-ocr"
        ) from e

    # Verify the tesseract binary is actually reachable
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise OCRDependencyError(
            "Tesseract binary not found on PATH. Install it:\n"
            "  macOS : brew install tesseract\n"
            "  Ubuntu: sudo apt-get install tesseract-ocr"
        ) from e

    return pytesseract, Image


def _load_pdf_deps():
    try:
        from pdf2image import convert_from_bytes
    except ImportError as e:
        raise OCRDependencyError(
            "PDF OCR requires `pdf2image` and the poppler binaries.\n"
            "  pip install pdf2image\n"
            "  macOS : brew install poppler\n"
            "  Ubuntu: sudo apt-get install poppler-utils"
        ) from e
    return convert_from_bytes


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #

def _preprocess_for_ocr(image):
    """Grayscale + light contrast bump for better OCR on phone photos."""
    from PIL import ImageOps
    gray = image.convert("L")
    # Autocontrast fixes uneven lighting on handwritten / phone-photo inputs
    return ImageOps.autocontrast(gray, cutoff=2)


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def extract_text_from_image(file_bytes: bytes, filename: str = "") -> str:
    """
    Run OCR on an image file.

    Args:
        file_bytes: Raw image bytes (PNG/JPG/JPEG).
        filename:   Original filename, used only for error messages.

    Returns:
        Extracted text (may be empty if the image has no readable text).

    Raises:
        OCRDependencyError: if pytesseract/Pillow/tesseract is missing.
        ValueError: if the bytes are not a valid image.
    """
    pytesseract, Image = _load_image_deps()

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # force-read so truncated files raise here, not later
    except Exception as e:
        raise ValueError(f"Could not read image '{filename}': {e}") from e

    preprocessed = _preprocess_for_ocr(image)
    text = pytesseract.image_to_string(preprocessed)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes, dpi: int = 200) -> str:
    """
    Render each PDF page to an image and OCR it, then concatenate.

    Args:
        file_bytes: Raw PDF bytes.
        dpi:        Render resolution. 200 is a solid default; go to 300
                    if the PDF has small handwriting.

    Returns:
        Concatenated text from all pages, separated by blank lines.

    Raises:
        OCRDependencyError: if pdf2image / poppler / tesseract is missing.
        ValueError:         if the bytes are not a valid PDF.
    """
    pytesseract, _Image = _load_image_deps()
    convert_from_bytes = _load_pdf_deps()

    try:
        pages = convert_from_bytes(file_bytes, dpi=dpi)
    except Exception as e:
        raise ValueError(f"Could not render PDF: {e}") from e

    page_texts: list[str] = []
    for i, page in enumerate(pages, start=1):
        preprocessed = _preprocess_for_ocr(page)
        text = pytesseract.image_to_string(preprocessed).strip()
        if text:
            page_texts.append(f"--- Page {i} ---\n{text}")

    return "\n\n".join(page_texts).strip()


def is_available() -> tuple[bool, Optional[str]]:
    """
    Probe whether OCR is ready to run.

    Returns:
        (True, None) if everything is installed.
        (False, reason) otherwise.
    """
    try:
        _load_image_deps()
        return True, None
    except OCRDependencyError as e:
        return False, str(e)
