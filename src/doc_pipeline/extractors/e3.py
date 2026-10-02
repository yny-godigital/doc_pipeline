"""
E3 (and the OCR half of E5): layout-aware OCR for scanned / flattened pages.

Renders the PDF page to a raster image at a resolution good enough for OCR,
then runs tesseract with layout preservation (page segmentation mode 3:
"fully automatic page segmentation, no OSD"). Returns per-block text plus a
mean confidence score, which the classifier's ocr_quality_check() uses to
decide whether this page should actually be reclassified as E5.

Swap `run_tesseract` for a call to a layout-aware VLM (e.g. a hosted
document-intelligence endpoint) if tesseract's layout handling proves too
weak for your engineering document scans -- the return shape is the contract
the rest of the pipeline depends on, not the OCR engine itself.
"""
import pytesseract
from PIL import Image
import io


def render_page_to_image(fitz_page, zoom: float = 2.0) -> Image.Image:
    mat = __import__("pymupdf").Matrix(zoom, zoom)
    pix = fitz_page.get_pixmap(matrix=mat)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def run_tesseract(image: Image.Image) -> dict:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 3")
    words, confidences = [], []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        if not text:
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < 0:  # tesseract uses -1 for non-text regions
            continue
        words.append(text)
        confidences.append(c)

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "text": " ".join(words),
        "word_count": len(words),
        "mean_confidence": round(mean_conf, 1),
    }


def extract(fitz_page) -> dict:
    image = render_page_to_image(fitz_page)
    ocr_result = run_tesseract(image)
    return ocr_result

