import pytesseract
from PIL import Image
import os
import sys
import traceback


class OCRProcessor:
    def __init__(self, tesseract_path=None):
        """
        Initialize the OCR Processor with an optional custom Tesseract path.
        """
        self._set_tesseract_path(tesseract_path)

    def _set_tesseract_path(self, tesseract_path):
        """
        Configure pytesseract with a given path or fallback to default.
        """
        try:
            if tesseract_path and os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            else:
                # Default path (Windows installation)
                default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                pytesseract.pytesseract.tesseract_cmd = default_path
        except Exception as e:
            print(f"[ERROR] Failed to set Tesseract path: {e}")
            sys.exit(1)

    def _load_image(self, img_path):
        """
        Load an image safely, with error handling.
        """
        try:
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            return Image.open(img_path)
        except Exception as e:
            print(f"[ERROR] Could not load image: {img_path}\n{traceback.format_exc()}")
            sys.exit(1)

    def extract_text(self, img_path):
        """
        Perform OCR on the given image and return extracted text.
        """
        try:
            image = self._load_image(img_path)
            raw_text = pytesseract.image_to_string(image)
            return self._clean_text(raw_text)
        except Exception as e:
            print(f"[ERROR] OCR extraction failed: {e}")
            sys.exit(1)

    def _clean_text(self, text):
        """
        Clean up OCR text by stripping whitespace.
        """
        return text.strip()


def main():
    # Path configuration
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image_file = "image.png"

    # Create OCR processor
    processor = OCRProcessor(tesseract_path)

    # Extract text
    extracted_text = processor.extract_text(image_file)

    # Display result with formatting
    print("=" * 50)
    print("Extracted OCR Text".center(50))
    print("=" * 50)
    print(extracted_text)
    print("=" * 50)


if __name__ == "__main__":
    main()
