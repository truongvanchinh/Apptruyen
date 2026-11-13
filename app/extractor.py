"""Trích xuất lời thoại trong bong bóng thoại từ PDF/ảnh manga.

Pipeline:
  1. PDF -> ảnh (PyMuPDF)
  2. Tìm bong bóng: vùng gần-trắng, ít màu, hình dạng đặc (elip/bo tròn), có chữ tối bên trong
  3. Sắp xếp: trên -> dưới, cùng hàng thì trái -> phải
  4. OCR từng bong bóng (EasyOCR, tiếng Việt)
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .vi_fix import fix_vietnamese

# =========================
# CONFIG
# =========================

DPI = 200                  # đủ nét cho OCR, nhẹ hơn 300
MAX_PAGE_WIDTH = 1600      # thu nhỏ trang quá to để xử lý nhanh

MIN_AREA_RATIO = 0.004     # diện tích bong bóng / diện tích trang
MAX_AREA_RATIO = 0.25

WHITE_V = 235              # độ sáng tối thiểu của nền bong bóng
WHITE_S = 30               # độ bão hòa tối đa (nền trắng, không màu)
MIN_FILL_RATIO = 0.55      # diện tích contour / diện tích bbox (loại vùng méo, dài)
MIN_SOLIDITY = 0.80        # diện tích contour / diện tích bao lồi
MIN_TEXT_RATIO = 0.01      # tỉ lệ điểm chữ tối bên trong bong bóng
MAX_TEXT_RATIO = 0.45

OCR_LANGUAGES = ["vi"]
USE_GPU = False


@dataclass
class Bubble:
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    score: float = 0.0
    contour: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class PageResult:
    page: int
    width: int
    height: int
    bubbles: list[Bubble] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "width": self.width,
            "height": self.height,
            "bubbles": [{k: v for k, v in asdict(b).items() if k != "contour"} for b in self.bubbles],
        }


# =========================
# INPUT -> IMAGES (RGB)
# =========================

def _limit_size(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= MAX_PAGE_WIDTH:
        return image
    scale = MAX_PAGE_WIDTH / w
    return cv2.resize(image, (MAX_PAGE_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)


def pdf_to_images(pdf_path: str | Path) -> Iterator[np.ndarray]:
    import pymupdf as fitz

    with fitz.open(pdf_path) as doc:
        matrix = fitz.Matrix(DPI / 72, DPI / 72)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=fitz.csRGB)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            yield _limit_size(image.copy())


def pdf_page_count(pdf_path: str | Path) -> int:
    import pymupdf as fitz

    with fitz.open(pdf_path) as doc:
        return doc.page_count


def load_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return _limit_size(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# =========================
# DETECT SPEECH BUBBLES
# =========================

def detect_bubbles(image: np.ndarray) -> list[Bubble]:
    height, width = image.shape[:2]
    page_area = width * height

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 1. Mặt nạ vùng trắng (nền bong bóng)
    white = ((hsv[..., 2] >= WHITE_V) & (hsv[..., 1] <= WHITE_S)).astype(np.uint8) * 255

    # 2. Cắt mặt nạ theo nét viền để bong bóng không dính vào trời/nền sáng xung quanh
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))
    white[edges > 0] = 0

    # 3. Contour ngoài cùng: chữ bên trong bong bóng chỉ là "lỗ", tự được lấp
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Mặt nạ "chữ": điểm tối
    dark = (gray < 110).astype(np.uint8)

    bubbles: list[Bubble] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        ratio = area / page_area
        if not MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 40 or h < 25:
            continue
        aspect = w / h
        if not 0.25 <= aspect <= 6:
            continue

        fill_ratio = area / (w * h)
        if fill_ratio < MIN_FILL_RATIO:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area else 0
        if solidity < MIN_SOLIDITY:
            continue

        # Vùng chạm mép trang và quá rộng thường là nền/khoảng trắng giữa các khung
        touches = x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2
        if touches and w > width * 0.6:
            continue

        # Phải có chữ ở phần lõi bong bóng
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [contour - [x, y]], -1, 1, thickness=cv2.FILLED)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        inside = mask.sum()
        if inside == 0:
            continue
        text_ratio = float((dark[y:y + h, x:x + w] & mask).sum()) / inside
        if not MIN_TEXT_RATIO <= text_ratio <= MAX_TEXT_RATIO:
            continue

        score = round(min(1.0, (fill_ratio + solidity) / 2), 3)
        bubbles.append(Bubble(x=int(x), y=int(y), w=int(w), h=int(h), score=score, contour=contour))

    return remove_duplicate_boxes(bubbles)


def remove_duplicate_boxes(boxes: list[Bubble], threshold: float = 0.65) -> list[Bubble]:
    boxes = sorted(boxes, key=lambda b: b.w * b.h, reverse=True)
    result: list[Bubble] = []
    for box in boxes:
        duplicate = False
        for kept in result:
            x1, y1 = max(box.x, kept.x), max(box.y, kept.y)
            x2 = min(box.x + box.w, kept.x + kept.w)
            y2 = min(box.y + box.h, kept.y + kept.h)
            if x2 <= x1 or y2 <= y1:
                continue
            inter = (x2 - x1) * (y2 - y1)
            if inter / min(box.w * box.h, kept.w * kept.h) > threshold:
                duplicate = True
                break
        if not duplicate:
            result.append(box)
    return result


# =========================
# SORT: TOP -> BOTTOM, LEFT -> RIGHT
# =========================

def sort_bubbles(bubbles: list[Bubble]) -> list[Bubble]:
    """Gom các bong bóng có phần chiều dọc chồng lấn nhau thành một hàng,
    rồi đọc hàng từ trên xuống, trong hàng từ trái sang phải."""
    rows: list[list[Bubble]] = []
    for b in sorted(bubbles, key=lambda b: b.y):
        for row in rows:
            top = min(r.y for r in row)
            bottom = max(r.y + r.h for r in row)
            overlap = min(bottom, b.y + b.h) - max(top, b.y)
            if overlap > 0.5 * min(b.h, bottom - top):
                row.append(b)
                break
        else:
            rows.append([b])

    rows.sort(key=lambda row: min(r.y for r in row))
    return [b for row in rows for b in sorted(row, key=lambda r: r.x)]


# =========================
# OCR
# =========================

_reader = None
_reader_lock = threading.Lock()


@lru_cache(maxsize=1)
def available_engines() -> list[str]:
    engines = []
    try:
        import easyocr  # noqa: F401
        engines.append("easyocr")
    except Exception:
        pass
    try:
        import pytesseract

        if "vie" in pytesseract.get_languages(config=""):
            engines.append("tesseract")
    except Exception:
        pass
    return engines


def get_reader():
    global _reader
    with _reader_lock:
        if _reader is None:
            import easyocr

            _reader = easyocr.Reader(OCR_LANGUAGES, gpu=USE_GPU)
        return _reader


def _ocr_easyocr(crop: np.ndarray) -> str:
    results = get_reader().readtext(crop, detail=1, paragraph=False)
    # Sắp các dòng chữ trong bong bóng theo thứ tự đọc
    lines = sorted(results, key=lambda r: (round(min(p[1] for p in r[0]) / 20), min(p[0] for p in r[0])))
    return " ".join(r[1] for r in lines)


def _ocr_tesseract(crop: np.ndarray) -> str:
    import pytesseract

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return pytesseract.image_to_string(binary, lang="vie", config="--psm 6")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return ""
    # Chữ trong bong bóng thường viết hoa toàn bộ: sửa các chữ OCR đọc nhầm thành thường
    if sum(c.isupper() for c in letters) / len(letters) >= 0.6:
        text = text.upper()
    text = re.sub(r"\s+([,.!?…])", r"\1", text)
    if text.endswith("'"):  # dấu "!" cuối câu hay bị đọc thành "'"
        text = text[:-1] + "!"
    return text


TARGET_CHAR_HEIGHT = 23   # cỡ chữ (px) cho kết quả OCR tốt nhất


def _char_height(crop: np.ndarray) -> float | None:
    """Ước lượng chiều cao chữ từ các thành phần liên thông màu tối."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, _, stats, _ = cv2.connectedComponentsWithStats((gray < 110).astype(np.uint8))
    heights = stats[1:, 3]
    heights = heights[(heights > 4) & (heights < crop.shape[0] * 0.4)]
    if len(heights) < 3:
        return None
    return float(np.percentile(heights, 75))  # bỏ qua dấu thanh (thành phần nhỏ)


def ocr_bubble(image: np.ndarray, bubble: Bubble, engine: str | None = None, trim: bool = True,
               fix: bool = True) -> str:
    h, w = image.shape[:2]
    x1, y1 = max(0, bubble.x), max(0, bubble.y)
    x2, y2 = min(w, bubble.x + bubble.w), min(h, bubble.y + bubble.h)
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return ""

    if bubble.contour is not None:
        # Tô trắng phần tranh nằm ngoài bong bóng (các góc của khung chữ nhật)
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.drawContours(mask, [bubble.contour - [x1, y1]], -1, 255, thickness=cv2.FILLED)
        crop[mask == 0] = 255

    if trim:
        # Chỉ giữ phần lõi để không đọc nhầm nét viền bong bóng
        mx, my = int(crop.shape[1] * 0.06), int(crop.shape[0] * 0.06)
        crop = crop[my:crop.shape[0] - my, mx:crop.shape[1] - mx]

    # Chuẩn hoá cỡ chữ để PDF ở độ phân giải nào cũng cho kết quả tương đương
    char_h = _char_height(crop)
    scale = TARGET_CHAR_HEIGHT / char_h if char_h else 1.0
    scale = min(max(scale, 0.3), 4.0)
    if abs(scale - 1) > 0.1:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp)

    engine = engine or (available_engines() or ["easyocr"])[0]
    text = clean_text(_ocr_easyocr(crop) if engine == "easyocr" else _ocr_tesseract(crop))
    return fix_vietnamese(text) if fix else text


# =========================
# PIPELINE
# =========================

def process_image(image: np.ndarray, page_number: int, ocr: bool = True,
                  engine: str | None = None, fix: bool = True) -> PageResult:
    bubbles = sort_bubbles(detect_bubbles(image))
    if ocr:
        for b in bubbles:
            b.text = ocr_bubble(image, b, engine, fix=fix)
    h, w = image.shape[:2]
    return PageResult(page=page_number, width=w, height=h, bubbles=bubbles)


def iter_input_images(path: Path) -> tuple[int, Iterator[np.ndarray]]:
    if path.suffix.lower() == ".pdf":
        return pdf_page_count(path), pdf_to_images(path)
    return 1, iter([load_image(path)])


def dialogue_text(pages: list[PageResult]) -> str:
    return "\n".join(b.text for p in pages for b in p.bubbles if b.text)


def draw_debug(image: np.ndarray, page: PageResult) -> np.ndarray:
    out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, b in enumerate(page.bubbles, 1):
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 0, 255), 3)
        cv2.putText(out, str(i), (b.x + 4, b.y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return out
