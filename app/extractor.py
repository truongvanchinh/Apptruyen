"""Trích xuất lời thoại trong bong bóng thoại từ PDF/ảnh manga, webtoon.

Pipeline (tìm chữ trước, rồi mới xét bong bóng):
  1. PDF -> ảnh ở độ phân giải gốc, cắt dải đen hai bên (ảnh chụp màn hình webtoon)
  2. Ghép 40% trang trước/sau làm ngữ cảnh, để bong bóng bị cắt giữa 2 trang vẫn nguyên vẹn
  3. Dò các dòng chữ (CRAFT của EasyOCR), gom dòng gần nhau thành khối
  4. Khối là bong bóng thoại nếu: nền quanh chữ đồng màu VÀ vùng nền đó được bao kín
     (bắn tia từ khối chữ ra mọi hướng, đa số tia phải chạm viền ở khoảng cách gần)
  5. Sắp xếp trên -> dưới, trái -> phải; OCR từng khối; sửa lỗi chính tả
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .vi_fix import agreed_words, fix_vietnamese, merge_ocr

# =========================
# CONFIG
# =========================

DPI = 200                  # khi trang PDF không phải một ảnh duy nhất
MAX_PAGE_WIDTH = 1200      # thu nhỏ trang quá to để xử lý nhanh
CONTEXT_RATIO = 0.4        # phần trang trước/sau ghép vào làm ngữ cảnh

BG_TOLERANCE = 50          # sai khác màu tối đa so với màu nền bong bóng
FLAT_TOLERANCE = 28        # nền bong bóng phẳng: sai khác màu trong khung chữ
MIN_BG_FLAT = 0.35         # tỉ lệ điểm ảnh trong khung chữ đúng màu nền
EDGE_STEP = 40             # bước nhảy màu tối thiểu để coi là viền bong bóng
N_RAYS = 48
MIN_RAY_HITS = 0.8         # tỉ lệ tia phải chạm viền bong bóng
MAX_TEXT_SATURATION = 45   # chữ có màu (không đen/xám/trắng) = watermark/hiệu ứng

SKIP_BIG_TEXT = True       # bỏ chữ nét cọ / tiếng hét cỡ lớn (OCR đọc kém, không phải lời thoại thường)
BIG_TEXT_WIDTH_RATIO = 0.12  # dòng chữ cao hơn 12% bề ngang trang
BIG_TEXT_RATIO = 2.5         # hoặc cao gấp 2,5 lần cỡ chữ lời thoại điển hình của tài liệu

TARGET_CHAR_HEIGHT = 23    # cỡ chữ (px) cho kết quả OCR tốt nhất
WATERMARK_RE = re.compile(r"\w\s*\.\s*(shop|com|net|org|vn|info|xyz|site|top|club)\b", re.I)
WATERMARK_WORDS = re.compile(r"manga|merge|cloud|\bcom\b|\bshop\b|team", re.I)  # chỉ xét khi chữ không dấu
OCR_LANGUAGES = ["vi"]
USE_GPU = False


@dataclass
class Line:
    """Một dòng chữ do bộ dò chữ trả về."""
    poly: np.ndarray  # 4 điểm (tl, tr, br, bl)
    x1: int
    y1: int
    x2: int
    y2: int
    h: float          # chiều cao dòng
    angle: float      # độ nghiêng (độ)


@dataclass
class Bubble:
    x: int
    y: int
    w: int
    h: int
    text: str = ""
    score: float = 0.0
    lines: list[Line] = field(default_factory=list, repr=False, compare=False)
    bg: tuple = field(default=(255, 255, 255), repr=False, compare=False)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "text": self.text, "score": self.score}


@dataclass
class PageResult:
    page: int
    width: int
    height: int
    bubbles: list[Bubble] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"page": self.page, "width": self.width, "height": self.height,
                "bubbles": [b.to_dict() for b in self.bubbles]}


# =========================
# INPUT -> IMAGES (RGB)
# =========================

def _limit_size(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= MAX_PAGE_WIDTH:
        return image
    scale = MAX_PAGE_WIDTH / w
    return cv2.resize(image, (MAX_PAGE_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)


def _render_page(page) -> np.ndarray:
    import pymupdf

    zoom, clip = DPI / 72, None
    # Trang chỉ gồm một ảnh lớn (truyện scan/chụp màn hình): lấy đúng độ phân giải gốc của ảnh
    images = page.get_images(full=True)
    if len(images) == 1:
        rects = page.get_image_rects(images[0][0])
        if rects:
            rect = rects[0] & page.rect
            if rect.width > 0 and rect.get_area() >= 0.5 * page.rect.get_area():
                zoom, clip = images[0][2] / rects[0].width, rect
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False,
                          colorspace=pymupdf.csRGB)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return _limit_size(image.copy())


def _side_bars(image: np.ndarray) -> np.ndarray:
    """Cột nào gần như đen kín (dải đen hai bên của ảnh chụp màn hình)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return (gray < 35).mean(axis=0) >= 0.985


def _find_trim(samples: list[np.ndarray]) -> tuple[int, int]:
    """Số cột cần cắt bên trái/phải, tính chung cho cả tài liệu."""
    if not samples or len({s.shape[1] for s in samples}) != 1:
        return 0, 0
    # theo đa số trang: vài trang có thể bị hình khác đè lên dải đen
    bars = np.mean([_side_bars(s) for s in samples], axis=0) >= 0.6
    width = len(bars)
    left = 0
    while left < width and bars[left]:
        left += 1
    right = 0
    while right < width - left and bars[width - 1 - right]:
        right += 1
    if left + right > width * 0.6:  # trang gần như đen kín: không cắt
        return 0, 0
    return left, right


def pdf_page_count(pdf_path: str | Path) -> int:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return doc.page_count


def pdf_to_images(pdf_path: str | Path) -> Iterator[np.ndarray]:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        count = doc.page_count
        picks = sorted({int(i * (count - 1) / 7) for i in range(8)}) if count > 1 else [0]
        left, right = _find_trim([_render_page(doc[i]) for i in picks])
        width = None
        for page in doc:
            image = _render_page(page)
            if width is None:
                width = image.shape[1]
            if image.shape[1] == width and (left or right):
                image = image[:, left:image.shape[1] - right]
            yield np.ascontiguousarray(image)


def load_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return _limit_size(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def iter_input_images(path: Path) -> tuple[int, Iterator[np.ndarray]]:
    if path.suffix.lower() == ".pdf":
        return pdf_page_count(path), pdf_to_images(path)
    return 1, iter([load_image(path)])


# =========================
# OCR ENGINES
# =========================

_reader = None
_reader_lock = threading.Lock()


@lru_cache(maxsize=1)
def available_engines() -> list[str]:
    """EasyOCR bắt buộc (dùng để dò chữ); Tesseract là lựa chọn thêm cho bước đọc chữ."""
    engines = []
    try:
        import easyocr  # noqa: F401
        engines.append("easyocr")
    except Exception:
        return []
    try:
        import pytesseract

        if "vie" in pytesseract.get_languages(config=""):
            engines = ["ensemble", "easyocr", "tesseract"]  # ghép cả hai là chính xác nhất
    except Exception:
        pass
    return engines


def get_reader():
    global _reader
    with _reader_lock:
        if _reader is None:
            import easyocr

            _reader = easyocr.Reader(OCR_LANGUAGES, gpu=USE_GPU, verbose=False)
        return _reader


# =========================
# 3. DÒ CHỮ + GOM KHỐI
# =========================

def _make_line(poly) -> Line:
    poly = np.array(poly, dtype=np.float32).reshape(4, 2)
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    top = poly[1] - poly[0]
    side = poly[3] - poly[0]
    return Line(poly=poly, x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                h=float(max(4.0, np.hypot(*side))),
                angle=float(math.degrees(math.atan2(top[1], top[0]))))


def detect_text_lines(image: np.ndarray) -> list[Line]:
    horizontal, free = get_reader().detect(image, canvas_size=2560, width_ths=0.7)
    lines = [_make_line([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]) for x1, x2, y1, y2 in horizontal[0]]
    lines += [_make_line(poly) for poly in free[0]]
    lines = [ln for ln in lines if ln.x2 - ln.x1 >= 6 and ln.y2 - ln.y1 >= 6]
    return [ln for ln in lines if _on_flat_background(image, ln)]


def _on_flat_background(image: np.ndarray, line: Line) -> bool:
    """Dòng chữ của lời thoại: nằm trên nền phẳng, mực không màu (đen/xám/trắng).
    Loại sớm chữ đè lên tranh và watermark để chúng không bị gom chung khối với lời thoại."""
    px = image[max(0, line.y1):line.y2, max(0, line.x1):line.x2].reshape(-1, 3).astype(np.int16)
    if len(px) == 0:
        return False
    dist = np.abs(px - np.median(px, axis=0)).max(axis=1)
    bg_frac = (dist < FLAT_TOLERANCE).mean()
    ink = px[dist > 90]
    ink_frac = len(ink) / len(px)
    if bg_frac < 0.3 or bg_frac + ink_frac < 0.6:
        return False
    if ink_frac < 0.03:
        return False  # chữ mờ, tương phản thấp: watermark bán trong suốt
    # tính theo từng điểm ảnh: watermark nhiều màu lấy trung vị sẽ ra xám
    colored = ((ink.max(axis=1) - ink.min(axis=1)) > MAX_TEXT_SATURATION).mean()
    if colored > 0.4:
        return False
    # mực phải đậm hẳn so với nền: chữ đen trên nền sáng (hoặc trắng trên nền tối)
    ink_level = float(np.median(ink.max(axis=1)))
    bg = np.median(px, axis=0)
    if bg.min() > 170:      # nền sáng
        return ink_level < 130
    if bg.max() < 85:       # nền tối
        return ink_level > 125
    return False            # nền màu lưng chừng: mảng tranh phẳng, không phải bong bóng


def _connected(a: Line, b: Line) -> bool:
    h = min(a.h, b.h)
    if max(a.h, b.h) / h > 2.2:  # cỡ chữ khác hẳn nhau: không cùng một lời thoại
        return False
    if abs(a.angle - b.angle) > 15:  # độ nghiêng khác nhau (vd chữ nghiêng và watermark nằm ngang)
        return False
    x_gap = max(a.x1, b.x1) - min(a.x2, b.x2)
    y_gap = max(a.y1, b.y1) - min(a.y2, b.y2)
    stacked = y_gap < 0.9 * h and x_gap < -0.2 * h      # dòng trên / dòng dưới
    side_by_side = y_gap < -0.5 * h and x_gap < 1.5 * h  # cùng một dòng bị tách đôi
    return stacked or side_by_side


def group_lines(lines: list[Line]) -> list[list[Line]]:
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if _connected(lines[i], lines[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[Line]] = {}
    for i, line in enumerate(lines):
        groups.setdefault(find(i), []).append(line)
    return list(groups.values())


def _bbox(lines: list[Line]) -> tuple[int, int, int, int]:
    return (min(l.x1 for l in lines), min(l.y1 for l in lines),
            max(l.x2 for l in lines), max(l.y2 for l in lines))


# =========================
# 4. KHỐI CHỮ CÓ NẰM TRONG BONG BÓNG?
# =========================

TRACE: list | None = None   # gán = [] để ghi lại lý do loại từng khối chữ (chẩn đoán)
TRACE_RAYS: list | None = None  # gán = [] để ghi lại điểm kết thúc từng tia: (x, y, loại)


def _trace(block: list[Line], reason: str) -> None:
    if TRACE is not None:
        TRACE.append((_bbox(block), reason))


def analyze_block(image: np.ndarray, block: list[Line], text_mask: np.ndarray) -> Bubble | None:
    """Trả về Bubble nếu khối chữ nằm trong một vùng nền đồng màu được bao kín.
    score = 0: nền phẳng, chữ hợp lệ nhưng không tự chứng minh được là bị bao kín ("mồ côi")."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = _bbox(block)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None

    # Màu nền = màu chiếm đa số trong khung chữ. Bong bóng có nền phẳng: điểm ảnh hoặc là nền,
    # hoặc là mực; còn chữ đè lên tranh (hiệu ứng, watermark) thì màu biến thiên liên tục.
    inner = image[y1:y2, x1:x2].reshape(-1, 3).astype(np.int16)
    bg = np.median(inner, axis=0)
    dist = np.abs(inner - bg).max(axis=1)
    bg_frac = (dist < FLAT_TOLERANCE).mean()
    ink = inner[dist > 90]
    ink_frac = len(ink) / len(inner)
    if bg_frac < MIN_BG_FLAT or ink_frac < 0.02 or bg_frac + ink_frac < 0.7:
        _trace(block, f"nền không phẳng bg={bg_frac:.2f} ink={ink_frac:.2f}")
        return None

    colored = ((ink.max(axis=1) - ink.min(axis=1)) > MAX_TEXT_SATURATION).mean()
    if colored > 0.4:
        _trace(block, f"chữ màu {colored:.2f}")
        return None  # chữ có màu: watermark, hiệu ứng

    # Vùng khảo sát quanh khối chữ
    limit = int(1.2 * max(bw, bh) + 40)
    rx1, ry1 = max(0, x1 - limit), max(0, y1 - limit)
    rx2, ry2 = min(W, x2 + limit), min(H, y2 + limit)
    region = image[ry1:ry2, rx1:rx2].astype(np.int16)
    diff = np.abs(region - bg).max(axis=2).astype(np.uint8)
    same = (diff < BG_TOLERANCE).astype(np.uint8)
    # Viền thật là một bước nhảy màu sắc nét; tranh chuyển màu dần thì không có
    sharp = cv2.morphologyEx(diff, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
    sharp = cv2.dilate(sharp, np.ones((9, 9), np.uint8)) >= EDGE_STEP
    # Nở phần "không phải nền" để bịt khe giữa các chấm của viền bong bóng dạng chấm
    k = 2 * max(1, round(W / 450)) + 1
    same = cv2.erode(same, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Chữ (của khối này và khối khác cùng bong bóng) không được tính là viền
    same[text_mask[ry1:ry2, rx1:rx2] > 0] = 1

    cx, cy = (x1 + x2) / 2 - rx1, (y1 + y2) / 2 - ry1
    rh, rw = same.shape
    near = 0.6 * max(bw, bh) + 20   # mép trang chỉ được coi là viền nếu nằm sát khối chữ
    hits, travel, edge_hits, soft, lost = [], [], 0, 0, 0
    for i in range(N_RAYS):
        theta = 2 * math.pi * i / N_RAYS
        dx, dy = math.cos(theta), math.sin(theta)
        # quãng đường tới khi tia ra khỏi khung chữ; tia được đi thêm tối đa `limit` nữa
        exit_t = min((bw / 2 + 2) / abs(dx) if abs(dx) > 1e-6 else 1e9,
                     (bh / 2 + 2) / abs(dy) if abs(dy) > 1e-6 else 1e9)
        t, end, found = 0.0, exit_t + limit, False
        while t < end:
            px, py = int(cx + dx * t), int(cy + dy * t)
            if not (0 <= px < rw and 0 <= py < rh):
                # Bong bóng bị mép trái/phải của trang cắt ngang: mép trang đóng vai trò viền
                at_side = (px + rx1 < 0 or px + rx1 >= W)
                if at_side and t - exit_t <= near:
                    edge_hits += 1
                    hits.append((min(max(px + rx1, 0), W - 1), py + ry1))
                    travel.append(1e9)  # không dùng để tính khung
                    found = True
                if TRACE_RAYS is not None:
                    TRACE_RAYS.append((px + rx1, py + ry1, "edge" if found else "out"))
                break
            if not same[py, px]:
                if sharp[py, px]:
                    hits.append((px + rx1, py + ry1))
                    travel.append(max(0.0, t - exit_t))
                    found = True
                else:
                    soft += 1
                if TRACE_RAYS is not None:
                    TRACE_RAYS.append((px + rx1, py + ry1, "hit" if found else "soft"))
                break
            t += 1.0
        if not found and t >= end:
            lost += 1
            if TRACE_RAYS is not None:
                TRACE_RAYS.append((int(cx + dx * t) + rx1, int(cy + dy * t) + ry1, "far"))

    ratio = len(hits) / N_RAYS
    # Nền không trắng/đen hẳn (mảng màu phẳng trong tranh) thì đòi hỏi bao kín gần như tuyệt đối
    plain = bg.min() >= 215 or bg.max() <= 50
    need = MIN_RAY_HITS if plain else 0.95
    detail = f"tia={ratio:.2f} mép={edge_hits} mềm={soft} xa={lost}"
    if ratio < need or edge_hits > 0.4 * N_RAYS:
        _trace(block, "không bao kín " + detail)
        # chữ trôi nổi trên nền trang, hoặc chữ lẻ tách khỏi khối chính của một bong bóng
        return Bubble(x=x1, y=y1, w=bw, h=bh, score=0.0, lines=list(block), bg=tuple(int(v) for v in bg))
    _trace(block, "OK " + detail)

    # Khung bong bóng = các điểm chạm gần; bỏ tia lọt qua chỗ chữ đè lên viền rồi chạm thứ khác ở xa
    travel_arr = np.array(travel)
    real = travel_arr[travel_arr < 1e8]
    cutoff = 2.5 * float(np.median(real)) + 20 if len(real) else 0
    pts = np.array(hits)[travel_arr <= cutoff]
    if len(pts) == 0:
        pts = np.array([[x1, y1], [x2, y2]])
    bx1, by1 = min(x1, pts[:, 0].min()), min(y1, pts[:, 1].min())
    bx2, by2 = max(x2, pts[:, 0].max()), max(y2, pts[:, 1].max())
    return Bubble(x=int(bx1), y=int(by1), w=int(bx2 - bx1), h=int(by2 - by1), score=round(ratio, 3),
                  lines=list(block), bg=tuple(int(v) for v in bg))


def _overlap(a: Bubble, b: Bubble) -> float:
    """IoU của hai khung bong bóng."""
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (a.w * a.h + b.w * b.h - inter)


def merge_bubbles(bubbles: list[Bubble]) -> list[Bubble]:
    """Hai khối chữ cho ra cùng một vùng bong bóng -> gộp thành một lời thoại."""
    result: list[Bubble] = []
    for b in sorted(bubbles, key=lambda b: b.w * b.h, reverse=True):
        for kept in result:
            if _overlap(b, kept) > 0.5:
                x1, y1 = min(kept.x, b.x), min(kept.y, b.y)
                x2, y2 = max(kept.x + kept.w, b.x + b.w), max(kept.y + kept.h, b.y + b.h)
                kept.x, kept.y, kept.w, kept.h = x1, y1, x2 - x1, y2 - y1
                kept.lines.extend(b.lines)
                break
        else:
            result.append(b)
    return result


def detect_bubbles(image: np.ndarray) -> list[Bubble]:
    lines = detect_text_lines(image)
    # Mặt nạ chữ theo đúng đa giác từng dòng (chữ nghiêng không che nhầm viền bong bóng)
    text_mask = np.zeros(image.shape[:2], np.uint8)
    for line in lines:
        pad = 2 * (int(line.h * 0.1) + 1) + 1   # nở ít thôi: viền bong bóng có thể nằm sát chữ
        one = np.zeros_like(text_mask)
        cv2.fillPoly(one, [cv2.convexHull(line.poly.astype(np.int32))], 1)
        text_mask |= cv2.dilate(one, np.ones((pad, pad), np.uint8))

    bubbles, orphans = [], []
    for block in group_lines(lines):
        bubble = analyze_block(image, block, text_mask)
        if bubble:
            (bubbles if bubble.score > 0 else orphans).append(bubble)
    bubbles = merge_bubbles(bubbles)

    # Chữ lẻ nằm gọn trong một bong bóng đã nhận (cùng màu nền) thuộc về bong bóng đó
    for orphan in orphans:
        cx, cy = orphan.x + orphan.w / 2, orphan.y + orphan.h / 2
        for b in bubbles:
            same_bg = max(abs(p - q) for p, q in zip(orphan.bg, b.bg)) < 30
            if same_bg and b.x <= cx <= b.x + b.w and b.y <= cy <= b.y + b.h:
                b.lines.extend(orphan.lines)
                break
    return bubbles


# =========================
# SORT: TOP -> BOTTOM, LEFT -> RIGHT
# =========================

def sort_bubbles(bubbles: list[Bubble]) -> list[Bubble]:
    """Gom các bong bóng chồng lấn nhau theo chiều dọc thành một hàng,
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
# 5. OCR
# =========================

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


def is_junk(text: str) -> bool:
    """Không phải lời thoại: rỗng, watermark (tên miền), hoặc một phụ âm lẻ loi do nét vẽ."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    if WATERMARK_RE.search(text):
        return True
    ascii_only = all(ord(c) < 128 for c in letters)
    if ascii_only and WATERMARK_WORDS.search(text):
        return True
    return len(letters) == 1 and letters[0].upper() not in "AĂÂEÊIOÔƠUƯYÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỪỨỬỮỰỜỚỞỠỢỒỐỔỖỘ"


def _ink_mask(crop: np.ndarray, bg) -> np.ndarray:
    return (np.abs(crop.astype(np.int16) - np.array(bg, np.int16)).max(axis=2) > 110).astype(np.uint8)


def _char_height(crop: np.ndarray, bg) -> float | None:
    """Ước lượng chiều cao chữ từ các thành phần liên thông khác màu nền."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(_ink_mask(crop, bg))
    heights = stats[1:, 3]
    heights = heights[(heights > 4) & (heights < crop.shape[0] * 0.6)]
    if len(heights) < 3:
        return None
    return float(np.percentile(heights, 75))  # bỏ qua dấu thanh (thành phần nhỏ)


def _ocr_easyocr(crop: np.ndarray) -> str:
    results = get_reader().readtext(crop, detail=1, paragraph=False, width_ths=0.7)
    items = []
    for box, text, _ in results:
        ys = [p[1] for p in box]
        items.append((min(ys), max(ys), min(p[0] for p in box), text))
    # Xếp thành hàng theo tâm dọc, trong hàng từ trái sang phải
    items.sort(key=lambda it: (it[0] + it[1]) / 2)
    rows: list[list] = []
    for it in items:
        center = (it[0] + it[1]) / 2
        if rows and abs(center - rows[-1][0]) < 0.5 * (it[1] - it[0]):
            rows[-1][1].append(it)
        else:
            rows.append([center, [it]])
    return " ".join(it[3] for _, row in rows for it in sorted(row, key=lambda it: it[2]))


def _ocr_tesseract(crop: np.ndarray, dark_bg: bool) -> str:
    import pytesseract

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    if dark_bg:
        gray = 255 - gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    return pytesseract.image_to_string(binary, lang="vie", config="--psm 6")


def ocr_crop(crop: np.ndarray, bg=(255, 255, 255), engine: str | None = None, fix: bool = True) -> str:
    if crop.size == 0:
        return ""
    # Chuẩn hoá cỡ chữ để ảnh ở độ phân giải nào cũng cho kết quả tương đương
    char_h = _char_height(crop, bg)
    scale = min(max(TARGET_CHAR_HEIGHT / char_h if char_h else 1.0, 0.3), 4.0)
    if abs(scale - 1) > 0.1:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp)
    pad = 16
    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[int(v) for v in bg])

    engine = engine or available_engines()[0]
    dark = sum(bg) < 380
    confirmed: frozenset = frozenset()
    if engine == "tesseract":
        text = clean_text(_ocr_tesseract(crop, dark))
    else:
        text = clean_text(_ocr_easyocr(crop))
        if engine == "ensemble" and text:
            # Tesseract đọc dấu tốt hơn ở cỡ chữ lớn hơn
            big = cv2.resize(crop, None, fx=1.55, fy=1.55, interpolation=cv2.INTER_CUBIC)
            tess = clean_text(_ocr_tesseract(big, dark))
            confirmed = agreed_words(text, tess)   # hai bộ đọc giống nhau: không sửa theo tần suất
            text = merge_ocr(text, tess)
    return fix_vietnamese(text, confirmed) if fix else text


def ocr_bubble(image: np.ndarray, bubble: Bubble, engine: str | None = None, fix: bool = True) -> str:
    crop, bg = bubble_crop(image, bubble)
    return ocr_crop(crop, bg, engine, fix)


def bubble_crop(image: np.ndarray, bubble: Bubble) -> tuple[np.ndarray, tuple]:
    """Ảnh phần chữ của bong bóng: chỉ giữ các dòng chữ đã dò, phần còn lại tô màu nền."""
    H, W = image.shape[:2]
    if not bubble.lines:  # vùng người dùng tự khoanh
        crop = image[max(0, bubble.y):min(H, bubble.y + bubble.h), max(0, bubble.x):min(W, bubble.x + bubble.w)]
        if crop.size == 0:
            return crop, (255, 255, 255)
        return crop, tuple(int(v) for v in np.median(crop.reshape(-1, 3), axis=0))

    x1, y1, x2, y2 = _bbox(bubble.lines)
    pad = int(0.4 * float(np.median([l.h for l in bubble.lines]))) + 2
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    crop = image[y1:y2, x1:x2].copy()

    mask = np.zeros(crop.shape[:2], np.uint8)
    for line in bubble.lines:
        cv2.fillPoly(mask, [cv2.convexHull((line.poly - [x1, y1]).astype(np.int32))], 255)
    mask = cv2.dilate(mask, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    crop[mask == 0] = bubble.bg

    # Chữ nghiêng: xoay cho nằm ngang trước khi đọc
    angle = float(np.median([l.angle for l in bubble.lines]))
    if abs(angle) > 4:
        h, w = crop.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        matrix[0, 2] += nw / 2 - w / 2
        matrix[1, 2] += nh / 2 - h / 2
        crop = cv2.warpAffine(crop, matrix, (nw, nh), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=bubble.bg)
    return crop, bubble.bg


# =========================
# PIPELINE
# =========================

def _fit_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    return cv2.resize(image, (width, max(1, round(image.shape[0] * width / image.shape[1]))))


def process_image(image: np.ndarray, page_number: int, ocr: bool = True, engine: str | None = None,
                  fix: bool = True, prev: np.ndarray | None = None, nxt: np.ndarray | None = None,
                  heights: list[float] | None = None) -> PageResult:
    """Xử lý một trang. `prev`/`nxt` là trang liền trước/sau (nếu có) để ghép ngữ cảnh:
    bong bóng bị cắt ở mép trang sẽ được nhìn thấy trọn vẹn và gán cho trang chứa tâm khối chữ.
    `heights`: cỡ chữ của các lời thoại đã nhận trước đó trong tài liệu (để nhận ra chữ to bất thường)."""
    H, W = image.shape[:2]
    parts, offset = [], 0
    if prev is not None:
        prev = _fit_width(prev, W)
        top = prev[-int(prev.shape[0] * CONTEXT_RATIO):]
        parts.append(top)
        offset = top.shape[0]
    parts.append(image)
    if nxt is not None:
        nxt = _fit_width(nxt, W)
        parts.append(nxt[:int(nxt.shape[0] * CONTEXT_RATIO)])
    window = np.vstack(parts) if len(parts) > 1 else image

    bubbles = []
    for b in detect_bubbles(window):
        _, ty1, _, ty2 = _bbox(b.lines)
        if offset <= (ty1 + ty2) / 2 < offset + H:  # còn lại thuộc trang trước/sau
            bubbles.append(b)

    heights = heights if heights is not None else []
    kept = []
    for b in sort_bubbles(bubbles):
        # Chữ nét cọ / tiếng hét, tiếng động lớn: to bất thường so với lời thoại -> bỏ, không đọc
        line_h = float(np.median([l.h for l in b.lines]))
        typical = float(np.median(heights)) if len(heights) >= 5 else None
        if SKIP_BIG_TEXT and (line_h > BIG_TEXT_WIDTH_RATIO * W or (typical and line_h > BIG_TEXT_RATIO * typical)):
            continue
        heights.append(line_h)
        if ocr:
            b.text = ocr_bubble(window, b, engine, fix)
            if is_junk(b.text):
                continue
        # đổi về toạ độ của trang, cắt phần tràn sang trang khác
        y1, y2 = max(0, b.y - offset), min(H, b.y + b.h - offset)
        b.y, b.h = y1, max(1, y2 - y1)
        kept.append(b)
    return PageResult(page=page_number, width=W, height=H, bubbles=kept)


def process_document(path: Path, ocr: bool = True, engine: str | None = None,
                     fix: bool = True) -> tuple[int, Iterator[tuple[PageResult, np.ndarray]]]:
    """Trả về (tổng số trang, iterator (kết quả trang, ảnh trang))."""
    total, images = iter_input_images(Path(path))

    def run() -> Iterator[tuple[PageResult, np.ndarray]]:
        prev = cur = None
        number = 0
        heights: list[float] = []
        for nxt in chain(images, [None]):
            if cur is not None:
                number += 1
                yield process_image(cur, number, ocr, engine, fix, prev, nxt, heights), cur
            prev, cur = cur, nxt

    return total, run()


def dialogue_text(pages: list[PageResult]) -> str:
    return "\n".join(b.text for p in pages for b in p.bubbles if b.text)


def draw_debug(image: np.ndarray, page: PageResult) -> np.ndarray:
    out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, b in enumerate(page.bubbles, 1):
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 0, 255), 3)
        cv2.putText(out, str(i), (b.x + 4, b.y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return out
