#!/usr/bin/env python3
"""Tìm bong bóng thoại trong PDF/ảnh truyện tranh và xuất thành các tấm ảnh ghép có đánh số.

    python extract_bubbles.py <file.pdf|ảnh> <thư_mục_ra>

Kết quả trong <thư_mục_ra>:
    sheets/sheet_01.jpg ...  mỗi tấm 8 bong bóng, ô nào cũng có nhãn "#số tr.trang"
    bubbles.json             danh sách bong bóng theo thứ tự đọc: id, page, sheet
    (chế độ dự phòng) pages/page_001.jpg ...  khi không có bộ dò chữ: ảnh từng trang đã cắt lề

Script KHÔNG đọc chữ. Việc đọc chữ do Claude làm bằng cách nhìn các tấm ảnh.
"""

from __future__ import annotations

import json
import sys
from itertools import chain
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manga_bubbles import extractor as E  # noqa: E402

COLS, ROWS = 4, 2
CELL_W, CELL_H, LABEL_H = 450, 360, 26


def iter_bubble_crops(path: Path):
    """Sinh (số trang, ảnh bong bóng) theo thứ tự đọc. Ảnh cắt từ cửa sổ đã ghép trang trước/sau,
    nên bong bóng vắt qua hai trang vẫn nguyên vẹn."""
    total, images = E.iter_input_images(path)
    prev = cur = None
    number = 0
    heights: list[float] = []
    for nxt in chain(images, [None]):
        if cur is not None:
            number += 1
            H, W = cur.shape[:2]
            parts, offset = [], 0
            if prev is not None:
                top = E._fit_width(prev, W)
                top = top[-int(top.shape[0] * E.CONTEXT_RATIO):]
                parts.append(top)
                offset = top.shape[0]
            parts.append(cur)
            if nxt is not None:
                bottom = E._fit_width(nxt, W)
                parts.append(bottom[:int(bottom.shape[0] * E.CONTEXT_RATIO)])
            window = np.vstack(parts) if len(parts) > 1 else cur

            mine = []
            for b in E.detect_bubbles(window):
                _, ty1, _, ty2 = E._bbox(b.lines)
                if offset <= (ty1 + ty2) / 2 < offset + H:
                    mine.append(b)
            for b in E.sort_bubbles(mine):
                line_h = float(np.median([l.h for l in b.lines]))
                typical = float(np.median(heights)) if len(heights) >= 5 else None
                if E.SKIP_BIG_TEXT and (line_h > E.BIG_TEXT_WIDTH_RATIO * W
                                        or (typical and line_h > E.BIG_TEXT_RATIO * typical)):
                    continue  # chữ nét cọ / tiếng hét cỡ lớn: bỏ
                heights.append(line_h)
                pad = 6
                y1, y2 = max(0, b.y - pad), min(window.shape[0], b.y + b.h + pad)
                x1, x2 = max(0, b.x - pad), min(W, b.x + b.w + pad)
                yield number, total, window[y1:y2, x1:x2].copy()
            print(f"trang {number}/{total}", file=sys.stderr, flush=True)
        prev, cur = cur, nxt


def make_cell(crop: np.ndarray, label: str) -> np.ndarray:
    cell = np.full((CELL_H, CELL_W, 3), 90, np.uint8)
    h, w = crop.shape[:2]
    scale = min((CELL_W - 10) / w, (CELL_H - LABEL_H - 8) / h, 3.0)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale, interpolation=interp)
    cell[LABEL_H:LABEL_H + small.shape[0], 5:5 + small.shape[1]] = small
    cv2.putText(cell, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return cell


def write_sheet(cells: list[np.ndarray], path: Path) -> None:
    blank = np.full((CELL_H, CELL_W, 3), 90, np.uint8)
    cells = cells + [blank] * (COLS * ROWS - len(cells))
    rows = [np.hstack(cells[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
    used = [r for i, r in enumerate(rows) if i == 0 or len(cells) and any(c is not blank for c in cells[i * COLS:(i + 1) * COLS])]
    cv2.imwrite(str(path), np.vstack(used), [cv2.IMWRITE_JPEG_QUALITY, 88])


def pages_fallback(path: Path, out: Path) -> None:
    """Không có bộ dò chữ: xuất ảnh từng trang đã cắt dải đen/lề để Claude đọc trực tiếp."""
    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    total, images = E.iter_input_images(path)
    for n, image in enumerate(images, 1):
        cv2.imwrite(str(pages_dir / f"page_{n:03d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "bubbles.json").write_text(json.dumps({"mode": "pages", "pages": total}, ensure_ascii=False))
    print(f"MODE pages: {total} trang -> {pages_dir}")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    path, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    try:
        import easyocr  # noqa: F401
    except Exception as exc:
        print(f"Không có EasyOCR ({exc}); chuyển sang chế độ đọc cả trang.", file=sys.stderr)
        pages_fallback(path, out)
        return

    sheets_dir = out / "sheets"
    sheets_dir.mkdir(exist_ok=True)
    for old in sheets_dir.glob("sheet_*.jpg"):
        old.unlink()
    index, cells, sheet_no, total_pages = [], [], 0, 0
    for i, (page, total_pages, crop) in enumerate(iter_bubble_crops(path), 1):
        if not cells:
            sheet_no += 1
        cells.append(make_cell(crop, f"#{i} tr.{page}"))
        index.append({"id": i, "page": page, "sheet": sheet_no})
        if len(cells) == COLS * ROWS:
            write_sheet(cells, sheets_dir / f"sheet_{sheet_no:02d}.jpg")
            cells = []
    if cells:
        write_sheet(cells, sheets_dir / f"sheet_{sheet_no:02d}.jpg")
    (out / "bubbles.json").write_text(json.dumps(
        {"mode": "bubbles", "pages": total_pages, "bubbles": index}, ensure_ascii=False, indent=1))
    print(f"MODE bubbles: {len(index)} bong bóng trên {total_pages} trang, {sheet_no} tấm ảnh -> {sheets_dir}")


if __name__ == "__main__":
    main()
