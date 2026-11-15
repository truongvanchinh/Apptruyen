#!/usr/bin/env python3
"""Tìm bong bóng thoại trong PDF/ảnh truyện tranh và xuất thành các tấm ảnh ghép có đánh số.

    python extract_bubbles.py <file.pdf|ảnh> <thư_mục_ra> [--narrate]

Kết quả trong <thư_mục_ra>:
    sheets/sheet_01.jpg ...  mỗi tấm 8 bong bóng phóng to, ô nào cũng có nhãn "#số tr.trang"
    bubbles.json             danh sách bong bóng theo thứ tự đọc: id, page, sheet
    scenes/scene_01.jpg ...  (chỉ với --narrate) ảnh toàn trang xếp cạnh nhau, đọc từ trái sang phải;
                             mỗi bong bóng được khoanh và đánh cùng số "#số" như trong sheets/
    (chế độ dự phòng) pages/page_001.jpg ...  khi không có bộ dò chữ: ảnh từng trang đã cắt lề

Script KHÔNG đọc chữ. Việc đọc chữ và viết lời dẫn do Claude làm bằng cách nhìn các tấm ảnh.
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


def iter_pages(path: Path):
    """Sinh (số trang, tổng số trang, ảnh trang, [(ảnh bong bóng, khung trong toạ độ trang)]) theo thứ tự đọc.
    Ảnh bong bóng cắt từ cửa sổ đã ghép trang trước/sau, nên bong bóng vắt qua hai trang vẫn nguyên vẹn."""
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
            found = []
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
                box = (b.x, max(0, b.y - offset), b.x + b.w, min(H, b.y + b.h - offset))
                found.append((window[y1:y2, x1:x2].copy(), box))
            yield number, total, cur, found
            print(f"trang {number}/{total}", file=sys.stderr, flush=True)
        prev, cur = cur, nxt


SCENE_COL_W, SCENE_COLS = 440, 4


def scene_column(page: np.ndarray, number: int, marks: list[tuple[int, tuple]]) -> np.ndarray:
    """Ảnh một trang (thu về bề ngang cố định) có khoanh và đánh số các bong bóng."""
    scale = SCENE_COL_W / page.shape[1]
    col = cv2.resize(cv2.cvtColor(page, cv2.COLOR_RGB2BGR), (SCENE_COL_W, max(1, round(page.shape[0] * scale))),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    for bubble_id, (x1, y1, x2, y2) in marks:
        p1, p2 = (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(col, p1, p2, (0, 0, 255), 2)
        label = f"#{bubble_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly = max(th + 4, p1[1])
        cv2.rectangle(col, (p1[0], ly - th - 4), (p1[0] + tw + 4, ly + 2), (0, 0, 255), -1)
        cv2.putText(col, label, (p1[0] + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    header = np.full((26, SCENE_COL_W, 3), 40, np.uint8)
    cv2.putText(header, f"trang {number}", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return np.vstack([header, col])


def write_scene(columns: list[np.ndarray], path: Path) -> None:
    height = max(c.shape[0] for c in columns)
    padded = [cv2.copyMakeBorder(c, 0, height - c.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(40, 40, 40))
              for c in columns]
    cv2.imwrite(str(path), np.hstack(padded), [cv2.IMWRITE_JPEG_QUALITY, 85])


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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    narrate = "--narrate" in sys.argv[1:]
    if len(args) != 2:
        sys.exit(__doc__)
    path, out = Path(args[0]), Path(args[1])
    out.mkdir(parents=True, exist_ok=True)
    try:
        import easyocr  # noqa: F401
    except Exception as exc:
        print(f"Không có EasyOCR ({exc}); chuyển sang chế độ đọc cả trang.", file=sys.stderr)
        pages_fallback(path, out)
        return

    sheets_dir, scenes_dir = out / "sheets", out / "scenes"
    sheets_dir.mkdir(exist_ok=True)
    for old in list(sheets_dir.glob("sheet_*.jpg")) + list(scenes_dir.glob("scene_*.jpg")):
        old.unlink()
    if narrate:
        scenes_dir.mkdir(exist_ok=True)

    index, cells, columns = [], [], []
    sheet_no = scene_no = total_pages = 0
    for page_no, total_pages, page, found in iter_pages(path):
        marks = []
        for crop, box in found:
            bubble_id = len(index) + 1
            if not cells:
                sheet_no += 1
            cells.append(make_cell(crop, f"#{bubble_id} tr.{page_no}"))
            index.append({"id": bubble_id, "page": page_no, "sheet": sheet_no})
            marks.append((bubble_id, box))
            if len(cells) == COLS * ROWS:
                write_sheet(cells, sheets_dir / f"sheet_{sheet_no:02d}.jpg")
                cells = []
        if narrate:
            columns.append(scene_column(page, page_no, marks))
            if len(columns) == SCENE_COLS:
                scene_no += 1
                write_scene(columns, scenes_dir / f"scene_{scene_no:02d}.jpg")
                columns = []
    if cells:
        write_sheet(cells, sheets_dir / f"sheet_{sheet_no:02d}.jpg")
    if columns:
        scene_no += 1
        write_scene(columns, scenes_dir / f"scene_{scene_no:02d}.jpg")
    (out / "bubbles.json").write_text(json.dumps(
        {"mode": "bubbles", "pages": total_pages, "bubbles": index}, ensure_ascii=False, indent=1))
    extra = f", {scene_no} tấm toàn trang -> {scenes_dir}" if narrate else ""
    print(f"MODE bubbles: {len(index)} bong bóng trên {total_pages} trang, {sheet_no} tấm ảnh -> {sheets_dir}{extra}")


if __name__ == "__main__":
    main()
