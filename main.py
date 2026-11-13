"""Chạy không cần web: python main.py --pdf input/manga.pdf --output output/dialogue.txt"""

import argparse
from pathlib import Path

import cv2

from app import extractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Trích lời thoại trong bong bóng thoại manga")
    parser.add_argument("--pdf", default="input/manga.pdf", help="File PDF hoặc ảnh")
    parser.add_argument("--output", default="output/dialogue.txt")
    parser.add_argument("--engine", choices=["easyocr", "tesseract"], default=None)
    parser.add_argument("--no-fix", action="store_true", help="Tắt tự sửa lỗi chính tả tiếng Việt")
    parser.add_argument("--debug", action="store_true", help="Lưu ảnh có khung bong bóng vào output/debug/")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    debug_dir = output.parent / "debug"
    if args.debug:
        debug_dir.mkdir(exist_ok=True)

    total, images = extractor.iter_input_images(Path(args.pdf))
    pages = []
    for number, image in enumerate(images, 1):
        page = extractor.process_image(image, number, engine=args.engine, fix=not args.no_fix)
        pages.append(page)
        print(f"Trang {number}/{total}: {len(page.bubbles)} bong bóng")
        for i, b in enumerate(page.bubbles, 1):
            print(f"  {i}. {b.text}")
        if args.debug:
            cv2.imwrite(str(debug_dir / f"page_{number}.jpg"), extractor.draw_debug(image, page))

    output.write_text(extractor.dialogue_text(pages), encoding="utf-8")
    print(f"\nĐã lưu: {output.resolve()}")


if __name__ == "__main__":
    main()
