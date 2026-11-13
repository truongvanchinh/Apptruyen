# Trích Lời Thoại Manga

Web app nhận **PDF/ảnh manga**, tự tìm **bong bóng thoại**, đọc chữ bên trong bằng OCR và xuất file `.txt`
(mỗi bong bóng một dòng, thứ tự trên → dưới, trái → phải). Chữ ngoài bong bóng (hiệu ứng, watermark, lời dẫn) bị bỏ qua.

## Tính năng

- Kéo thả PDF/JPG/PNG/WEBP, xử lý từng trang và hiện kết quả dần dần.
- Xem từng trang, có khung đánh số quanh bong bóng đã nhận diện.
- Sửa trực tiếp nội dung, bỏ tích để loại khỏi file, đổi thứ tự ↑↓, xoá.
- **Khoanh tay** bong bóng bị sót: kéo chuột trên ảnh, chữ được đọc ngay.
- Tự sửa lỗi OCR tiếng Việt dựa trên cấu trúc âm tiết (vd `TRLYỀN → TRUYỀN`, `TPÊN → TRÊN`).
- Tải `.txt` hoặc sao chép toàn bộ.
- Có bản dòng lệnh (`main.py`) để chạy hàng loạt.

## Cài đặt

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

EasyOCR tải model (~100MB) trong lần chạy đầu. Có GPU NVIDIA thì đặt `USE_GPU = True` trong `app/extractor.py`.

Tuỳ chọn Tesseract (nhanh hơn, kém chính xác hơn với font truyện tranh):
Ubuntu `sudo apt install tesseract-ocr tesseract-ocr-vie`, Windows cài từ
[UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) và chọn gói Vietnamese.

## Chạy web

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Mở http://localhost:8000

## Chạy dòng lệnh

```bash
python main.py --pdf input/chapter-01.pdf --output output/chapter-01.txt --debug
```

`--debug` lưu ảnh từng trang có khung bong bóng vào `output/debug/` để kiểm tra nhận diện.

## Cách hoạt động

| Bước | Cách làm | File |
|---|---|---|
| PDF → ảnh | PyMuPDF render 200 DPI | `app/extractor.py` |
| Tìm bong bóng | Vùng gần-trắng ít màu, cắt theo nét viền (Canny) để tách khỏi trời/nền sáng, lọc theo diện tích, độ đặc, tỉ lệ khung và phải có chữ tối bên trong | `detect_bubbles` |
| Thứ tự đọc | Gom bong bóng chồng lấn theo chiều dọc thành hàng; hàng trên → dưới, trong hàng trái → phải | `sort_bubbles` |
| OCR | Tô trắng phần ngoài bong bóng, chuẩn hoá cỡ chữ ~23px, EasyOCR tiếng Việt | `ocr_bubble` |
| Hậu xử lý | Viết hoa đồng nhất, sửa âm tiết không hợp lệ | `clean_text`, `app/vi_fix.py` |

## Giới hạn hiện tại

- Bong bóng không viền, nền tối/màu, hoặc dính liền khung tranh có thể bị sót → dùng khoanh tay.
- OCR font truyện tranh vẫn sai dấu thanh ở vài chữ (vd `TỨ → TÚ`), cần soát lại.
- Hướng nâng cấp: huấn luyện YOLO nhận diện bong bóng; dùng model OCR fine-tune cho font comic tiếng Việt (VietOCR).
