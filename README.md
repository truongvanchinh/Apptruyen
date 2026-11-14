# Trích Lời Thoại Manga

Web app nhận **PDF/ảnh manga, webtoon**, tự tìm **bong bóng thoại**, đọc chữ bên trong bằng OCR và xuất file `.txt`
(mỗi bong bóng một dòng, thứ tự trên → dưới, trái → phải). Chữ ngoài bong bóng (hiệu ứng âm thanh, watermark,
chữ đè lên tranh) bị bỏ qua.

## Tính năng

- Kéo thả PDF/JPG/PNG/WEBP, xử lý từng trang và hiện kết quả dần dần.
- Nhận được nhiều kiểu bong bóng: viền đen, viền chấm, hình gai, hình mây, chữ nghiêng, bong bóng chạm mép trang.
- Webtoon bị cắt khúc thành nhiều trang: bong bóng nằm vắt qua hai trang vẫn được đọc trọn vẹn.
- Tự cắt dải đen hai bên (PDF tạo từ ảnh chụp màn hình), tự loại watermark và chữ hiệu ứng.
- Xem từng trang có khung đánh số; sửa nội dung, bỏ tích để loại, đổi thứ tự ↑↓, xoá.
- **Khoanh tay** bong bóng bị sót: kéo chuột trên ảnh, chữ được đọc ngay.
- Tải `.txt` hoặc sao chép toàn bộ. Có bản dòng lệnh (`main.py`).

## Cài đặt

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

EasyOCR tải model (~100MB) trong lần chạy đầu. Có GPU NVIDIA thì đặt `USE_GPU = True` trong `app/extractor.py`.

**Nên cài thêm Tesseract + gói tiếng Việt** – app sẽ tự dùng chế độ "kết hợp" cho dấu tiếng Việt chính xác hơn rõ rệt:

- Ubuntu/Debian: `sudo apt install tesseract-ocr tesseract-ocr-vie`
- macOS: `brew install tesseract tesseract-lang`
- Windows: cài từ [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki), chọn gói *Vietnamese*,
  thêm thư mục cài đặt vào `PATH`.

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

Hướng tiếp cận là **tìm chữ trước, xét bong bóng sau** (`app/extractor.py`):

| Bước | Cách làm |
|---|---|
| PDF → ảnh | PyMuPDF, lấy đúng độ phân giải gốc của ảnh nhúng; cắt dải đen hai bên |
| Ngữ cảnh | Ghép 40% trang trước/sau vào mỗi trang để bong bóng bị cắt ở mép vẫn nguyên vẹn |
| Dò chữ | Bộ dò CRAFT của EasyOCR cho các dòng chữ |
| Lọc dòng | Giữ dòng nằm trên nền phẳng sáng/tối, mực đen/trắng đậm → loại watermark, chữ hiệu ứng, chữ đè lên tranh |
| Gom khối | Dòng gần nhau, cùng cỡ chữ và cùng độ nghiêng thành một khối |
| Xét bong bóng | Bắn 48 tia từ tâm khối chữ: ≥80% tia phải chạm một đường viền sắc nét ở gần → khối nằm trong vùng kín = bong bóng |
| Thứ tự đọc | Hàng trên → dưới, trong hàng trái → phải |
| OCR | Chỉ giữ phần chữ, xoay thẳng chữ nghiêng, chuẩn hoá cỡ chữ; EasyOCR đọc khung chữ, Tesseract sửa dấu |
| Hậu xử lý | Sửa âm tiết không hợp lệ theo cấu trúc tiếng Việt; sửa dấu theo bảng tần suất từ của `wordfreq` — chỉ đổi một dấu, chỉ khi biến thể phổ biến hơn áp đảo, bỏ qua chữ hai bộ OCR đọc giống nhau (`app/vi_fix.py`) |

Chẩn đoán khi một bong bóng bị sót: đặt `extractor.TRACE = []` trước khi xử lý, sau đó xem lý do từng khối chữ bị loại.

## Giới hạn hiện tại

- OCR vẫn sai dấu ở một số chữ, nên cần soát lại trước khi dùng. Bước sửa theo tần suất không hiểu ngữ cảnh,
  nên đôi khi đổi một chữ hiếm nhưng đúng thành chữ phổ biến hơn; bỏ tích "Tự sửa lỗi chính tả" nếu không muốn.
- Chữ kiểu viết tay/nét cọ lớn (tiếng hét) đọc kém; bong bóng chỉ có một ký tự lớn có thể bị sót → dùng khoanh tay.
- Bong bóng nền màu lưng chừng (không sáng, không tối) bị bỏ qua.
- Tốc độ trên CPU khoảng 4–5 giây/trang.
- Đã thử VietOCR: kém hơn cách ghép hiện tại trên font truyện tranh (0,86 so với 0,98 trên 7 bong bóng mẫu) nên không dùng.
- Hướng nâng cấp: fine-tune model đọc chữ cho font truyện tranh; thêm ngữ cảnh (cặp từ) cho bước sửa dấu.
