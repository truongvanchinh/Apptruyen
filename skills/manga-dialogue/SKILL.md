---
name: manga-dialogue
description: Trích lời thoại trong bong bóng thoại của truyện tranh (manga, manhua, manhwa, webtoon) từ file PDF hoặc ảnh ra file .txt, mỗi bong bóng một dòng theo thứ tự đọc; có thêm chế độ "có lời dẫn" viết xen các dòng mô tả cảnh ngắn dựa vào tranh để thành kịch bản đọc liền mạch (dùng cho review, lồng tiếng, tóm tắt truyện). Dùng skill này bất cứ khi nào người dùng gửi PDF/ảnh truyện tranh và muốn lấy chữ, lời thoại, thoại nhân vật, script, kịch bản, transcript, "đọc chữ trong bong bóng", "xuất thoại ra txt", "extract dialogue / speech bubbles", "viết kịch bản review truyện", "thoại kèm mô tả", kể cả khi họ chỉ nói "lấy chữ trong truyện này" hay "chép lời thoại chương này". Skill tự tìm và cắt bong bóng bằng script rồi Claude đọc chữ từ ảnh, nên chính xác hơn OCR thường và tốn ít hạn mức hơn đọc cả trang.
---

# Trích lời thoại truyện tranh

Mục tiêu: từ một file PDF/ảnh truyện tranh, tạo file `.txt` chứa lời thoại trong bong bóng,
mỗi bong bóng một dòng, theo thứ tự đọc (trên xuống dưới, cùng hàng thì trái sang phải).

Cách làm chia hai phần, mỗi phần giao cho bên làm tốt nhất:

1. **Script tìm và cắt bong bóng** (`scripts/extract_bubbles.py`). Nó xử lý những việc máy móc mà
   mắt thường dễ sót khi lướt cả chương: cắt dải đen/lề của ảnh chụp màn hình, ghép các trang bị
   cắt khúc để bong bóng vắt qua hai trang vẫn nguyên vẹn, loại watermark, chữ hiệu ứng và chữ
   nét cọ cỡ lớn, sắp đúng thứ tự đọc.
2. **Claude đọc chữ** từ các tấm ảnh bong bóng. OCR thường hay sai dấu tiếng Việt với font truyện
   tranh (RỒI thành RỖI, NGƯỜI thành NGUỒI); đọc bằng mắt thì không bị.

## Bước 1: Chạy script

```bash
pip install pymupdf opencv-python-headless numpy easyocr   # nếu chưa có
python <thư mục skill>/scripts/extract_bubbles.py "<file.pdf>" <thư_mục_làm_việc>
# thêm --narrate nếu người dùng muốn kịch bản có lời dẫn (xem "Chế độ có lời dẫn")
```

Khoảng 2–5 giây mỗi trang trên CPU; với chương dài hãy chạy nền và chờ. Lần đầu EasyOCR tải model
dò chữ (~80MB).

Dòng cuối script in ra cho biết chế độ:

- `MODE bubbles: N bong bóng ... K tấm ảnh` — trường hợp bình thường, sang Bước 2.
- `MODE pages: ...` — không cài/tải được EasyOCR. Script đã xuất ảnh từng trang (đã cắt lề) vào
  `pages/`. Sang phần "Chế độ dự phòng" ở cuối.

## Bước 2: Đọc các tấm ảnh

Mở lần lượt `sheets/sheet_01.jpg`, `sheet_02.jpg`, ... bằng công cụ xem ảnh. Mỗi tấm có tối đa 8 ô;
mỗi ô là một bong bóng, góc trên ghi nhãn `#số tr.trang`. Số tăng dần đúng theo thứ tự đọc.

Với mỗi ô, chép lại chữ trong bong bóng thành **một dòng**:

- Chép đúng như in, kể cả dấu câu (`!`, `?`, `...`, `~`). Nối các dòng trong bong bóng bằng một
  dấu cách. Giữ nguyên kiểu chữ hoa/thường của truyện.
- Không dịch, không sửa câu chữ, không thêm tên nhân vật hay mô tả. Người dùng cần đúng nguyên văn
  để dùng tiếp (lồng tiếng, dịch, đối chiếu).
- Nếu truyện in sai chính tả thì vẫn chép như in; đó là nội dung gốc, không phải lỗi đọc.
- Ô chú thích hình chữ nhật (tên nhân vật, "SÁNG SỚM", lời dẫn trong khung) được tính là lời thoại:
  chép bình thường.
- **Bỏ qua** ô không phải lời thoại mà script cắt nhầm: nét vẽ hiệu ứng, chữ tượng thanh vẽ tay,
  mẩu watermark/tên trang web, ô không có chữ. Script cố tình cắt rộng tay hơn là sót, nên việc
  lọc lần cuối là của bạn.
- Trong ô có thể lẫn mẩu watermark mờ đè lên bong bóng, hoặc hình vẽ nhỏ: chỉ chép lời thoại.
- Chữ nào thật sự không đọc được thì ghi `[?]` tại chỗ đó thay vì đoán. Một chỗ đánh dấu rõ ràng
  có ích hơn một chữ đoán sai mà người dùng không biết để soát.

Đọc hết tấm này mới sang tấm khác và ghi kết quả dần vào file, để chương dài không bị dồn quá nhiều
ảnh trong một lượt.

## Bước 3: Ghi file và báo cáo

Ghi file `<tên file gốc>.txt` (UTF-8) vào nơi người dùng lấy được, mỗi bong bóng một dòng, đúng
thứ tự số, không kèm số thứ tự hay số trang trừ khi người dùng yêu cầu.

Báo lại ngắn gọn: số lời thoại đã chép, số ô đã bỏ qua và vì sao, các chỗ `[?]` kèm số trang để
người dùng soát. Đừng dán toàn bộ nội dung chương vào khung chat; đưa file là đủ, chỉ trích vài
dòng nếu cần minh hoạ.

## Chế độ có lời dẫn

Dùng khi người dùng muốn kịch bản đọc liền mạch chứ không chỉ lời thoại: "có mô tả", "kèm lời dẫn",
"để khỏi cụt", "kịch bản review", hoặc họ đưa một file mẫu có các dòng tả cảnh xen giữa lời thoại.
Nếu không rõ họ muốn loại nào, hỏi một câu trước khi chạy, vì chế độ này tốn hạn mức hơn (phải nhìn
cả trang, không chỉ bong bóng).

Chạy script với `--narrate`. Ngoài `sheets/`, script xuất thêm `scenes/scene_01.jpg`, ...: mỗi tấm
là 4 trang xếp cạnh nhau, đọc từ trái sang phải, mỗi trang từ trên xuống. Bong bóng được khoanh đỏ
và đánh cùng số `#n` như trong `sheets/`.

Cách làm:

1. Đọc lần lượt các tấm trong `scenes/`. Đây là nguồn chính: vừa thấy cảnh, vừa đọc được thoại,
   kể cả bong bóng script không khoanh (script có thể sót; tranh thì không).
2. Chỉ mở tấm tương ứng trong `sheets/` khi chữ trong `scenes/` bị nhỏ, bị nhãn số che, hoặc bạn
   không chắc một dấu. Không cần mở hết.
3. Viết file theo dòng thời gian của truyện, xen hai loại dòng:
   - **Lời thoại**: nguyên văn trong bong bóng và ô chú thích, mỗi bong bóng một dòng, viết hoa đầu
     câu và tên riêng (không viết hoa toàn bộ), giữ dấu câu như in.
   - **Lời dẫn**: câu ngắn do bạn viết từ tranh, viết thường cả dòng, kết bằng dấu phẩy. Nói ai
     làm gì, ở đâu, biểu cảm ra sao, đủ để người nghe không nhìn tranh vẫn theo được.

Viết lời dẫn thế nào cho tốt:

- Một dòng cho mỗi nhịp truyện, không phải mỗi khung tranh. Cảnh hành động hay chuyển cảnh thì tả
  kỹ hơn; đoạn hai người đối đáp liên tục thì chỉ cần một dòng mở đầu kiểu "cô quay lại đáp,".
- Trước một lời thoại đổi người nói, thêm dòng dẫn ngắn cho biết ai nói ("tên cầm đầu gạt đi,").
  Thiếu dòng này là chỗ người nghe dễ bị lạc nhất.
- Tiếng động và chữ hiệu ứng trong tranh (ÀO, VÈO, KHỤ KHỤ) đưa vào lời dẫn thay vì chép thành
  dòng riêng: "vèo, một mũi tên lao tới,".
- Chỉ tả điều thấy được trong tranh. Đừng đoán tên, quan hệ hay động cơ mà truyện chưa nói; khi
  chưa biết tên thì gọi theo đặc điểm ("chàng trai tóc xanh", "tên cầm đầu") cho tới khi có người
  gọi tên.
- Giọng văn theo người dùng. Mặc định trung tính. Nếu họ đưa file mẫu hoặc dặn riêng (ví dụ gọi
  nữ chính là "nữ 9", thêm vài câu bình luận vui), làm theo đúng kiểu đó nhưng tiết chế: vài câu
  cả chương là đủ, nhiều quá sẽ lấn át truyện.
- Bỏ watermark, lời kêu gọi like/share, trang giới thiệu nhóm dịch và phần giao diện trang web lọt
  vào ảnh.

Ví dụ về nhịp điệu (nội dung minh hoạ):

```
trong rừng đêm, một cô gái bị dồn tới mép vực,
Các người là ai?
tên cầm đầu cười khẩy,
Chỉ là người được thuê thôi.
cô cắn môi, rồi ngả người ra sau,
```

Ghi file `<tên file gốc>.txt` và báo cáo như Bước 3, thêm một ý: lời dẫn là cách hiểu của bạn từ
tranh, người dùng nên soát những chỗ quan trọng về cốt truyện.

## Lưu ý về giới hạn

- Script chỉ lấy bong bóng nền sáng hoặc tối hẳn có viền rõ. Bong bóng nền màu lưng chừng, hoặc
  bong bóng chỉ có một ký tự rất to, có thể bị sót. Nếu người dùng cần đủ tuyệt đối, nói rõ điều
  này và đề nghị họ soát vài trang.
- Chữ nét cọ / tiếng hét cỡ lớn bị bỏ có chủ ý. Nếu người dùng muốn lấy cả loại này, đặt
  `SKIP_BIG_TEXT = False` trong `scripts/manga_bubbles/extractor.py` rồi chạy lại.
- Nếu số bong bóng trông ít bất thường so với số trang (ví dụ dưới 1 bong bóng mỗi 2 trang với
  truyện nhiều thoại), xem thử vài ảnh trang để kiểm tra trước khi tin kết quả.

## Chế độ dự phòng (MODE pages)

Khi không có bộ dò chữ, đọc trực tiếp các ảnh trong `pages/` theo thứ tự `page_001.jpg`, ... và
tự làm phần việc của script:

- Chỉ chép chữ nằm trong bong bóng thoại và ô chú thích; bỏ watermark, chữ hiệu ứng, chữ nét cọ.
- Thứ tự: trên xuống dưới, cùng hàng thì trái sang phải.
- Các trang thường là một dải dọc bị cắt khúc: bong bóng bị cắt ở đáy trang này tiếp tục ở đầu
  trang sau. Ghép hai nửa thành một dòng, đừng chép thành hai lời thoại.

Chế độ này tốn hạn mức hơn nhiều (đọc cả trang thay vì chỉ bong bóng), nên báo cho người dùng biết
và chỉ dùng khi không cài được EasyOCR.
