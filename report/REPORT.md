# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (CUDA)

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

## 1. Phân loại ảnh

Sử dụng `sample_id = traffic` từ `day1_lab_outputs/classification_predictions.json` và ảnh `visuals/classification_top5.png`.

1. **Record hạng 1:** `class_id = 468`; `class_name = cab`; `rank = 1`; `score = 0.51087`; `taxonomy_name = ImageNet-1K` (đối chiếu `classification_predictions.json`, `sample_id = traffic`).

2. **Đơn vị nhãn:** Record phân loại **cả bức ảnh**, không phải một vật thể riêng lẻ. Mô hình xếp hạng các lớp cho toàn ảnh và không trả về vị trí hay bounding box của từng phương tiện.

3. **Nguồn danh sách lớp:** Danh sách lớp được xác định bởi taxonomy `ImageNet-1K` đi kèm checkpoint phân loại. Mô hình không tự tạo ra tên lớp mới trong lúc dự đoán.

4. **Vì sao cần cả ba trường:** `class_id` là mã định danh dùng trong xử lý dữ liệu; `class_name` giúp con người đọc và kiểm tra nhãn; `taxonomy_name` xác định hệ thống lớp chứa mã và tên đó. Nếu thiếu taxonomy, cùng một mã lớp có thể bị diễn giải nhầm giữa các bộ dữ liệu.

5. **Quy tắc chọn một lớp:** Guideline cần nêu tiêu chí xác định chủ thể đại diện cho ảnh có nhiều đối tượng, ví dụ dựa trên mục đích bộ dữ liệu và mức độ nổi bật của đối tượng. Những ảnh có nhiều chủ thể tương đương phải có quy tắc ưu tiên rõ ràng; nếu vẫn mơ hồ, annotator chuyển ca cho người phụ trách để thống nhất.

6. **Vì sao score không phải ground truth:** `score` chỉ là mức độ tin cậy do mô hình tính cho prediction, không phải xác nhận nhãn đúng. Ảnh `traffic` có nhiều xe buýt và các loại phương tiện, trong khi top-1 của model là `cab`; vì vậy cần đối chiếu nội dung ảnh và guideline thay vì mặc nhiên tin dự đoán cao nhất.

## 2. Phát hiện vật thể

Sử dụng `sample_id = kitchen` từ `day1_lab_outputs/detection_predictions.json` và ảnh `visuals/detection_predictions.png`.

1. **Một record thực tế:** `class_name = person`; `score = 0.912647`; `bbox_xyxy = [385.33, 69.24, 498.92, 348.92]`; `bbox_width = 113.59`; `bbox_height = 279.68` (đối chiếu `detection_predictions.json`, `sample_id = kitchen`).

2. **Giải thích bounding box:** Với định dạng `xyxy`, cặp `(x_min, y_min)` là góc trên-trái và `(x_max, y_max)` là góc dưới-phải, tính bằng pixel từ gốc `(0, 0)` ở góc trên-trái ảnh. Hộp của người đứng nấu ăn kéo từ góc trên-trái `(385.33, 69.24)` đến góc dưới-phải `(498.92, 348.92)`. Chiều rộng là `498.92 - 385.33 = 113.59` pixel, chiều cao là `348.92 - 69.24 = 279.68` pixel.

3. **So sánh ngưỡng:** Chọn ngưỡng `0.35` và `0.60`. JSON được xuất ở ngưỡng `0.35` có **11 prediction** riêng cho ảnh `kitchen`. Trong 11 record này có **6 prediction với score ≥ 0.60**; do đó khi lọc cùng bộ kết quả theo ngưỡng `0.60` còn 6, giảm 5 so với ngưỡng `0.35`. Số 6 được **suy ra bằng cách lọc score của JSON ở ngưỡng 0.35**, không phải số đo của một lượt inference độc lập ở ngưỡng 0.60. Ngưỡng chính thức dùng tạo JSON vẫn là `0.35`.

4. **Việc reviewer làm thêm khi hạ ngưỡng:** Reviewer phải xem thêm những hộp có score thấp để phát hiện false positive, lớp bị đoán sai, các hộp trùng nhau hoặc hộp lệch. Hạ ngưỡng có thể giúp phát hiện thêm vật thể nhưng tăng khối lượng kiểm tra; model score không tự quyết định nhãn ground truth.

5. **Quy tắc vẽ box đề xuất:** Với mỗi vật thể thuộc taxonomy và đủ điều kiện gán nhãn theo guideline, annotator vẽ một bounding box vừa khít vùng vật thể nhìn thấy, tránh lấy quá nhiều nền và dùng cùng một quy ước cho vật thể bị che khuất/cắt mép. Không tự suy đoán phần không nhìn thấy nếu guideline không quy định.

6. **Ca bị che/cắt mép:** Trong ảnh `kitchen` gốc có một phần cánh tay/bàn tay của người xuất hiện ở mép trái, trong khi phần lớn cơ thể nằm ngoài khung hình. Đây là ca cần quy tắc về mức độ hiển thị tối thiểu và cách xử lý vật thể cắt mép. Nếu guideline chưa xác định rõ có gán nhãn toàn bộ người trong tình huống này hay không, annotator nên **escalate** cho Lab Coach, tránh tự áp dụng một cách khác nhau giữa các ảnh.

## 3. Phân đoạn theo từng đối tượng

Sử dụng `sample_id = kitchen` từ `day1_lab_outputs/segmentation_predictions.json` và ảnh `visuals/segmentation_prediction.png`.

1. **Một record thực tế:** `instance_id = kitchen-001`; `class_name = person`; `score = 0.899166`; `polygon_point_count = 350`. Năm điểm đầu của `polygon_xy` là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]` (đối chiếu `segmentation_predictions.json`, `sample_id = kitchen`).

2. **Thông tin polygon bổ sung:** Bounding box chỉ mô tả vùng hình chữ nhật bao quanh vật thể. Polygon chứa các điểm theo biên, nhờ vậy mô tả hình dáng vật thể và phân biệt vùng thuộc vật thể với phần nền nằm trong box chính xác hơn.

3. **Ý nghĩa instance_id:** `instance_id` phân biệt các cá thể khác nhau, kể cả khi chúng có cùng `class_name` (ví dụ nhiều chiếc bát). Nó không phải `class_id` vì không cho biết loại vật thể; cũng không phải tracking ID vì bài lab xử lý ảnh tĩnh, không theo dõi cùng một vật qua nhiều khung hình.

4. **Quy tắc vẽ biên mask đề xuất:** Vẽ biên mask bám sát phần nhìn thấy của từng vật thể, không ăn sang nền hoặc vật thể bên cạnh; các cá thể cùng lớp vẫn phải có mask riêng. Nếu biên bị che hoặc không phân biệt được, áp dụng thống nhất guideline; không tự suy đoán đường biên khuất.

5. **Ca khó về biên:** Ở khu vực các bát và đồ ăn trên bàn phía trước ảnh `kitchen`, nhiều vật thể tiếp xúc hoặc che khuất nhau khiến ranh giới khó tách. Nếu guideline có quy tắc chỉ vẽ phần nhìn thấy, annotator áp dụng quy tắc đó cho từng instance; nếu không thể xác định ranh giới hoặc quy tắc chưa rõ, cần **escalate** cho Lab Coach. Reviewer kiểm tra mask có chồng lấn, bỏ sót hoặc ăn sang vật bên cạnh không.

## 4. Vòng đời và kiểm tra chất lượng

Trong production: `ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`. Bài lab này chỉ quan sát prediction, không tạo ground truth chính thức và không huấn luyện mô hình.

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp đại diện cho **toàn ảnh**, theo taxonomy và guideline. | `traffic`: top-1 là `cab` (score `0.51087`) dù ảnh có nhiều phương tiện, gồm xe buýt; khó chọn một lớp duy nhất cho cả cảnh. | Xác định lớp đại diện theo tiêu chí guideline, không chép máy móc dự đoán top-1; chuyển ca mơ hồ. | Kiểm tra lớp được chọn có phù hợp nội dung ảnh, taxonomy và quy tắc ưu tiên không. |
| Phát hiện vật thể | Mỗi vật thể hợp lệ có `class_id`/`class_name` và một `bbox_xyxy`. | `kitchen`: một phần người xuất hiện ở mép trái, bị cắt khỏi khung hình; cần quy tắc về độ hiển thị và box. | Vẽ box theo quy tắc thống nhất; hỏi Lab Coach nếu chưa rõ có gán nhãn đối tượng cắt mép hay không. | Kiểm tra thiếu/trùng hộp, sai lớp, box lệch hoặc bao quá nhiều nền, và tính nhất quán ở mép ảnh. |
| Instance segmentation | Mỗi cá thể có lớp và một polygon/mask riêng, kể cả nhiều vật cùng lớp. | `kitchen`: các bát/đồ vật phía trước tiếp xúc hoặc che lấp nhau, gây khó phân biệt đường biên. | Vẽ mask cho từng instance theo phần nhìn thấy và guideline; chuyển ca có biên mơ hồ. | Kiểm tra biên mask có sát vật thể, ăn sang nền/vật khác hoặc gộp nhầm các instance không. |

## 5. An toàn dữ liệu

1. **Quy tắc bảo vệ dữ liệu:** Chỉ dùng ba ảnh COCO công khai trong phạm vi bài lab; không tải ảnh cá nhân, thông tin định danh, dữ liệu khách hàng hoặc dữ liệu nội bộ lên Colab hay GitHub công khai.

2. **Khi gặp dữ liệu không đúng phạm vi:** Dừng xử lý, không tự tải lên hoặc chia sẻ dữ liệu và báo ngay cho Lab Coach/người phụ trách để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

- [x] Đã điền đủ các câu trả lời và đối chiếu record Classification, Detection, Segmentation với ba tệp JSON.
- [ ] `report/REPORT.md` (file đã điền đủ).
- [ ] `report/day1_lab_outputs/classification_predictions.json`.
- [ ] `report/day1_lab_outputs/detection_predictions.json`.
- [ ] `report/day1_lab_outputs/segmentation_predictions.json`.
- [ ] `report/day1_lab_outputs/IMAGE_ATTRIBUTION.md`.
- [ ] `report/day1_lab_outputs/visuals/classification_top5.png`.
- [ ] `report/day1_lab_outputs/visuals/detection_predictions.png`.
- [ ] `report/day1_lab_outputs/visuals/segmentation_prediction.png`.
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
- [ ] Đã commit đúng cấu trúc trong repository GitHub.
- [ ] Đã nộp link repository trên LMS.
