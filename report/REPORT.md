# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.11 / PyTorch 2.x / Ultralytics 8.4.14

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.51 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Trả lời: Record này mô tả cho cả bức ảnh. Trong file JSON kết quả không hề có tọa độ vùng chứa (bounding box) hay đa giác nào cả, vì bài toán phân loại ảnh chỉ đưa ra 1 nhãn đại diện cho toàn bộ tấm hình.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Trả lời: Đi kèm cố định với taxonomy (ImageNet-1K) từ lúc huấn luyện checkpoint `yolo11n-cls.pt`, model chỉ tính xác suất rồi ghép vào danh sách có sẵn chứ không tự sinh ra nhãn mới.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   Trả lời: Vì mỗi bộ dữ liệu (taxonomy) có hệ thống nhãn và cách đánh số ID khác nhau. Cùng một tên tiếng Anh nhưng ở bộ dữ liệu này có thể mang ID khác hoặc định nghĩa khác bộ kia, nên phải lưu kèm cả 3 trường để khi đối chiếu dữ liệu không bị nhầm.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Trả lời: Guideline cần quy định rõ thứ tự ưu tiên khi chọn nhãn: ví dụ chọn vật thể có diện tích lớn nhất, nằm ở chính giữa ảnh, hoặc chọn lớp phản ánh đúng bối cảnh chính của bức ảnh.
6. Vì sao không được lấy `score` của model làm ground truth?
   Trả lời: Vì `score` chỉ là mức độ tự tin của model khi dự đoán, không đảm bảo kết quả đó đúng với thực tế. Chẳng hạn như ở ảnh `traffic`, cảnh thực tế là đường phố có nhiều xe buýt và ô tô con nhưng model vẫn đoán là `cab` (xe taxi) với score cao nhất.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=bowl ; `score`=0.72 ; `bbox_xyxy`=[252.08, 300.32, 320.41, 368.65] ; `bbox_width`=68.33 ; `bbox_height`=68.33
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   Trả lời: Tính từ gốc tọa độ (0,0) ở góc trên bên trái bức ảnh, hộp này bắt đầu từ điểm trên-trái tại (x=252.08, y=300.32) và kéo tới điểm dưới-phải tại (x=320.41, y=368.65). Chiều rộng và chiều cao của hộp đều là 68.33 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   Trả lời: Ở ngưỡng `0.20` model giữ lại 17 prediction. Khi tăng ngưỡng lên `0.60` thì chỉ còn lại 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Trả lời: Reviewer sẽ tốn công rà soát hơn vì xuất hiện nhiều hộp nhiễu (nhận diện nhầm vật thể) và các hộp bị vẽ chồng đè lên nhau trên cùng một vật, phải lọc và xóa bớt các hộp sai đi.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Trả lời: Khung box phải kéo sát vào 4 cạnh ngoài cùng nhìn thấy được của vật thể, không được vẽ cắt lẹm vào phần thịt của vật và cũng không để dư khoảng trống nền xung quanh quá 2 pixel.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Trả lời: Trên ảnh có người phụ nữ đứng ở góc phải bị bàn bếp che nửa dưới và khuất một phần ở mép ảnh, hoặc mấy cái bát xếp chồng lên nhau ở kệ dưới. Với trường hợp phổ biến như vật bị che một phần hoặc chạm mép ảnh thì nên đưa thẳng vào guideline (ví dụ: chỉ vẽ box bao quanh phần nhìn thấy được nếu vật lộ ra trên 30%). Còn nếu vật bị che mờ hoàn toàn khó đoán là bát hay đĩa thì mới cần escalation hỏi Lab Coach để tránh tự đoán mò.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.90 ; số điểm polygon=`polygon_point_count`=42 ; 3–5 điểm đầu của `polygon_xy`=[[120.5, 45.2], [122.0, 48.5], [125.1, 50.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Trả lời: Polygon nối các điểm bám sát theo đường viền thực tế của vật thể, giúp tách hẳn vật thể ra khỏi phần nền (background) thừa ở 4 góc mà hình hộp chữ nhật không làm được.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   Trả lời: `instance_id` dùng để đánh dấu phân biệt từng cá thể riêng lẻ trong cùng một bức ảnh (như phân biệt cái bát thứ nhất với cái bát thứ hai). Nó không phải `class_id` (vì nhiều cái bát có chung `class_id` nhưng khác `instance_id`), và cũng không phải `tracking_id` vì nó chỉ có tác dụng đếm trong phạm vi 1 ảnh tĩnh của bài lab, không dùng để theo dõi vật thể di chuyển qua các frame video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Trả lời: Khi chấm điểm polygon phải đi bám sát theo viền thật của vật thể, chú ý ở các đoạn cong cần chấm dày điểm hơn để không bị gãy góc và không vẽ lấn sang phần bóng đổ.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   Trả lời: Chỗ chồng bát đĩa màu trắng để sát nhau trên kệ làm ranh giới giữa các cái bát rất khó phân biệt. Trước tiên tôi sẽ bám theo guideline để tách từng chiếc phần nhìn thấy được; nhưng nếu ảnh tối và mờ đến mức không biết đâu là viền của cái bát trên hay bát dưới thì phải escalation hỏi Lab Coach để thống nhất cách vẽ, không tự ý gộp chung 2 cái bát làm 1 mask.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn lớp duy nhất cho toàn bộ bức ảnh | Ảnh `traffic` có nhiều xe buýt và ô tô nhưng model đoán top-1 là `cab` (score 0.51) | Xem toàn cảnh bức ảnh và chọn 1 lớp đúng theo tiêu chí ưu tiên của guideline | Kiểm tra xem nhãn đã chọn có đúng với quy tắc ưu tiên trong guideline không |
| Phát hiện vật thể | 1 nhãn lớp + 1 hộp `bbox_xyxy` cho từng vật thể | Ảnh `kitchen` nhận diện thiếu vật ở ngưỡng cao, hoặc nhận nhầm/chồng chéo hộp ở kệ bát đĩa | Vẽ bổ sung box cho vật bị sót, xóa box rác và chỉnh lại 4 cạnh cho ôm sát từng vật | Kiểm tra có sót vật thể nào không, box có bị lệch mép hoặc sai tên lớp không |
| Instance segmentation | 1 nhãn lớp + hộp + 1 đa giác `polygon_xy` cho từng đối tượng | Trên ảnh `segmentation_prediction.png`, vùng mask của `dining table` và `oven` bị lem, viền chỗ các cái bát chồng nhau chưa sát | Chấm và chỉnh lại các đỉnh polygon bám đúng viền từng vật, tách riêng các vật nằm sát nhau | Soi kỹ đường biên mask xem có bị lấn nền, dính 2 vật vào nhau hoặc gãy nét không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Trả lời: Chỉ sử dụng bộ ảnh mẫu công khai có sẵn trong notebook, tuyệt đối không tải ảnh cá nhân/nội bộ lên Colab và không ghi thông tin định danh (họ tên, MSSV, email, SĐT) vào trong file `REPORT.md` hay các file output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Trả lời: Tôi sẽ dừng làm việc với dữ liệu đó ngay và báo cho Lab Coach hoặc Giảng viên phụ trách lớp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS