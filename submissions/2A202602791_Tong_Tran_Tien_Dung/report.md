# DeepWeeds Lab Day 2 — Báo cáo thực nghiệm

> Bản chờ kết quả Kaggle. Ô Bước 5 trong `code/lab_day2.ipynb` sẽ thay tài liệu này bằng các bảng và số liệu tính từ file dự đoán thật. Không điền số liệu chưa chạy.

## 1. Tóm tắt

Ghi cấu hình cuối, macro-F1 và top-1 test trung bình ± std qua ít nhất ba seed, cùng mức cải thiện so với T00 + I00.

## 2. Dữ liệu và thiết lập

Ghi số ảnh train/val/test fold 0, biểu đồ phân bố lớp, phần cứng Kaggle, phiên bản thư viện, seed và công thức T00. Xác nhận split không chồng lấp và không dùng test để chọn cấu hình.

## 3. So sánh backbone

Trích bảng `Backbones` và biểu đồ F1 theo độ trễ. Nêu lý do chọn 1–2 backbone dựa trên số đo.

## 4. Công thức huấn luyện

Trích bảng `Training`; so sánh từng trục với T00 và một cấu hình kết hợp. Phân biệt chênh lệch nhỏ với nhiễu qua seed.

## 5. Phương pháp suy luận

Trích bảng `Inference` và `Latency`; nêu ECE trước/sau khớp T trên val và đánh đổi chất lượng với p50/p95/p99.

## 6. Chung kết và phân tích lỗi

Trích `Final`, `PerClass`, ma trận nhầm lẫn và xem ảnh từ `misclassified_seed0.csv`. Nêu recall Chinee Apple và Snake Weed.

## 7. Kết luận và khuyến nghị

Nêu cấu hình tốt nhất, chênh lệch với mốc có vượt std không, và lựa chọn khi cần p95 ≤ 100 ms.

## 8. Hạn chế và việc tiếp theo

Nêu giới hạn của một fold chia ngẫu nhiên, số seed, rủi ro lệch phân phối theo địa điểm hoặc thời gian và các thí nghiệm chưa làm.

## 9. Phụ lục

Link Kaggle notebook; danh sách `exp_id`, cấu hình và đường dẫn log trong `runs/`.
