# DeepWeeds Lab Day 2 — Báo cáo thực nghiệm

## 1. Tóm tắt

Bài toán phân loại DeepWeeds 9 lớp, dùng fold 0 do tác giả cung cấp. Cấu hình chung kết convnext_tiny, công thức T04, suy luận I02 + temperature scaling. Macro-F1 test qua 3 seed: 0.9776 ± 0.0002; mốc T00 + I00: 0.9696 ± 0.0016; chênh lệch +0.0080. Top-1 test chung kết: 0.9822 ± 0.0002.

## 2. Dữ liệu và thiết lập

Fold 0, train/val/test tách sẵn và không gộp. Xem số đếm, biểu đồ lớp, ảnh mẫu và kiểm tra pipeline trong notebook Bước 0. Cấu hình và log từng lần chạy nằm trong `runs/<exp_id>/seed<k>/`. Môi trường lần chạy: python: 3.13.15, torch: 2.11.0+cu128, torchvision: 0.26.0+cu128, timm: 1.0.29, gpu: Tesla T4, weight_tag: timm/convnext_tiny.in12k_ft_in1k. **Bổ sung tại đây:** link notebook và nhận xét EDA từ dữ liệu thật.

## 3. So sánh backbone

Nguồn: sheet `Backbones`, biểu đồ `curves/backbone_tradeoff.png`.

| exp_id | backbone | macro_f1_val | top1_val | latency_batch1_p50_ms |
|---|---|---|---|---|
| B01 | resnet50 | 0.7920 | 0.8526 | 7.4037 |
| B02 | resnext50_32x4d | 0.8071 | 0.8495 | 7.6849 |
| B03 | convnext_tiny | 0.9641 | 0.9720 | 8.0838 |
| B04 | deit_small_patch16_224 | 0.9570 | 0.9680 | 6.4077 |
| B05 | efficientnet_b0 | 0.7986 | 0.8520 | 10.3915 |

**Bổ sung phân tích:** mô hình nào hội tụ nhanh, mô hình nào cân bằng F1 và độ trễ, vì sao chọn backbone đi tiếp.

## 4. Công thức huấn luyện

Nguồn: sheet `Training`. Mỗi T01–T06 thay một trục so với T00; T07 kết hợp trục B và C. Chỉ một seed ở vòng sàng, nên chênh lệch nhỏ chưa đủ để khẳng định.

| exp_id | axis | macro_f1_val | delta_vs_T00 | f1_chinee_apple_val | f1_snake_weed_val |
|---|---|---|---|---|---|
| T00 | baseline | 0.9641 | 0.0000 | 0.9252 | 0.9193 |
| T01 | A_init | 0.8512 | -0.1129 | 0.8165 | 0.7677 |
| T02 | A_init | 0.3386 | -0.6255 | 0.2200 | 0.3439 |
| T03 | B_aug | 0.9580 | -0.0060 | 0.9155 | 0.9185 |
| T04 | B_aug | 0.9699 | 0.0058 | 0.9374 | 0.9242 |
| T05 | C_loss | 0.9658 | 0.0017 | 0.9256 | 0.9200 |
| T06 | C_loss | 0.9607 | -0.0034 | 0.9263 | 0.9144 |
| T07 | B_plus_C | 0.9694 | 0.0054 | 0.9349 | 0.9249 |

**Bổ sung phân tích:** yếu tố nào giúp, yếu tố nào không, và mức chênh lệch so với nhiễu qua seed.

## 5. Phương pháp suy luận

Nguồn: sheet `Inference`, `Latency`, `curves/inference_tradeoff.png`. Phương pháp chọn từ val: I02.

| exp_id | macro_f1_val | ece_val | p50_ms | p95_ms |
|---|---|---|---|---|
| I00 | 0.9699 | 0.0043 | 6.2513 | 9.4199 |
| I01 | 0.9713 | 0.0047 | 12.8038 | 13.6328 |
| I02 | 0.9766 | 0.0067 | 30.2341 | 31.2999 |
| I03 | 0.9713 | 0.0046 | 12.2817 | 13.3702 |
| I07 | 0.9699 | 0.0039 | 6.1404 | 6.4655 |

**Bổ sung phân tích:** ECE trước/sau hiệu chuẩn và đánh đổi F1 với p50/p95/p99.

## 6. Chung kết và lỗi dự đoán

Nguồn: sheet `Final`, `PerClass`, file dự đoán test và `curves/F01_confusion.png`.

| exp_id | seed | macro_f1_val | macro_f1_test | top1_test | ece_test |
|---|---|---|---|---|---|
| T00 | 0 | 0.9641 | 0.9682 | 0.9746 | 0.0166 |
| T00 | 1 | 0.9685 | 0.9713 | 0.9775 | 0.0134 |
| T00 | 2 | 0.9668 | 0.9693 | 0.9758 | 0.0150 |
| T00 | mean ± std | 0.9664 | 0.9696 | 0.9760 | 0.0150 |
| F01 | 0 | 0.9766 | 0.9775 | 0.9823 | 0.0054 |
| F01 | 1 | 0.9793 | 0.9778 | 0.9820 | 0.0036 |
| F01 | 2 | 0.9761 | 0.9774 | 0.9823 | 0.0038 |
| F01 | mean ± std | 0.9773 | 0.9776 | 0.9822 | 0.0043 |

Recall test Chinee Apple: 0.9366 ± 0.0092; Snake Weed: 0.9641 ± 0.0075. Cặp nhầm nhiều nhất tính trên tổng ma trận: Negatives → Prickly Acacia: 24 lần. Xem `misclassified_seed0.csv` và ảnh gốc để viết giả thuyết nguyên nhân.

## 7. Kết luận và khuyến nghị

Macro-F1 cải thiện +0.0080 so với mốc. Lợi thế lớn hơn std quan sát qua seed. **Bổ sung:** cấu hình đạt giới hạn p95 ≤ 100 ms nếu có, và thành phần đóng góp nhiều nhất theo bảng số liệu.

## 8. Hạn chế và việc tiếp theo

Chỉ một fold và ba seed ở chung kết. Fold được chia ngẫu nhiên, không theo địa điểm chụp, nên điểm test có thể lạc quan khi gặp ảnh khác miền. Ghi rõ các thí nghiệm chưa làm và lỗi phát hiện sau khi mở test, nếu có.

## 9. Phụ lục

Danh sách cấu hình, seed và log: xem `results.xlsx` và `runs/`. **Bổ sung link notebook Kaggle sau khi chạy.**
