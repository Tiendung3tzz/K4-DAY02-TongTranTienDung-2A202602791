# DeepWeeds - Lab Day 2

**Sinh viên:** Tống Trần Tiến Dũng - **MSSV:** 2A202602791  
**Notebook Kaggle:** https://www.kaggle.com/code/dung3te/notebook1167a23f8c

## Kết quả đã chạy

Hoàn tất 5 backbone, 8 cấu hình huấn luyện, 5 cách suy luận, chung kết và mốc với seed 0/1/2. Cấu hình chọn trên val: **ConvNeXt-Tiny + CutMix (T04) + 5 crop (I02) + temperature scaling**.

| Chỉ số test | T00 + I00 | F01 |
|---|---|---|
| Macro-F1 | 0.9696 ± 0.0016 | **0.9776 ± 0.0002** |
| Top-1 | 0.9760 ± 0.0014 | **0.9822 ± 0.0002** |
| ECE | 0.0150 ± 0.0016 | **0.0043 ± 0.0010** |

Std mẫu qua ba seed (ddof = 1). Macro-F1 tăng 0.007969, tương đương 0.80 điểm phần trăm. Phép đo I02 trên T04 seed 0 cho p95 batch 1 = **31.2999 ms**, Tesla T4, FP32. Không bao gồm đọc ảnh/tiền xử lý CPU; chưa đo riêng toàn bộ F01 + TS.

Tự chấm bằng `eval.py` gốc: **19/20 mục I**, theo ngưỡng tạm thời; không phải điểm toàn bài hay điểm giảng viên xác nhận. Bản Kaggle ban đầu thiếu latency, chấm 17/18 ý đã có số liệu, được giữ ở `eval_out/grade_I_kaggle.json`. Bản bổ sung ở `grade_I.json` và `grade_I.md` chỉ dùng CSV và số đo đã lưu.

## File nộp và bằng chứng

| File/thư mục | Nội dung |
|---|---|
| `report.md` | Báo cáo kết quả, nhận xét, giới hạn và phụ lục |
| `results.xlsx` | 7 sheet Backbones, Training, Inference, Final, PerClass, Latency, Summary |
| `code/lab_day2.ipynb` | Notebook đã chạy, có output EDA và các thí nghiệm |
| `code/*.py` | Dataset, model, loss, train, inference, benchmark, workflow và tiện ích đọc output |
| `predictions/` | CSV val/test từng seed F01, T00 và F01uncal |
| `runs/<ID>/seed<k>/` | Config, environment, history, summary, kiểm tra split và marker test |
| `curves/` | Đường cong train, EDA, đánh đổi F1/latency, ma trận nhầm lẫn |
| `eval_out/` | Chỉ số chính thức, per-class, confusion CSV và tự chấm |
| `final_selection.json` | Công thức, suy luận và seed đã chốt trước test |
| `misclassified_seed0.csv` | 62 ảnh test dự đoán sai của F01 seed 0 |

Gói nộp giữ cấu trúc repository: `eval.py` gốc ở thư mục gốc, bài làm trong `submissions/2A202602791_Tong_Tran_Tien_Dung/`. Không kèm dataset hay `best.pt`; checkpoint không có trong output đã tải về. Không thể tiếp tục từ trọng số cũ chỉ bằng notebook hoặc log.

## Kiểm tra lại từ CSV đã lưu

Chạy từ thư mục gốc repository. Cần NumPy, pandas, openpyxl; không cần GPU hoặc ảnh:

```bash
python submissions/2A202602791_Tong_Tran_Tien_Dung/code/score_saved.py
```

Nếu có CSV split chính thức, thêm đối chiếu tên ảnh/nhãn:

```bash
python submissions/2A202602791_Tong_Tran_Tien_Dung/code/score_saved.py --labels-dir /kaggle/working/data/labels
```

Script gọi evaluator gốc trên các CSV và lấy p95 I02 batch 1 từ workbook để tự chấm. Không ghi lại workbook/báo cáo đã hoàn thiện. Notebook Kaggle đã chấm với `test_subset0.csv` và `val_subset0.csv` thành công. Bản tải về thiếu CSV split gốc nên kiểm tra cục bộ chưa tái đối chiếu độc lập Filename/Label với nguồn đó. Xem `eval_out/artifact_audit.json`.

## Ảnh minh họa dự đoán sai

Báo cáo có ma trận nhầm lẫn và ví dụ Filename, nhãn, confidence. Output chưa kèm ảnh test gốc. Để bổ sung bảng ảnh lỗi, chạy ô sau trên Kaggle đã gắn ảnh Input. Chỉ đọc ảnh và CSV có sẵn:

```python
import subprocess, sys
subprocess.run([
    sys.executable, str(SUBMISSION_DIR / "code" / "make_error_gallery.py"),
    "--images-dir", str(IMAGES_DIR),
    "--submission-dir", str(SUBMISSION_DIR),
], check=True)
```

Script tạo `curves/F01_error_examples.png` gồm 6 ảnh, nhãn thật, dự đoán, confidence và Filename; bổ sung hình vào mục phân tích lỗi của `report.md`. Tải lại hai file này để cập nhật bản nộp. Nhận xét nguyên nhân phải dựa trên ảnh thực tế; báo cáo hiện tại ghi rõ giới hạn này.

## Tái lập trên Kaggle

Môi trường đã ghi: Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, Tesla T4. Tag cuối: `timm/convnext_tiny.in12k_ft_in1k`. Tag từng backbone nằm trong workbook/báo cáo. Bộ tiền huấn luyện khác nhau là giới hạn của so sánh.

1. Bật GPU, gắn DeepWeeds images và CSV fold 0 chính thức vào Input.
2. Chỉnh `REPO_DIR`, `IMAGES_DIR`, `LABELS_DIR` trong notebook. Lần chạy gốc: ảnh `/kaggle/input/datasets/quchuy2k4/images`, nhãn `/kaggle/working/data/labels`.
3. Chạy Bước 0 kiểm tra 17,509 ảnh, giao tập rỗng và không thiếu ảnh. Giữ CSV fold 0, không gộp val vào train.
4. Dùng chung `train.run(cfg)`: 12 epoch, batch 64, AdamW, LR backbone/head 1e-4/1e-3, weight decay 0.05, warmup 1 epoch rồi cosine, AMP. Chọn checkpoint theo macro-F1 val.
5. Bước 1-3 chọn trên val. Bước 4 chốt cấu hình, train mốc/chung kết cùng seed 0/1/2, ghi test một lần mỗi seed. Bài hiện tại đã đọc test; không đổi cấu hình theo test rồi chạy lại.

Run hoàn tất chỉ tái sử dụng được khi config khớp và có summary cùng checkpoint. Một run dở chưa lưu optimizer/scheduler để resume giữa epoch. Giữ `test_started.json` khi khôi phục output.

Kiểm tra code từ `code/`: `python -m unittest -v test_pipeline.py`. Test repository từ thư mục gốc: `python -m unittest discover -s tests`. `eval.py` giữ nguyên.

## Khi nộp

- Giữ `report.md` cạnh `curves/` để hình hiển thị đúng.
- Giữ đủ CSV ba seed F01/T00, cùng F01uncal và val để kiểm tra hiệu chuẩn.
- Mở quyền xem notebook cho giảng viên. Link lấy từ README do sinh viên cung cấp; quyền truy cập chưa được xác minh trong lần hoàn thiện cục bộ.
- Nếu cần ảnh minh họa từng ca sai, chạy ô tạo bảng ảnh trên rồi tải lại báo cáo/hình.

Nguồn mọi con số là CSV, log Kaggle và `eval.py` gốc. Không điền số liệu cho thí nghiệm chưa chạy.
