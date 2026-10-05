"""Make an error gallery from saved predictions and original images, without a model."""
from pathlib import Path
import argparse
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = ['Chinee Apple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly Acacia',
         'Rubber Vine', 'Siam Weed', 'Snake Weed', 'Negative']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images-dir', type=Path, required=True)
    ap.add_argument('--submission-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    args = ap.parse_args()
    sub = args.submission_dir
    frame = pd.read_csv(sub / 'predictions/F01_seed0_test.csv')
    wrong = frame.loc[frame.y_true != frame.y_pred].copy()
    if wrong.empty: raise ValueError('No saved misclassified images')
    wrong['confidence'] = wrong[[f'p{i}' for i in range(9)]].max(axis=1)
    discussed = ['20170207-153439-0.jpg', '20170729-091822-2.jpg', '20171113-060629-1.jpg']
    chosen = [wrong.loc[wrong.Filename == name].iloc[0] for name in discussed if (wrong.Filename == name).any()]
    used = {row.Filename for row in chosen}
    candidates = wrong.sort_values('confidence', ascending=False)
    pairs = {(int(row.y_true), int(row.y_pred)) for row in chosen}
    for _, row in candidates.iterrows():
        pair = (int(row.y_true), int(row.y_pred))
        if row.Filename not in used and pair not in pairs:
            chosen.append(row);used.add(row.Filename);pairs.add(pair)
        if len(chosen) >= 6: break
    for _, row in candidates.iterrows():
        if len(chosen) >= 6: break
        if row.Filename not in used: chosen.append(row);used.add(row.Filename)
    missing = [str(args.images_dir / row.Filename) for row in chosen if not (args.images_dir / row.Filename).is_file()]
    if missing: raise FileNotFoundError(f'Missing original images: {missing}')
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, row in zip(axes.flat, chosen):
        with Image.open(args.images_dir / row.Filename) as im: ax.imshow(im.convert('RGB'))
        ax.set_title(f'True: {NAMES[int(row.y_true)]}\nPred: {NAMES[int(row.y_pred)]} ({row.confidence:.4f})\n{row.Filename}', fontsize=9)
        ax.axis('off')
    for ax in list(axes.flat)[len(chosen):]: ax.axis('off')
    fig.suptitle('F01 seed 0: saved test errors after temperature scaling', fontsize=13)
    fig.tight_layout()
    output = sub / 'curves/F01_error_examples.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160);plt.close(fig)
    pd.DataFrame(chosen)[['Filename','y_true','y_pred','confidence']].to_csv(sub / 'eval_out/error_gallery_examples.csv', index=False)
    report = sub / 'report.md'
    if report.exists():
        content = report.read_text(encoding='utf-8')
        caption = '![Ảnh test dự đoán sai từ CSV F01 seed 0](curves/F01_error_examples.png)'
        heading = '### 6.4. Tự chấm bằng eval.py'
        if caption not in content:
            content = content.replace(heading, caption + '\n\n' + heading, 1) if heading in content else content + '\n\n' + caption + '\n'
            content = content.replace(
                'Bản output tải về không kèm ảnh test gốc, nên báo cáo chỉ cung cấp tên file và số liệu, chưa gán nguyên nhân cụ thể cho từng ảnh. Script `code/make_error_gallery.py` có thể tạo ảnh minh họa trên Kaggle từ các dự đoán đã có, không cần chạy lại model; cách dùng nằm trong README.',
                'Bảng ảnh lỗi bên dưới được tạo từ CSV F01 seed 0 và ảnh Input gốc trên Kaggle, không chạy lại model. Các giả thuyết nguyên nhân cần được xác nhận bằng việc xem từng ảnh; bảng này chỉ minh họa sáu trường hợp, chưa khảo sát toàn bộ lỗi bằng hình ảnh.')
            content = content.replace('Chưa có ảnh test gốc trong gói tải về để phân tích lỗi bằng hình ảnh.',
                                      'Bảng ảnh lỗi chỉ minh họa sáu trường hợp; chưa khảo sát toàn bộ lỗi bằng hình ảnh.')
            report.write_text(content, encoding='utf-8')
    print(f'Created {output}; no new model/test inference was run.')


if __name__ == '__main__':
    main()
