# DeepWeeds Lab Day 2 — Kaggle run instructions
kaggle : `https://www.kaggle.com/code/dung3te/notebook1167a23f8c`
The implementation is in `code/`. The notebook `code/lab_day2.ipynb` covers Steps 0–5. No dataset or checkpoints are stored in this submission.

1. Upload or clone the repository into Kaggle Working and attach the original DeepWeeds images and official CSV files as Kaggle Inputs.
2. Open the notebook and edit `REPO_DIR`, `IMAGES_DIR`, and `LABELS_DIR` in its first cells to match your Kaggle paths. Run the Step 0 cells. The split check requires all 17,509 images and the official fold 0 files.
3. Keep the fold 0 CSVs unchanged. Use train only for optimization, val for selection, and test only after the final configuration is fixed.
4. Run `python -m unittest -v test_pipeline.py` from the `code/` folder for the code checks. The repository's original tests run with `python -m unittest discover -s tests` from the repository root; they intentionally inspect the untouched `starter/` templates.
5. After Step 0 passes, a baseline experiment can be launched from `code/` with `python train.py --set exp_id=T00 backbone=resnet50 seed=0 images_dir=/kaggle/input/.../images labels_dir=/kaggle/input/.../labels`. The code defaults to val-only reporting. Set `save_test_predictions=true` only for the final run after choosing the configuration using val.

## Step 1 on Kaggle

Enable a GPU, then run the Step 1 notebook cell after Step 0. It trains five architectures with identical T00 settings and seed 0: ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small, and EfficientNet-B0. Every run selects its checkpoint by val macro-F1. The cell writes actual measurements to `results.xlsx` / `Backbones`, individual learning curves to `curves/`, and an F1-versus-latency chart. It resumes completed runs when the saved configuration and checkpoint still match. Batch-1 latency uses 10 warmups and 50 synchronized measurements; image loading and preprocessing are excluded.

Review the completed table and Pareto candidates before naming the 1–2 backbones for Steps 2–3. With only one screening seed, small score differences are inconclusive.

If a run finished every epoch but stopped while loading `best.pt`, keep its `runs/<exp_id>/seed<k>/` folder. After updating the code and restarting the Kaggle kernel, rerun the Step 1 cell. It checks `history.csv` against the saved checkpoint, completes evaluation, and writes the missing summary without repeating the training epochs. Do not delete the cloned repository while it contains these run files.

## Steps 2–5 on Kaggle

- Step 2 runs T00 and six one-factor trials across initialization, augmentation, and loss, then tests one B+C combination. The `Training` sheet records every change and its val macro-F1 difference from T00.
- Step 3 evaluates I00, horizontal flip, five crops, logit aggregation, and temperature scaling on val. It measures batch 1 and batch 16 with synchronized GPU timing and writes `Inference` and `Latency` sheets.
- Step 4 requires you to review the val results and set `RUN_FINAL_TRAINING=True`. After F01 and T00 are trained with seeds 0, 1, 2, set `RUN_FINAL_TEST=True` to create test predictions. Each seed has a `test_started.json` marker; an interrupted test pass requires manual inspection before any rerun. The notebook then calls the original `eval.py score` and `grade` on the saved CSVs.
- Step 5 builds the seven-sheet `results.xlsx`, a confusion matrix, a misclassification list, and a data-filled `report.md`. Read the report and add your qualitative analysis, hardware details, library versions, and Kaggle notebook link.

The checked-in `report.md` is a template until Kaggle produces actual predictions. No metrics are pre-filled or invented.

The notebook installs `timm`, `thop`, `matplotlib`, and `openpyxl` on Kaggle. PyTorch, torchvision, pandas, NumPy, scikit-learn, and Pillow must also be available. Record actual library versions, the Kaggle notebook link, and measured outputs after running; these values cannot be filled in without the Kaggle run.
