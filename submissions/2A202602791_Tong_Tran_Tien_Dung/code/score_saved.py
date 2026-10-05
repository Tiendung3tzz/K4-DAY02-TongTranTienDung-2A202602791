"""Recompute scores from saved CSVs; no model or test image loader."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submission-dir', type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--labels-dir', type=Path)
    args = ap.parse_args()
    sub = args.submission_dir.resolve()
    evaluator = Path(__file__).resolve().parents[3] / 'eval.py'
    if not evaluator.is_file(): raise FileNotFoundError(f'Original evaluator missing: {evaluator}; retain repository layout.')
    pred, out = sub / 'predictions', sub / 'eval_out'
    seeds = json.loads((sub / 'final_selection.json').read_text(encoding='utf-8'))['seeds']
    def files(exp, split):
        paths = [pred / f'{exp}_seed{s}_{split}.csv' for s in seeds]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing: raise FileNotFoundError(f'Missing saved predictions: {missing}')
        return list(map(str, paths))
    base = [sys.executable, '-X', 'utf8', str(evaluator)]
    labels, common = args.labels_dir, []
    if labels is not None:
        for name in ['test_subset0.csv', 'val_subset0.csv']:
            if not (labels / name).is_file(): raise FileNotFoundError(labels / name)
        common += ['--test-csv', str(labels / 'test_subset0.csv')]
        if (labels / 'labels.csv').is_file(): common += ['--labels', str(labels / 'labels.csv')]
    else:
        print('No original split CSVs: saved predictions checked without independent reference validation.')
    for exp in ['F01', 'T00']:
        subprocess.run(base + ['score', '--pred', *files(exp, 'test'), '--tag', exp, '--out', str(out)] + common, check=True)
    latency = pd.read_excel(sub / 'results.xlsx', sheet_name='Latency')
    measured = latency.loc[(latency['config'] == 'I02') & (latency['batch'] == 1), 'p95_ms']
    if len(measured) != 1 or not pd.notna(measured.iloc[0]): raise ValueError('Expected one measured I02 batch-1 p95 value')
    command = base + ['grade', '--final', *files('F01', 'test'), '--baseline', *files('T00', 'test'),
                     '--uncal', *files('F01uncal', 'test'), '--final-val', *files('F01', 'val'),
                     '--latency-p95-ms', str(float(measured.iloc[0])), '--latency-method', 'proper', '--out', str(out)] + common
    if labels is not None: command += ['--val-csv', str(labels / 'val_subset0.csv')]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
