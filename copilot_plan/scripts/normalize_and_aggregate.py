#!/usr/bin/env python3
"""Normalize Ultralytics results.csv across runs, extract mAP50 and loss series,
copy key artifacts to copilot_plan/experiment/, and generate comparison plots.

Usage: python3 copilot_plan/scripts/normalize_and_aggregate.py
"""
import csv
import json
import os
from pathlib import Path
import shutil
import math

try:
    import pandas as pd
    import matplotlib.pyplot as plt
except Exception:
    pd = None
    plt = None

ROOT = Path.cwd()
CSV_OUT = ROOT / 'copilot_plan' / 'experiment_res' / 'csv'
CSV_OUT.mkdir(parents=True, exist_ok=True)
EXP_DIR = ROOT / 'copilot_plan' / 'experiment'
EXP_DIR.mkdir(parents=True, exist_ok=True)

def find_results():
    runs = []
    # copilot local outputs
    for p in (ROOT / 'copilot_plan' / 'train_outputs').glob('*'):
        if p.is_dir() and (p / 'results.csv').exists():
            runs.append((p.name, p / 'results.csv'))
    # Ultralytics runs
    ul_base = ROOT / 'ultralytics' / 'runs' / 'detect' / 'copilot_plan' / 'train_outputs'
    if ul_base.exists():
        for p in ul_base.glob('*'):
            if p.is_dir() and (p / 'results.csv').exists():
                runs.append((p.name, p / 'results.csv'))
    # also check absolute /ultralytics path (training may write to root-level ultralytics)
    abs_ul = Path('/') / 'ultralytics' / 'runs' / 'detect' / 'copilot_plan' / 'train_outputs'
    if abs_ul.exists():
        for p in abs_ul.glob('*'):
            if p.is_dir() and (p / 'results.csv').exists():
                runs.append((p.name, p / 'results.csv'))
    return runs

def detect_map_column(fieldnames):
    # prefer explicit 0.5 columns
    for h in fieldnames:
        if '0.5' in h and 'mAP' in h:
            return h
    for h in fieldnames:
        if 'mAP50' in h or 'mAP@0.5' in h or ("mAP" in h and "0.5" in h):
            return h
    for h in fieldnames:
        if 'mAP' in h:
            return h
    return None

def normalize_run(name, csvpath):
    out_m = CSV_OUT / f"{name}_mAP50.csv"
    out_l = CSV_OUT / f"{name}_loss.csv"
    rows = []
    with open(csvpath) as f:
        reader = csv.DictReader(f)
        mcol = detect_map_column(reader.fieldnames or [])
        # determine loss columns
        box = next((h for h in reader.fieldnames if 'box' in h.lower()), None)
        cls = next((h for h in reader.fieldnames if 'cls' in h.lower()), None)
        dfl = next((h for h in reader.fieldnames if 'dfl' in h.lower()), None)
        epcol = next((h for h in reader.fieldnames if h.lower() in ('epoch','ep','epoch')), 'epoch')
        for r in reader:
            rows.append(r)
    # write mAP csv
    with open(out_m, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch','mAP50'])
        for r in rows:
            try:
                epoch = int(r.get(epcol) or r.get('Epoch') or r.get('ep') or 0)
            except Exception:
                epoch = 0
            val = ''
            if mcol and r.get(mcol) not in (None,''):
                try:
                    val = float(r.get(mcol))
                except Exception:
                    val = ''
            w.writerow([epoch, val])
    # write loss csv
    with open(out_l, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch','box_loss','cls_loss','dfl_loss'])
        for r in rows:
            try:
                epoch = int(r.get(epcol) or r.get('Epoch') or r.get('ep') or 0)
            except Exception:
                epoch = 0
            bl = r.get(box) or ''
            cl = r.get(cls) or ''
            dl = r.get(dfl) or ''
            w.writerow([epoch, bl, cl, dl])
    return {'name': name, 'mcol': mcol, 'box': box, 'cls': cls, 'dfl': dfl, 'epoch_col': epcol}

def make_plots():
    # gather all *_mAP50.csv
    files = list(CSV_OUT.glob('*_mAP50.csv'))
    if not files:
        print('No mAP50 series found')
        return
    if pd is None or plt is None:
        print('pandas/matplotlib not available; skipping plots')
        return
    df_list = []
    for f in files:
        name = f.stem.replace('_mAP50','')
        d = pd.read_csv(f)
        d = d.dropna(subset=['mAP50'])
        if d.empty:
            continue
        df_list.append((name, d.sort_values('epoch')))
    # mAP plot
    plt.figure(figsize=(8,6))
    for name,d in df_list:
        plt.plot(d['epoch'], d['mAP50'], label=name)
    plt.xlabel('epoch')
    plt.ylabel('mAP50')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROOT / 'copilot_plan' / 'experiment_res' / 'mAP50_comparison.png')
    plt.close()
    # loss plot (use loss csvs)
    files_l = list(CSV_OUT.glob('*_loss.csv'))
    df_list = []
    for f in files_l:
        name = f.stem.replace('_loss','')
        d = pd.read_csv(f)
        if 'box_loss' in d.columns:
            df_list.append((name, d.sort_values('epoch')))
    plt.figure(figsize=(8,6))
    for name,d in df_list:
        plt.plot(d['epoch'], d['box_loss'], label=name)
    plt.xlabel('epoch')
    plt.ylabel('box_loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROOT / 'copilot_plan' / 'experiment_res' / 'loss_comparison.png')
    plt.close()

def create_run_index(name, results_csv_path):
    # create copilot_plan/experiment/<name>/ and copy artifacts
    run_dir = EXP_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    # copy results.csv
    try:
        shutil.copy(results_csv_path, run_dir / 'results.csv')
    except Exception:
        pass
    # copy gpu util if exists
    local_gpu = ROOT / 'copilot_plan' / 'train_outputs' / name / 'gpu_util_report.json'
    ul_gpu = ROOT / 'ultralytics' / 'runs' / 'detect' / 'copilot_plan' / 'train_outputs' / name / 'gpu_util_report.json'
    for g in (local_gpu, ul_gpu):
        if g.exists():
            try:
                shutil.copy(g, run_dir / 'gpu_util_report.json')
            except Exception:
                pass
            break
    # copy best.pt if exists
    best1 = ROOT / 'copilot_plan' / 'train_outputs' / name / 'weights' / 'best.pt'
    best2 = ROOT / 'ultralytics' / 'runs' / 'detect' / 'copilot_plan' / 'train_outputs' / name / 'weights' / 'best.pt'
    for b in (best1, best2):
        if b.exists():
            try:
                shutil.copy(b, run_dir / 'best.pt')
            except Exception:
                pass
            break
    # parse a small summary from results.csv if present
    summary = {'name': name}
    try:
        with open(results_csv_path) as f:
            reader = csv.DictReader(f)
            mcol = detect_map_column(reader.fieldnames or [])
            best_val = -1
            best_epoch = None
            for r in reader:
                try:
                    val = float(r.get(mcol, -1)) if mcol else -1
                except Exception:
                    val = -1
                if val > best_val:
                    best_val = val
                    best_epoch = r.get('epoch') or r.get('Epoch')
            if best_epoch is not None:
                summary['best_epoch'] = int(best_epoch)
                summary['best_mAP'] = best_val
    except Exception:
        pass
    with open(run_dir / 'index.json','w') as f:
        json.dump(summary, f, indent=2)

def main():
    runs = find_results()
    if not runs:
        print('No runs found')
        return
    info = []
    for name, csvp in runs:
        print('Normalizing', name)
        res = normalize_run(name, csvp)
        info.append(res)
        create_run_index(name, csvp)
    make_plots()
    print('Done. CSV outputs in', CSV_OUT)

if __name__ == '__main__':
    main()
