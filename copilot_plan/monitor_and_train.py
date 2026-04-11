#!/usr/bin/env python3
import argparse
import threading
import time
import subprocess
import json
from pathlib import Path

from ultralytics import YOLO


def monitor(stop_event, interval, samples, gpu_idx):
    cmd = ['nvidia-smi', '-i', str(gpu_idx), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits']
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(cmd)
            line = out.decode().strip().split(',')
            util = int(line[0].strip())
            mem = int(line[1].strip())
            samples.append({'t': time.time(), 'util': util, 'mem': mem})
        except Exception:
            samples.append({'t': time.time(), 'util': 0, 'mem': 0})
        time.sleep(interval)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--model', type=str, default='yolo26n.pt')
    p.add_argument('--name', type=str, default='tune_gpu_test')
    p.add_argument('--interval', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=None)
    args = p.parse_args()

    out_dir = Path('copilot_plan/train_outputs') / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    stop_event = threading.Event()
    th = threading.Thread(target=monitor, args=(stop_event, args.interval, samples, args.gpu))
    th.start()

    model = YOLO(args.model)
    start = time.time()
    try:
        # build train kwargs and include patience if provided
        train_kwargs = dict(data='data/urpc.yaml', epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.gpu, workers=args.workers, project=str(out_dir.parent), name=args.name, exist_ok=True, cache=False)
        if args.patience is not None:
            train_kwargs['patience'] = args.patience
        model.train(**train_kwargs)
    except Exception as e:
        stop_event.set()
        th.join()
        print('Training failed:', e)
        raise
    stop_event.set()
    th.join()
    elapsed = time.time() - start

    utils = [s['util'] for s in samples if 'util' in s]
    mems = [s['mem'] for s in samples if 'mem' in s]
    avg_util = sum(utils)/len(utils) if utils else 0
    max_util = max(utils) if utils else 0
    avg_mem = sum(mems)/len(mems) if mems else 0
    report = {
        'name': args.name,
        'model': args.model,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'workers': args.workers,
        'elapsed_s': elapsed,
        'samples': len(samples),
        'avg_gpu_util': avg_util,
        'max_gpu_util': max_util,
        'avg_gpu_mem_mb': avg_mem
    }
    # attempt to read Ultralytics results.csv to capture best epoch/mAP
    try:
        ul_run_dir = Path('/ultralytics/runs/detect') / 'copilot_plan' / 'train_outputs' / args.name
        results_csv = out_dir / args.name / 'results.csv'
        if not results_csv.exists() and ul_run_dir.exists():
            results_csv = ul_run_dir / 'results.csv'
        if results_csv.exists():
            import csv
            with results_csv.open() as rc:
                reader = csv.DictReader(rc)
                best_epoch = None
                best_val = -1.0
                # find a header with 'mAP' preferably 0.5
                mcol = None
                for h in reader.fieldnames:
                    if '0.5' in h and 'mAP' in h:
                        mcol = h
                        break
                if mcol is None:
                    for h in reader.fieldnames:
                        if 'mAP' in h:
                            mcol = h
                            break
                # iterate rows
                for row in reader:
                    try:
                        val = float(row.get(mcol, -1)) if mcol else -1
                    except Exception:
                        val = -1
                    if val > best_val:
                        best_val = val
                        best_epoch = row.get('epoch') or row.get('Epoch')
                if best_epoch is not None:
                    report['best_epoch'] = int(best_epoch)
                    report['best_mAP'] = best_val
    except Exception:
        pass

    with open(out_dir / 'gpu_util_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        # also save a copy next to Ultralytics run directory if present
        try:
            ul_run_dir = Path('/ultralytics/runs/detect') / 'copilot_plan' / 'train_outputs' / args.name
            ul_run_dir.mkdir(parents=True, exist_ok=True)
            with open(ul_run_dir / 'gpu_util_report.json', 'w') as f2:
                json.dump(report, f2, indent=2)
        except Exception:
            pass
        print('GPU util report saved to', out_dir / 'gpu_util_report.json')


if __name__ == '__main__':
    main()
