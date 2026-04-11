#!/usr/bin/env python3
import time
import subprocess
import json
from pathlib import Path


def find_report(name: str):
    p1 = Path('copilot_plan/train_outputs') / name / 'gpu_util_report.json'
    p2 = Path('/ultralytics/runs/detect') / 'copilot_plan' / 'train_outputs' / name / 'gpu_util_report.json'
    for p in (p1, p2):
        if p.exists():
            return p
    return None


def append_summary(report: dict):
    md = Path('copilot_plan/experiment_res/experiment_summary.md')
    md.parent.mkdir(parents=True, exist_ok=True)
    line = f"- {report.get('name')} — imgsz={report.get('imgsz')}, batch={report.get('batch')}, workers={report.get('workers')}, avg_gpu_util={report.get('avg_gpu_util'):.2f}%, max_gpu_util={report.get('max_gpu_util')}%, avg_mem={report.get('avg_gpu_mem_mb'):.0f}MB\n"
    with md.open('a') as f:
        f.write(line)
    print('Appended summary for', report.get('name'))


def run_trial(trial):
    cmd = [
        'python3', 'copilot_plan/monitor_and_train.py',
        '--epochs', '200',
        '--imgsz', str(trial['imgsz']),
        '--batch', str(trial['batch']),
        '--workers', str(trial['workers']),
        '--gpu', str(trial.get('gpu', 0)),
        '--model', trial.get('model', 'copilot_plan/weights/yolo26n.pt'),
        '--name', trial['name']
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    baseline = 'baseline_100_ep_pat10'
    print('Waiting for baseline report...')
    while True:
        p = find_report(baseline)
        if p:
            print('Found baseline report at', p)
            break
        time.sleep(30)

    report = json.loads(p.read_text())
    append_summary(report)

    trials = [
        {'name': 'gpu_tune_trial7', 'imgsz': 1024, 'batch': 40, 'workers': 12},
        {'name': 'gpu_tune_trial8', 'imgsz': 1024, 'batch': 32, 'workers': 12},
        {'name': 'gpu_tune_trial9', 'imgsz': 1280, 'batch': 16, 'workers': 12},
    ]

    for t in trials:
        try:
            run_trial(t)
        except subprocess.CalledProcessError as e:
            print('Trial failed:', t['name'], e)
            continue

        # wait for report
        for _ in range(60):
            rp = find_report(t['name'])
            if rp:
                rpt = json.loads(rp.read_text())
                append_summary(rpt)
                break
            time.sleep(5)
        else:
            print('No gpu report found for', t['name'])

    print('All trials attempted.')


if __name__ == '__main__':
    main()
