import os, csv, json, shutil
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UL_DIR = '/ultralytics/runs/detect/copilot_plan/train_outputs'
EXP_DIR = os.path.join(ROOT, 'experiment')
RES_DIR = os.path.join(ROOT, 'experiment_res', 'experiments')

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

for run in os.listdir(UL_DIR):
    run_src = os.path.join(UL_DIR, run)
    if not os.path.isdir(run_src):
        continue
    # skip tuning/benchmark trial runs
    if 'trial' in run.lower():
        print(f'Skipping trial run {run}')
        continue
    results_src = os.path.join(run_src, 'results.csv')
    if not os.path.exists(results_src):
        continue
    # ensure target run dir in EXP_DIR
    dst_run = os.path.join(EXP_DIR, run)
    os.makedirs(dst_run, exist_ok=True)
    shutil.copy2(results_src, dst_run)
    # copy gpu report if exists
    gsrc = os.path.join(run_src, 'gpu_util_report.json')
    if os.path.exists(gsrc):
        shutil.copy2(gsrc, dst_run)
    # copy best.pt if exists
    best = os.path.join(run_src, 'weights', 'best.pt')
    if os.path.exists(best):
        dstw = os.path.join(dst_run, 'weights')
        os.makedirs(dstw, exist_ok=True)
        shutil.copy2(best, dstw)
    # compute best epoch and best mAP50 from results.csv
    best_epoch = None
    best_mAP = 0.0
    with open(results_src, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row.get('epoch') or row.get('Epoch') or 0)
                m = float(row.get('metrics/mAP50(B)') or row.get('metrics/mAP50') or row.get('mAP50') or 0.0)
                if m >= best_mAP:
                    best_mAP = m
                    best_epoch = epoch
            except Exception:
                continue
    index = {'name': run, 'best_epoch': best_epoch or 0, 'best_mAP': round(best_mAP,5)}
    with open(os.path.join(dst_run, 'index.json'), 'w') as jf:
        json.dump(index, jf)
    # also copy to experiment_res/experiments
    dst_res_run = os.path.join(RES_DIR, run)
    os.makedirs(dst_res_run, exist_ok=True)
    for fname in ['results.csv', 'index.json', 'gpu_util_report.json']:
        srcf = os.path.join(dst_run, fname)
        if os.path.exists(srcf):
            shutil.copy2(srcf, dst_res_run)
    print(f'Indexed {run}: best_epoch={index["best_epoch"]} best_mAP={index["best_mAP"]}')
print('Done.')
