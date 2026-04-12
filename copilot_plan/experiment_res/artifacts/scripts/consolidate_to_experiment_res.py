import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'experiment')
DST = os.path.join(ROOT, 'experiment_res', 'experiments')

os.makedirs(DST, exist_ok=True)

copied = {}
for name in sorted(os.listdir(SRC)):
    run_dir = os.path.join(SRC, name)
    if not os.path.isdir(run_dir):
        continue
    dst_dir = os.path.join(DST, name)
    os.makedirs(dst_dir, exist_ok=True)
    copied[name] = []
    # files to copy from run root
    for fname in ['index.json', 'results.csv', 'gpu_util_report.json']:
        srcf = os.path.join(run_dir, fname)
        if os.path.exists(srcf):
            shutil.copy2(srcf, dst_dir)
            copied[name].append(fname)
    # weights may be under weights/best.pt
    wsrc = os.path.join(run_dir, 'weights', 'best.pt')
    if os.path.exists(wsrc):
        dstwdir = os.path.join(dst_dir, 'weights')
        os.makedirs(dstwdir, exist_ok=True)
        shutil.copy2(wsrc, dstwdir)
        copied[name].append('weights/best.pt')

# also copy per-run csvs from experiment_res/csv if present
csv_src_dir = os.path.join(ROOT, 'experiment_res', 'csv')
if os.path.isdir(csv_src_dir):
    for f in os.listdir(csv_src_dir):
        for run in copied.keys():
            if f.startswith(run + '_'):
                dst_run_dir = os.path.join(DST, run)
                shutil.copy2(os.path.join(csv_src_dir, f), dst_run_dir)
                copied[run].append('csv/' + f)

# print summary
for run, files in copied.items():
    if files:
        print(f"{run}: copied {len(files)} files: {', '.join(files)}")
    else:
        print(f"{run}: no files found to copy")
print('\nDone.')
