#!/usr/bin/env python3
"""Export best .pt models in copilot_plan/experiment_res/weights/ to ONNX (FP16 if supported).
Usage: python3 copilot_plan/scripts/export_models.py
"""
import sys
from pathlib import Path

ROOT = Path.cwd()
WEIGHT_DIR = ROOT / 'copilot_plan' / 'experiment_res' / 'weights'
OUT_DIR = WEIGHT_DIR

try:
    from ultralytics import YOLO
except Exception as e:
    print('ultralytics not installed or import failed:', e)
    sys.exit(2)

models = list(WEIGHT_DIR.glob('*_best.pt'))
if not models:
    print('No best.pt models found in', WEIGHT_DIR)
    sys.exit(0)

for m in models:
    name = m.stem
    out_path = OUT_DIR / f'{name}.onnx'
    if out_path.exists():
        print('Skipping existing', out_path)
        continue
    print('Exporting', m, '->', out_path)
    try:
        model = YOLO(str(m))
        # try FP16 ONNX export first
        model.export(format='onnx', imgsz=640, half=True, opset=12, device=0)
        # ultralytics export writes to same folder by default; move/rename if needed
        # check common names
        cand = Path('best.onnx')
        if cand.exists():
            cand.rename(out_path)
        else:
            # search workdir for .onnx
            for f in Path('.').glob('**/*.onnx'):
                if f.stat().st_mtime > (m.stat().st_mtime - 1):
                    try:
                        f.rename(out_path)
                        break
                    except Exception:
                        pass
        print('Exported to', out_path)
    except Exception as e:
        print('FP16 ONNX export failed for', m, e)
        print('Retrying without half...')
        try:
            model = YOLO(str(m))
            model.export(format='onnx', imgsz=640, half=False, opset=12, device=0)
            cand = Path('best.onnx')
            if cand.exists():
                cand.rename(out_path)
            else:
                for f in Path('.').glob('**/*.onnx'):
                    if f.stat().st_mtime > (m.stat().st_mtime - 1):
                        try:
                            f.rename(out_path)
                            break
                        except Exception:
                            pass
            print('Exported to', out_path)
        except Exception as e2:
            print('ONNX export failed for', m, e2)

print('Done.')
