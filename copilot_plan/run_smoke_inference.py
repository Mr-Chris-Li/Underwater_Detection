from pathlib import Path
import time
import json
import sys

OUT_DIR = Path('copilot_plan/smoke_outputs')
WEIGHTS = Path('copilot_plan/weights/yolo26n.pt')
# Prefer yolov11 weight if available; otherwise fall back to model name which Ultralytics can download
LOCAL_YOLOV11 = Path('copilot_plan/weights/yolov11n.pt')
CANDIDATES = [Path('data/URPC2021/images/val'), Path('data/URPC2021/images/train'), Path('data')]
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / 'smoke_run.log'

def find_images(max_images=10):
    imgs = []
    for d in CANDIDATES:
        if d.exists():
            imgs.extend(sorted([p for p in d.rglob('*.jpg')]))
            imgs.extend(sorted([p for p in d.rglob('*.png')]))
        if len(imgs) >= max_images:
            break
    return imgs[:max_images]


def main():
    with open(LOG, 'w') as logf:
        logf.write('Starting smoke inference\n')
        logf.flush()
        # choose weights: prefer local yolov11 if present and non-empty
        chosen = None
        if LOCAL_YOLOV11.exists() and LOCAL_YOLOV11.stat().st_size > 1024:
            chosen = str(LOCAL_YOLOV11)
            logf.write(f'Using local yolov11 weights: {LOCAL_YOLOV11}\n')
        elif WEIGHTS.exists() and WEIGHTS.stat().st_size > 1024:
            chosen = str(WEIGHTS)
            logf.write(f'Using fallback weights: {WEIGHTS}\n')
        else:
            # no local file; try model name (Ultralytics will download if available)
            chosen = 'yolov11n.pt'
            logf.write('No suitable local weights found; will attempt to use model name yolov11n.pt (will be downloaded by ultralytics if available)\n')
        imgs = find_images(10)
        if not imgs:
            logf.write('No images found in candidate dirs.\n')
            print('No images found in candidate dirs.', file=sys.stderr)
            return 3
        logf.write(f'Found {len(imgs)} images\n')
        try:
            from ultralytics import YOLO
        except Exception as e:
            logf.write(f'Import error: {e}\n')
            print('Missing dependency: ultralytics. Please pip install ultralytics', file=sys.stderr)
            return 4
        try:
            model = YOLO(chosen)
        except Exception as e:
            logf.write(f'Error loading model {chosen}: {e}\n')
            print(f'Error loading model {chosen}: {e}', file=sys.stderr)
            return 6
        summary = []
        for img in imgs:
            t0 = time.time()
            try:
                # save=True will save results into OUT_DIR/run
                res = model.predict(source=str(img), device=0, save=True, project=str(OUT_DIR), name='run', exist_ok=True)
            except Exception as e:
                logf.write(f'Error running predict on {img}: {e}\n')
                print(f'Error running predict on {img}: {e}', file=sys.stderr)
                return 5
            t_ms = (time.time() - t0) * 1000
            # extract confidences if available
            confs = []
            if len(res) and hasattr(res[0], 'boxes'):
                try:
                    for b in res[0].boxes:
                        # b.conf may be tensor-like
                        val = float(getattr(b, 'conf', 0.0))
                        confs.append(val)
                except Exception:
                    pass
            summary.append({
                'image': str(img),
                'time_ms': t_ms,
                'n_detections': len(confs),
                'max_conf': max(confs) if confs else 0.0,
                'mean_conf': sum(confs)/len(confs) if confs else 0.0
            })
            logf.write(f'Processed {img} in {t_ms:.1f} ms, detections={len(confs)}\n')
            logf.flush()
        # write summary
        with open(OUT_DIR / 'smoke_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        logf.write('Completed successfully\n')
    print('Done. Results saved to', OUT_DIR)
    return 0

if __name__ == '__main__':
    sys.exit(main())
