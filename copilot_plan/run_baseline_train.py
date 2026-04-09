from pathlib import Path
import sys, time

OUT = Path('copilot_plan/train_outputs')
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / 'baseline_run.log'

with open(LOG, 'w') as logf:
    logf.write('Starting baseline training (20 epochs)\n')
    try:
        from ultralytics import YOLO
    except Exception as e:
        logf.write(f'Import error: {e}\n')
        print('Missing dependency: ultralytics. Please pip install ultralytics', file=sys.stderr)
        sys.exit(2)
    model_name = 'yolov11n.pt'
    logf.write(f'Using model: {model_name}\n')
    try:
        model = YOLO(model_name)
    except Exception as e:
        logf.write(f'Error loading model {model_name}: {e}\n')
        logf.write('Attempting fallback to copilot_plan/weights/yolo26n.pt\n')
        try:
            model = YOLO('copilot_plan/weights/yolo26n.pt')
            logf.write('Fell back to copilot_plan/weights/yolo26n.pt\n')
        except Exception as e2:
            logf.write(f'Fallback model load failed: {e2}\n')
            print(f'Error loading model {model_name} and fallback failed: {e2}', file=sys.stderr)
            sys.exit(3)
    logf.write('Beginning baseline training (20 epochs)\n')
    start = time.time()
    try:
        model.train(data='data/urpc.yaml', epochs=20, imgsz=640, batch=16, device=0, project=str(OUT), name='baseline', exist_ok=True)
    except Exception as e:
        logf.write(f'Training failed: {e}\n')
        print(f'Training failed: {e}', file=sys.stderr)
        sys.exit(4)
    elapsed = time.time() - start
    logf.write(f'Baseline training completed in {elapsed:.1f}s\n')
print('Baseline training started; logs and outputs in', OUT)
sys.exit(0)
