from pathlib import Path
import sys, time

OUT = Path('copilot_plan/train_outputs')
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / 'eca_run.log'

with open(LOG, 'w') as logf:
    logf.write('Starting ECA short training (10 epochs)\n')
    try:
        from ultralytics import YOLO
    except Exception as e:
        logf.write(f'Import error: {e}\n')
        print('Missing dependency: ultralytics. Please pip install ultralytics', file=sys.stderr)
        sys.exit(2)
    model_cfg = 'copilot_plan/models/yolo11_eca.yaml'
    logf.write(f'Using model cfg: {model_cfg}\n')
    try:
        model = YOLO(model_cfg)
    except Exception as e:
        logf.write(f'Error creating model from cfg {model_cfg}: {e}\n')
        # fallback to yolov11n or yolo26n
        try:
            model = YOLO('copilot_plan/weights/baseline_best.pt')
            logf.write('Fell back to copilot_plan/weights/baseline_best.pt\n')
        except Exception as e2:
            logf.write(f'Fallback failed: {e2}\n')
            print('Failed to instantiate model from cfg and fallback.', file=sys.stderr)
            sys.exit(3)
    logf.write('Beginning ECA training (10 epochs)\n')
    start = time.time()
    try:
        model.train(data='data/urpc.yaml', epochs=10, imgsz=384, batch=8, device=0, project=str(OUT), name='eca_test', exist_ok=True)
    except Exception as e:
        logf.write(f'Training failed: {e}\n')
        print(f'Training failed: {e}', file=sys.stderr)
        sys.exit(4)
    elapsed = time.time() - start
    logf.write(f'ECA training completed in {elapsed:.1f}s\n')
print('ECA short training started; logs and outputs in', OUT)
sys.exit(0)
