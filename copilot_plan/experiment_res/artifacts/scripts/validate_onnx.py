#!/usr/bin/env python3
"""Validate ONNX models in copilot_plan/experiment_res/weights/ by
- running onnx.checker.check_model
- running a single inference with onnxruntime
Outputs a JSON summary and per-model logs.
"""
import json
from pathlib import Path
import time

ROOT = Path.cwd()
WD = ROOT / 'copilot_plan' / 'experiment_res' / 'weights'
OUT = WD / 'onnx_validation.json'
logs = {}

try:
    import onnx
    import onnxruntime as ort
    import numpy as np
except Exception as e:
    print('Required packages missing:', e)
    raise

models = sorted(WD.glob('*.onnx'))
if not models:
    print('No .onnx files found in', WD)
    raise SystemExit(0)

for m in models:
    name = m.name
    entry = {'path': str(m), 'onnx_check': False, 'inference': False, 'error': None}
    try:
        model = onnx.load(str(m))
        onnx.checker.check_model(model)
        entry['onnx_check'] = True
    except Exception as e:
        entry['error'] = f'ONNX check failed: {e}'
        logs[name] = entry
        continue
    # prepare random input
    try:
        # prefer CPUExecutionProvider to avoid missing CUDA deps in the environment
        sess = ort.InferenceSession(str(m), providers=['CPUExecutionProvider'])
        inp_meta = sess.get_inputs()[0]
        shape = inp_meta.shape
        # replace dynamic dims with 1 or 640
        in_shape = [1 if (isinstance(d, str) or d is None) else d for d in shape]
        # if any dims still None, set to 1/3/640/640 fallback
        if any(d is None for d in in_shape):
            in_shape = [1,3,640,640]
        # determine expected input element type
        input_type = getattr(inp_meta, 'type', '')
        # create random input and cast as appropriate
        x = np.random.randn(*[int(d) for d in in_shape])
        if 'float16' in str(input_type).lower():
            x = x.astype(np.float16)
        else:
            x = x.astype(np.float32)
        t0 = time.time()
        out = sess.run(None, {inp_meta.name: x})
        t1 = time.time()
        entry['inference'] = True
        entry['runtime_ms'] = (t1 - t0) * 1000.0
        entry['output_shapes'] = [list(o.shape) for o in out]
    except Exception as e:
        entry['error'] = f'Inference failed: {e}'
    logs[name] = entry

with open(OUT, 'w') as f:
    json.dump(logs, f, indent=2)

print('Validation summary written to', OUT)
