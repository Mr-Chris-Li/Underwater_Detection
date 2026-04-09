#!/usr/bin/env python3
"""冒烟测试脚本：检查环境、列出样例图片并打印建议的 `yolo` 推理命令。

用法：
    python3 copilot_plan/smoke_test.py

该脚本不会直接执行 `yolo predict`，仅做检查并生成命令，便于手动或自动运行。
"""
import os
import shutil
import sys
from pathlib import Path

def find_images(root_dirs, max_images=10):
    imgs = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    for d in root_dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.rglob('*'):
            if f.suffix.lower() in exts:
                imgs.append(str(f))
                if len(imgs) >= max_images:
                    return imgs
    return imgs

def main():
    print('=== Smoke Test: Environment Check ===')
    try:
        import torch
        print('torch available:', torch.__version__)
        print('cuda available:', torch.cuda.is_available())
    except Exception as e:
        print('torch import failed:', e)

    yolo_cli = shutil.which('yolo')
    print('yolo CLI found at:', yolo_cli)

    # candidate image folders (project-local)
    candidates = [
        'data/URPC2021/images/train',
        'data/URPC2021/images/val',
        'data/RUIE_Dataset/RUIE/UTTS/pic_A',
        'data/RUIE_Dataset/RUIE/UTTS/pic_B',
    ]
    imgs = find_images(candidates, max_images=10)
    if imgs:
        print('\nFound sample images (up to 10):')
        for i, p in enumerate(imgs, 1):
            print(f'  {i}. {p}')
    else:
        print('\nNo sample images found in expected locations. Place images under data/... and retry.')

    print('\nSuggested `yolo` predict command (edit model path as needed):')
    src = ' '.join(imgs) if imgs else 'data/your_images'
    print(f"yolo predict model=PATH/TO/yolov11n.pt source={src} device=0 conf=0.25 hide_labels=False save=True")

    print('\nIf you prefer Python API, example:')
    print("python3 -c \"from ultralytics import YOLO; m=YOLO('PATH/TO/yolov11n.pt'); m.predict(source='data/your_images', device=0, conf=0.25)\"")

    print('\nNext steps:')
    print('- Ensure model file (yolov11n.pt) is available at given path or use a downloaded checkpoint.')
    print('- Run the `yolo predict ...` command to verify inference pipeline.')

if __name__ == '__main__':
    main()
