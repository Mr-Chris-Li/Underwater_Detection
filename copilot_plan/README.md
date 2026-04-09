# Copilot Plan — Smoke Test

此目录包含用于项目初期的冒烟测试脚本，帮助在目标服务器上快速验证环境与 `yolo` CLI/库的可用性。

Files:
- `smoke_test.py`: 环境与样例图片检查，生成 `yolo predict` 建议命令。

Usage:

```bash
python3 copilot_plan/smoke_test.py
```

下一步建议：在 A10 服务器上运行上面的脚本；若 `yolo` CLI 可用且模型存在，运行生成的 `yolo predict` 命令以完成冒烟测试。
