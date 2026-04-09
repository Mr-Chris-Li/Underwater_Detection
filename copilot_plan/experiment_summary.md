# 实验进展与改进目标 — YOLOv11 水下目标检测

## 概要

本报告汇总了在当前工作区针对 URPC 数据集基于 Ultralytics/YOLO 的一系列“冒烟”验证、短训与改进性实验进展，并列出下一步的改进方向与目标。

## 已完成的测试与训练（摘要）

- 冒烟推理（10 张图片）：生成文件 `copilot_plan/smoke_outputs/smoke_summary.json`。单张图片推理时间范围约 20–617 ms，单图检测数多为 0–2。
- 短训验证（1 epoch）：使用 `copilot_plan/run_smoke_train.py` 运行以验证训练流程；在缺失 `yolov11n.pt` 时回退到 `yolo26n.pt`，短训成功并产生运行输出。
- 基线训练（20 epochs）：运行脚本 `copilot_plan/run_baseline_train.py`，输出目录为 `/ultralytics/runs/detect/copilot_plan/train_outputs/baseline/`；观测到基线 best mAP50 ≈ 0.607，mAP50-95 ≈ 0.313（短训/日志摘录）。
- ECA 注意力模块验证（10 epochs）：已在代码中集成 ECA 并用 `copilot_plan/models/yolo11_eca.yaml` 进行短训，`eca_test` 运行结果与权重已保存。
- WIoU v3 验证（10 epochs）：运行 `copilot_plan/run_wiou_train.py`，`wiou_test` 最终指标（epoch 10）：Precision ≈ 0.720，Recall ≈ 0.592，mAP50 ≈ 0.657，mAP50-95 ≈ 0.352；权重已复制到 `copilot_plan/weights/wiou_best.pt` 和 `copilot_plan/weights/wiou_last.pt`。

## 关键产物位置

- 冒烟推理摘要： `copilot_plan/smoke_outputs/smoke_summary.json`
- 训练运行目录（所有 runs）： `/ultralytics/runs/detect/copilot_plan/train_outputs/`（包含 `baseline/`, `eca_test/`, `wiou_test/` 等子目录）
- 快照权重与备份： `copilot_plan/weights/`（包含 baseline/wiou 的 best/last 等）

## 已解决的问题与待处理问题

- 已解决：
  - 本地缺失 `yolov11n.pt`（移除零字节文件并使用 `yolo26n.pt` 回退）。
  - `data/urpc.yaml` 的路径不匹配问题已改为绝对路径以确保训练能读到图像/标签。
- 待处理：
  - 检测到 122 个空标签文件（已统计，尚未清理或修复）。

## 改进方向与工作目标（建议优先级）

1. 数据清理（优先）
   - 修复或移除 122 个空标签文件，确保训练/验证集标签一致性，避免对评估造成偏差。
2. 架构对比与优化
   - 实现并比较 FPN / PANet 变体（在短训 10–20 epochs 上快速验证），与当前 baseline、ECA、WIoU 结果对比。
3. 消融实验
   - 系统比较：Baseline vs ECA vs WIoU；同时对比 IoU 损失（WIoU vs CIoU）与注意力模块的影响。记录 `results.csv` 并绘制对比曲线。
4. 稳定训练对照
   - 对候选最优方案扩展训练至 50–100 epochs，以获得稳定、可比较的最终指标。
5. 导出与基准
   - 导出最佳权重为 ONNX / TorchScript / TFLite（按需），并建立推理基准脚本测量速度与内存占用。
6. 报告自动化
   - 编写脚本聚合各 run 的 `results.csv`，自动生成对比图表与表格（PNG/Markdown 报告），便于复现与分享。

## 建议的下一步（可选）

- 若优先级为“准确比较”：先执行“数据清理（第1项）”，再并行跑 FPN/PANet 短训（第2项）。
- 若优先级为“快速检验架构变化”：在不清理标签的情况下先做 FPN/PANet 短训以快速得到方向性信号，但最终对比仍建议在清理后复现。

---

报告文件位置： `copilot_plan/experiment_summary.md`

如需我现在执行“清理空标签”或开始“FPN/PANet 短训”，回复我优先级或直接下令 “开始清理” / “开始 FPN 短训”。
