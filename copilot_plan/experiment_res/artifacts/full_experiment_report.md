**Underwater_Detection — 全量实验数据与参数汇总**

生成时间：2026-04-12

说明：本报告聚合训练关键参数、日志与导出验证结果，便于审阅与归档。报告引用的原始文件位于 `copilot_plan/experiment_res/` 下；压缩包 `Underwater_Detection_export.zip` 已生成（不包含数据集）。

**一、重要路径与文件（工作区相对）**
- 实验汇总：[copilot_plan/experiment_res/experiment_summary.md](copilot_plan/experiment_res/experiment_summary.md)
- ONNX 验证报告（详尽）：[copilot_plan/experiment_res/validation_report_onnx.md](copilot_plan/experiment_res/validation_report_onnx.md)
- ONNX 验证原始 JSON：[copilot_plan/experiment_res/weights/onnx_validation.json](copilot_plan/experiment_res/weights/onnx_validation.json)
- ONNX CPU 基准：[copilot_plan/experiment_res/weights/onnx_bench_cpu_results.json](copilot_plan/experiment_res/weights/onnx_bench_cpu_results.json)
- ONNX GPU 尝试日志（错误）：[copilot_plan/experiment_res/weights/onnx_bench_results.json](copilot_plan/experiment_res/weights/onnx_bench_results.json)
- 中心化权重目录：[copilot_plan/experiment_res/weights](copilot_plan/experiment_res/weights)

**二、训练参数与关键配置（摘录）**
- 模型配置示例（在启动训练时使用的模型文件）：copilot_plan/models/yolo11_eca.yaml （请在仓库中查看以获取完整超参数与网络结构）。
- 典型训练启动命令（我用于复现实验的命令示例）：
```
python3 copilot_plan/monitor_and_train.py \
  --model copilot_plan/models/yolo11_eca.yaml \
  --name wiou_eca_ablation_full \
  --epochs 200 --imgsz 640 --batch 32 --workers 8 --gpu 0 --patience 10 --seed 0 --wiou 3
```
- 主要超参：`epochs`, `imgsz`, `batch`, `workers`, `seed`, `wiou` 值见上列命令或模型 yaml。

**三、训练性能摘要（摘自 experiment_summary）**

| Experiment | Epochs (best) | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| baseline | 20 | 0.79341 | 0.70073 | 0.78734 | 0.45959 |
| baseline_100_ep_pat10 | 86 | 0.81830 | 0.76095 | 0.82867 | 0.49739 |
| baseline_200 | 29 | 0.78308 | 0.69924 | 0.77569 | 0.43934 |
| wiou_test_200 | 102 | 0.82814 | 0.76326 | 0.83238 | 0.50303 |
| eca_repro_full | 69 | 0.81961 | 0.75188 | 0.82126 | 0.48816 |

（完整表见 [experiment_summary.md](copilot_plan/experiment_res/experiment_summary.md)）

**四、导出与推理验证关键结果（摘要）**
- 导出模型（ONNX）文件及大小：
  - copilot_plan/experiment_res/weights/baseline_200_best.onnx — 4,924,158 bytes
  - copilot_plan/experiment_res/weights/wiou_eca_ablation_full_best.onnx — 4,924,158 bytes
  - copilot_plan/experiment_res/weights/wiou_test_200_best.onnx — 4,972,011 bytes
- ONNX 检查与 CPU 推理：均通过。短测 runtime_ms ~35 ms；200-run mean ~28 ms（详见 onnx_bench_cpu_results.json）。

**五、训练日志与曲线数据**
- Ultralytics runs 目录含完整训练日志、`metrics.csv`（loss/mAP 曲线数据）和 `events` 文件：请在 `ultralytics/runs/detect/` 下对应 run 文件夹中查看并下载需要的 `metrics.csv` 用于绘图。
- 示例路径（替换为实际 run 名）：`ultralytics/runs/detect/copilot_plan/train_outputs/<run>/metrics.csv`。

**六、压缩包与校验信息**
- 已生成压缩包（仓库根）：`Underwater_Detection_export.zip`（已排除 `data/` 与常见 `datasets/` 路径）。
- 压缩包 SHA256 校验：
```
71b35ded8b6f48d8d6eb1ce50b598b778daeddce90011174c7755c28eee3ea1b  Underwater_Detection_export.zip
```

**七、下步建议（可按需执行）**
1. 若要部署到 GPU/TensorRT：在目标机器上安装匹配的 CUDA/cuBLAS，并在该环境中运行 ONNX -> TensorRT 转换与基准。
2. 导出额外的 FP32 版本 ONNX 以提高兼容性（不影响原始权重），并对比 FP16/FP32 的推理性能。
3. 从 `ultralytics/runs/detect/.../metrics.csv` 导出损失与 mAP 曲线为 PNG/CSV，以便报告中嵌入可视化图表（我可以代劳）。

---

如需我现在：
- (A) 把 `metrics.csv` 中的训练损失与 mAP 曲线绘图并把图像写入 `copilot_plan/experiment_res/plots/`；
- (B) 生成 CSV 对比表 `copilot_plan/experiment_res/weights/onnx_validation_table.csv`；
- (C) 将 `Underwater_Detection_export.zip` 上传到某个临时共享（需提供目标上传位置或授权）。

请告诉我要执行哪项，我会继续。
