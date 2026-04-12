**ONNX 导出与验证详尽报告**

**概览：**
- **目的**：把已导出的 ONNX 模型的完整验证过程、结果、问题与建议整理成可复现的报告，便于部署决策（CPU vs GPU / TensorRT）。
- **生成时间**：2026-04-12
- **仓库路径**：本报告位于 [copilot_plan/experiment_res/validation_report_onnx.md](copilot_plan/experiment_res/validation_report_onnx.md)

**环境与前提**
- 主机：Linux（见会话上下文）
- Python：3.12（工作区使用）
- 框架/工具：Ultralytics YOLO（本地修改版）、PyTorch、ONNX、ONNX Runtime 已安装。
- 注意：当前环境 **ONNX Runtime CUDA (GPU) provider 无法正常加载**（缺少系统库 libcublasLt.so.12），因此 GPU 推理试验在本机不可用，所有成功推理均在 CPU 上运行。

**关键文件（位置）**
- 权重与导出文件目录：[copilot_plan/experiment_res/weights](copilot_plan/experiment_res/weights)
  - ONNX 验证结果：[copilot_plan/experiment_res/weights/onnx_validation.json](copilot_plan/experiment_res/weights/onnx_validation.json)
  - ONNX CPU 基准：[copilot_plan/experiment_res/weights/onnx_bench_cpu_results.json](copilot_plan/experiment_res/weights/onnx_bench_cpu_results.json)
  - ONNX GPU 运行尝试结果（错误）：[copilot_plan/experiment_res/weights/onnx_bench_results.json](copilot_plan/experiment_res/weights/onnx_bench_results.json)
- 实验训练指标汇总：[copilot_plan/experiment_res/experiment_summary.md](copilot_plan/experiment_res/experiment_summary.md)

---

**导出/验证过程要点（按时间顺序）**
1. 将多次训练的 `best.pt`（baseline、WIoU、ECA、WIoU+ECA）集中到 [copilot_plan/experiment_res/weights](copilot_plan/experiment_res/weights)。
2. 使用自写脚本 `copilot_plan/scripts/export_models.py` 将 `*.pt` 导出为 ONNX（优先 FP16 导出）。
3. 初次使用 `validate_onnx.py` 时出现两类问题：
   - ONNX Runtime CUDA provider 无法加载，报错缺少 `libcublasLt.so.12`（系统层面 CUDA/cuBLAS 依赖缺失）。
   - 导出的 ONNX 为 float16 输入模型，验证脚本最初传入 float32，导致推理报错（输入 dtype 不匹配）。
4. 修正措施：
   - 在验证脚本中优先使用 CPU 执行提供器（避免 GPU provider 报错）。
   - 在推理前检查模型输入类型；若模型期望 `float16`，则将随机测试张量强制转换为 `float16`。
5. 重新运行验证，ONNX checker 与 CPU 推理均通过；对每个模型完成 200 次的 CPU 基准测试（得到 p50/p90/p99/mean 与 FPS）。

---

**导出与验证汇总表**

| 模型 | ONNX 路径 | ONNX 大小 (bytes) | ONNX 检查 | 推理(ORT CPU) | 单次均值 ms | CPU FPS | 训练 mAP50 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_200_best | [copilot_plan/experiment_res/weights/baseline_200_best.onnx](copilot_plan/experiment_res/weights/baseline_200_best.onnx) | 4,924,158 | ✅ | ✅ | 35.49 (验证短测) / 28.16 (200-run mean) | 35.51 | 0.77569 ([experiment_summary](copilot_plan/experiment_res/experiment_summary.md)) |
| wiou_eca_ablation_full_best | [copilot_plan/experiment_res/weights/wiou_eca_ablation_full_best.onnx](copilot_plan/experiment_res/weights/wiou_eca_ablation_full_best.onnx) | 4,924,158 | ✅ | ✅ | 35.12 / 28.31 | 35.32 | 0.82126 (eca_repro_full best) / wiou mix: 0.83129 reported best epoch |
| wiou_test_200_best | [copilot_plan/experiment_res/weights/wiou_test_200_best.onnx](copilot_plan/experiment_res/weights/wiou_test_200_best.onnx) | 4,972,011 | ✅ | ✅ | 36.29 / 28.52 | 35.07 | 0.83238 |

说明：表中“单次均值 ms”列同时给出 `onnx_validation.json`（短测的 runtime_ms）与 `onnx_bench_cpu_results.json`（200-run基准 mean_ms），以便参考两类测量。

---

**详细结果（原始 JSON 摘要）**
- ONNX 验证（简要）来自 [copilot_plan/experiment_res/weights/onnx_validation.json](copilot_plan/experiment_res/weights/onnx_validation.json)：

  - `baseline_200_best.onnx`：onnx_check=true, inference=true, runtime_ms=35.494, output_shapes=[1,300,6]
  - `wiou_eca_ablation_full_best.onnx`：onnx_check=true, inference=true, runtime_ms=35.124, output_shapes=[1,300,6]
  - `wiou_test_200_best.onnx`：onnx_check=true, inference=true, runtime_ms=36.285, output_shapes=[1,300,6]

- ONNX CPU 基准（200 runs）来自 [copilot_plan/experiment_res/weights/onnx_bench_cpu_results.json](copilot_plan/experiment_res/weights/onnx_bench_cpu_results.json)：

  - `baseline_200_best`：mean 28.16 ms, p50 27.96 ms, p90 28.55 ms, p99 33.77 ms, FPS 35.51
  - `wiou_eca_ablation_full_best`：mean 28.31 ms, p50 28.03 ms, p90 28.73 ms, p99 33.46 ms, FPS 35.32
  - `wiou_test_200_best`：mean 28.52 ms, p50 28.30 ms, p90 29.19 ms, p99 32.74 ms, FPS 35.07

- ONNX GPU 运行尝试（失败）记录在 [copilot_plan/experiment_res/weights/onnx_bench_results.json](copilot_plan/experiment_res/weights/onnx_bench_results.json)：

  - 报错摘要：ORT CUDA provider 报错“Unexpected input data type. Actual: (tensor(float)), expected: (tensor(float16))”——初始原因是测试数据类型与模型期望不匹配。此外，设备上 CUDA/库不齐全（缺少 libcublasLt.so.12），因此即使修正 dtype，ORT GPU provider 也可能无法加载。

---

**遇到的问题与诊断**
1. ONNX 导出为 float16（FP16）权重/输入：Ultralytics 导出脚本被配置为优先 FP16 导出，这会使 ONNX 模型期望 `float16` 输入；若在 GPU 上运行时，ONNX Runtime 的 CUDA provider 通常可以接受 FP16，但若 provider 未安装或缺少系统依赖会导致加载失败。
2. ONNX Runtime GPU provider 缺失系统库：尝试加载 CUDA provider 时出现 `libcublasLt.so.12` 缺失错误，说明该机器上缺少与当前 ONNX Runtime GPU binary 对应的 CUDA/cuBLAS 运行时库版本；这是系统级依赖问题，需在目标机器上安装相应 CUDA Toolkit / cuBLAS（或使用与系统 CUDA 版本匹配的 onnxruntime 包）。
3. 初始验证脚本输入 dtype 错配：验证脚本最初统一使用 float32 测试张量，导致 FP16 模型推理报 INVALID_ARGUMENT；已在验证脚本中添加对模型输入 dtype 的检测与转换，解决该问题（目前 CPU 推理均成功）。

---

**可复现命令（在工作区根目录运行）**

导出（样例，FP16 优先）
```bash
python3 copilot_plan/scripts/export_models.py \
  --weights copilot_plan/experiment_res/weights/wiou_eca_ablation_full_best.pt \
  --out copilot_plan/experiment_res/weights/wiou_eca_ablation_full_best.onnx \
  --fp16
```

CPU 验证（与脚本已自动化）
```bash
python3 copilot_plan/scripts/validate_onnx.py copilot_plan/experiment_res/weights/*.onnx
```

ONNX CPU 基准（示例，使用已记录脚本/命令）
```bash
python3 copilot_plan/scripts/onnx_benchmark.py --device cpu --runs 200 copilot_plan/experiment_res/weights/*.onnx
```

尝试 GPU（注意：在本机可能失败，示例仅供目标机使用）
```bash
# 确保目标机器安装并可用 CUDA + libcublasLt 对应版本
python3 copilot_plan/scripts/onnx_benchmark.py --device cuda --runs 200 copilot_plan/experiment_res/weights/*.onnx
```

---

**建议与下一步（优先级排序）**
1. 若目标部署为 GPU/TensorRT：在目标机器上（或容器）确保安装匹配的 CUDA / cuBLAS / TensorRT 版本，然后执行：
   - 重新安装与系统 CUDA 对应的 `onnxruntime-gpu`（或使用官方兼容的 wheel）。
   - 使用 TensorRT 将 ONNX 转为 engine（针对 FP16 最优），并在目标 GPU 上跑基准。建议使用 `trtexec` 或 `tensorrt` Python API 做 end-to-end 测试。
2. 若短期内只能使用当前机器（CPU）：使用 `onnx_bench_cpu_results.json` 中的结果作为基线（约 28 ms mean, ~35 FPS 单线程），并在多线程/并发场景下评估吞吐。
3. 可选：如果希望避免 FP16 兼容性问题，可以在导出时强制使用 FP32（降低一点模型大小与单次推理时间，但更通用）。示例：将 `export_models.py` 的导出参数改为 `--fp32` 或 `--no-fp16`。
4. 建议在后续导出中同时保留 FP16 与 FP32 两套 ONNX，以便在不同部署目标间互相对照。

---

**结论（要点）**
- 已成功将三套最佳权重导出为 ONNX 并在 CPU 上通过 checker 与推理验证，完成 200-run 的 CPU 基准测试；结果显示三者在 CPU 单线程下均约为 28 ms mean (~35 FPS)。
- GPU 推理尚不可用于本机，主要受限于系统缺少 CUDA/cuBLAS 库（libcublasLt.so.12），以及导出为 FP16 导致的输入 dtype 要求（可在验证脚本中兼容）。
- 推荐在目标 GPU 环境（或容器）上完成 TensorRT 转换与 GPU 基准，以获取部署级别的 FPS。若需要，我可以继续：执行 TensorRT 转换脚本（在有 GPU 和正确系统依赖的机器上），或生成 FP32 版本的 ONNX 并再跑一次基准。

---

如需我把报告合并进 README 或生成一个对比表格（CSV）便于后续绘图，我可以继续处理。是否现在生成附加的 CSV 对比表并提交为 `copilot_plan/experiment_res/weights/onnx_validation_table.csv`？
