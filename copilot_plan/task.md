### 第一阶段：环境审计与“冒烟测试”（Smoke Test）

**目标**：确保开发链路闭环，排除环境与框架兼容性风险。

1. **依赖深度自检**：
   - 验证 `PyTorch 2.x` 与 `CUDA 12.x` 的协同（利用 `torch.cuda.is_available()`）。
   - 克隆并安装 `Ultralytics` 框架，确认 `yolo` 命令在终端可唤起。
2. **小规模可行性验证（10张图片测试）**：
   - **任务**：从 URPC 数据集抽取 10 张具有典型水下特征（蓝绿偏色、浑浊）的图像。
   - **执行**：使用官方 YOLOv11n 权重进行推理，观察预测框分布及置信度。
   - **产出**：确认推理管道通畅，建立对原始图像质量的直观感知。

------

### 第二阶段：数据工程与基准线（Baseline）建立

**目标**：量化原始性能，准备高质量的“实验土壤”。

1. **数据集标准化**：
   - 将 URPC/RUIE 转换为 YOLO 格式（`.yaml` 配置及路径映射）。
   - 编写数据检查脚本，核对是否存在空的 `label` 文件或坐标越界。
2. **基准训练（Benchmark）**：
   - 在不加任何改进的情况下，运行 YOLOv11n 完整训练。
   - **记录**：保存 `results.csv` 和 PR 曲线，作为后续所有消融实验的参照物。

------

### 第三阶段：核心算法改进（模块化开发）

**目标**：按照规划书进行模型重构。建议按以下顺序逐一突破：

1. **注意力机制集成（ECA/CA）**：
   - 在 `ultralytics/nn/modules` 中注册自定义类。
   - 修改 `yolov11-underwater.yaml`，在 Neck 部分嵌入注意力模块。
   - **验证**：进行一次 10-20 epoch 的短训练，观察特征图（Heatmap）是否更聚焦于目标区域。
2. **损失函数替换（WIoU v3）**：
   - 在损失计算源代码中引入 `WIoU` 逻辑。
   - 重点观察边界框收敛速度（Box Loss）是否优于官方默认的 `IoU/CIoU`。
3. **特征金字塔（FPN/PANet）优化**：
   - 针对小目标（如海胆、扇贝），调整特征融合层级，增加浅层特征的传递权重。

------

### 第四阶段：消融实验与稳健性评估

**目标**：用数据支撑论文逻辑。

1. **消融实验矩阵**：
   - Exp A: Baseline
   - Exp B: Baseline + ECA
   - Exp C: Baseline + ECA + WIoU v3
   - Exp D: Baseline + ECA + WIoU v3 + 增强策略
2. **跨域泛化测试**：
   - 直接使用在 URPC 上训练好的权重，在 RUIE 不同浊度子集上进行测试。
   - **产出**：性能波动热力图，分析模型在哪些水质下表现最差。

------

### 第五阶段：部署转换与性能封测

**目标**：验证“实时性”指标。

1. **模型压缩与导出**：
   - 执行 `model.export(format='engine', half=True)` 导出为 TensorRT 格式。
2. **推理速度分析**：
   - 编写脚本统计 100 帧图像的平均推理延迟。
   - 对比 FP32 与 FP16 在 RTX 显卡上的性能增益（FPS）。

------

### 实验结果存档规则

- 所有自动或手动生成的实验报告/评估文件应集中存放于 `copilot_plan/experiment_res/`。
- 命名格式：`experiment_YYYYMMDDhhmm.md`，其中时间使用生成时间（24 小时制），例如 `experiment_202604092329.md`。
- 若同一分钟内有多份报告，请在文件名后添加后缀 `_v2`、`_v3` 等以示区分，例如 `experiment_202604092329_v2.md`。
- 所有相关脚本应在生成报告后将报告移动并重命名到该目录，且在 `copilot_plan/task.md` 中记录此次变更（如本次所示）。

------

### 自动化后续任务（由 Copilot 触发）

A. 规范化并聚合 `results.csv`：扫描所有 run 的 `results.csv`，统一列名（例如把 mAP@0.5 标准化为 `mAP50`），并在 `copilot_plan/experiment_res/csv/` 下生成每个 run 的 `_mAP50.csv` 与 `_loss.csv`，最后重新生成比较图 `mAP50_comparison.png` 与 `loss_comparison.png`。

B. 为每个 run 生成索引并同步关键文件：在 `copilot_plan/experiment/` 下为每个 run 建立子目录（如 `copilot_plan/experiment/eca_test_200/`），并将该 run 的 `results.csv`、`gpu_util_report.json`、`weights/best.pt`（如存在）复制或索引到该目录，同时生成 `index.json` 包含 run 名、best_epoch、best_mAP、路径等元信息。

C. 复现 `eca_test` 在 200 epoch 预算下：使用现有 `monitor_and_train.py` 启动 `eca_test_200`（参数 `--epochs 200 --imgsz 640 --batch 32 --workers 8 --gpu 0 --name eca_test_200 --patience 10`），并确保训练过程产生的 `results.csv` 与 `gpu_util_report.json` 被纳入上面的索引同步流程。
