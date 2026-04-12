# 实验模型汇总

下面表格列出从 `copilot_plan/experiment/*/gpu_util_report.json` 中提取的主要字段：`name`、`model`、`epochs`、`imgsz`、`batch`。

| name | model | epochs | imgsz | batch |
|---|---|---:|---:|---:|
| eca_repro_short | copilot_plan/weights/yolo26n.pt | 50 | 640 | 32 |
| eca_seed1_short | copilot_plan/models/yolo11_eca.yaml | 20 | 384 | 8 |
| eca_test_200 | copilot_plan/weights/yolo26n.pt | 200 | 640 | 32 |
| eca_test_diag | copilot_plan/models/yolo11_eca.yaml | 20 | 384 | 8 |
| gpu_tune_trial1 | copilot_plan/weights/yolo26n.pt | 1 | 640 | 32 |
| gpu_tune_trial2 | copilot_plan/weights/yolo26n.pt | 1 | 640 | 32 |
| gpu_tune_trial3 | copilot_plan/weights/yolo26n.pt | 1 | 640 | 64 |
| gpu_tune_trial4 | copilot_plan/weights/yolo26n.pt | 1 | 1024 | 32 |
| gpu_tune_trial7 | copilot_plan/weights/yolo26n.pt | 1 | 1024 | 40 |
| gpu_tune_trial8 | copilot_plan/weights/yolo26n.pt | 1 | 1024 | 32 |
| wiou_test_200 | copilot_plan/weights/yolo26n.pt | 200 | 640 | 32 |

注：部分子目录中未找到 `gpu_util_report.json`（可能尚未运行或未保存报告），因此未列入表格。
