# BigSmall on Windows / RTX 50 Series

## 已完成的快速验证

在仓库根目录运行：

```powershell
conda run -n bigsmall-rppg python tools/run_bigsmall_smoke.py
```

脚本会在 CUDA 上依次加载官方 Fold 1/2/3 权重，并检查三个任务头：12 个面部动作单元、BVP 和呼吸。

## 从零重建环境

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_bigsmall_windows.ps1
```

RTX 50 系显卡需要包含 CUDA 12.8 kernel 的 PyTorch。仓库原始 `setup.sh` 固定的 PyTorch 2.1/CUDA 12.1 不适用于本机 GPU；Mamba 扩展与 BigSmall 无关，因此 Windows 环境不安装它。

## 使用 BP4D+ 做正式论文评测

BigSmall 的官方评测数据是受许可约束的 BP4D+，仓库与预训练权重不包含原始数据。获得 BP4D+ 后：

1. 把原始数据放到 `data/BP4DPlus/RawData`（目录可自行更改）。
2. Fold 1 配置已经改成本机相对路径、单 GPU、正式 Fold 1 权重和 `only_test` 模式。
3. 如果还没有预处理缓存，先把 TRAIN/VALID/TEST 三处 `DO_PREPROCESS` 设为 `True` 运行一次；预处理完成后改回 `False`。
4. 执行：

```powershell
conda run -n bigsmall-rppg python main.py --config_file configs/train_configs/BP4D_BP4D_BIGSMALL_FOLD1.yaml
```

Fold 2/3 使用同目录下对应 YAML 和权重。论文的完整三折指标需要 BP4D+ 数据；没有该数据时，快速验证脚本已覆盖环境、CUDA、模型结构和正式权重的完整推理链路。
