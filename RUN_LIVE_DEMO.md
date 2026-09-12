# BigSmall 摄像头与情感状态 Demo

这版 Demo 同时显示 BigSmall 的 rPPG、呼吸、AU 输出，以及一个实验性的二维情感状态估计：

- Valence（效价）：主要由 AU 面部证据估计，范围为 -1 到 +1。
- Arousal（唤醒度）：融合 AU、HR、HR 趋势、呼吸率和 RMSSD，范围为 0 到 1。
- AU 证据是多标签输出，微笑、紧张和皱眉等不会再互相覆盖。

它只能表达“当前证据更接近哪种状态”，不是经过情绪数据集监督训练的正式情绪分类器，也不能用于医疗诊断。

## 摄像头运行

```powershell
conda run -n bigsmall-rppg python demo_bigsmall.py --source camera --camera-id 0
```

启动后：

1. 前 5 秒保持自然、中性的表情，用于建立个人 AU 基线。
2. 出现 `Face baseline ready - you may smile` 后再微笑。
3. 完整生理基线约需 20 秒；HR 约 8 秒后出现，呼吸约 15 秒，HRV 至少需要 30 秒视频。
4. 按 `Q` 或 `Esc` 退出。

微笑调试时观察两处：

- `Smile (AU06+12)` 是经过个人基线校准的微笑证据。
- `AU06/12` 显示 BigSmall 的原始 AU06（脸颊提升）和 AU12（嘴角提升）概率。

HR 已恢复为第一版的 BigSmall 单通道计算，暂不运行 POS 或融合算法。`Signal FPS` 用于显示实际进入 BigSmall 的有效帧率。

人脸跟踪已改成 MediaPipe Face Landmarker。绿色框来自完整面部 landmark；短时丢失时会显示橙色 `TRACKING WEAK` 并沿用最近位置，持续丢失约 1 秒后才显示 `NO FACE`。摄像头读取线程只保留最新帧，避免推理较慢时累积历史画面。

如果摄像头打开后是黑屏，先关闭 Windows 相机、会议软件等可能占用设备的程序，并检查隐私挡板及“设置 → 隐私和安全性 → 相机 → 允许桌面应用访问相机”。也可手动切换后端：

```powershell
conda run -n bigsmall-rppg python demo_bigsmall.py --source camera --camera-backend dshow
conda run -n bigsmall-rppg python demo_bigsmall.py --source camera --camera-backend msmf
```

## 用 UBFC-rPPG 小样本回归

快速无窗口测试：

```powershell
conda run -n bigsmall-rppg python demo_bigsmall.py --source clips --no-realtime --headless --max-frames 300
```

完整回放并录制结果：

```powershell
conda run -n bigsmall-rppg python demo_bigsmall.py --source clips --no-realtime --headless --record
```

UBFC 子集只用于验证推理、生理计算、UI 和导出链路，不能验证真人主动微笑的交互效果。

## 输入普通视频

```powershell
conda run -n bigsmall-rppg python demo_bigsmall.py --source video --input D:\video\face.avi
```

## 输出

默认写入 `runs/bigsmall_demo`：

- `session.csv`：BigSmall HR、有效采样率、人脸跟踪状态、运动量、呼吸率、RMSSD、SDNN、SNR、Valence/Arousal、状态、置信度、六类证据及 12 个原始 AU。
- `last_frame.jpg`：最后一帧界面截图。
- `preview.mp4`：仅在传入 `--record` 时生成。

当前 HRV 是由摄像头脉搏波峰间期计算的实验性 PRV。低信号质量时不要把 HRV 或情感状态当作可靠结论。
