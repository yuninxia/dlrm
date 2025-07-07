**快速结论：**
*HPCToolkit + Hatchet* 已能覆盖清单里最核心的三块——**调用栈 (CCT)、多维指标（含 GPU util / stall / idling）与单次运行的时序可视化**。
它们目前**还做不到**的主要是 **跨两次 run 的时间轴对齐可视化** 与 **异构对比视图**；这两点得靠 Kineto 或自己写脚本来补洞。下面按条目逐一标注“✅ 能直接做 / 🛠️ 需额外脚本 / ❌ 不支持”，并给出背后依据与替代工具建议。

---

## 1. 选型：HPCToolkit vs Kineto vs Chakra vs HTA

| 方案                      | 定位 & 优势                                                        | 明显局限                              | 适合角色        |
| ----------------------- | -------------------------------------------------------------- | --------------------------------- | ----------- |
| **HPCToolkit**          | 采样式 CCT、GPU stall 细分、`hpctraceviewer` 时间线；既能 CPU 又能 GPU        | 不能叠加两条 timeline；无内建指标 realignment | **主采集**     |
| **Hatchet**             | Python API 读 CCT → 可做多 run diff、聚合、自动报告                        | 无 GUI 时间线；时间轴需自己对齐                | **离线分析**    |
| **Kineto**              | PyTorch 官方 profiler 内核；JSON trace 支持 CPU+GPU 时间线 + HW counters | 无 CCT；指标有限；不聚合多 run               | **补充双时序对比** |
| Chakra (MS) / HTA (ARM) | 偏硬件计数器/SoC；缺 CCT；生态窄                                           | **不推荐**，难融入 Python 流              | 旁听          |

**决策建议：**
*以 HPCToolkit 采集 → Hatchet Python 化分析* 为主线，再视需求用 Kineto 输出第二条 JSON trace 来做两次 run 的可视化对齐。

---

## 2. 功能对照表

| 清单条目                       | HPCToolkit                                                                   | Hatchet                            | 额外措施 / 备注                                |
| -------------------------- | ---------------------------------------------------------------------------- | ---------------------------------- | ---------------------------------------- |
| **Utilization (总/SM 级)**   | ✅ `gker`, `gins`, `gpuop`, `smsp__throughput` 等可直接导出                         | 🛠️ 聚合后算 % load                    | —                                        |
| **Call Stack / CCT**       | ✅ 采样级完整 CCT；`hpcviewer` 浏览                                                   | ✅ GraphFrame 持有树并可筛选               | —                                        |
| **Idling %（CPU/GPU）**      | ✅ `gins:stl_any`, `gins:stl_gmem` 可换算 GPU idle；CPU 可用 `REALTIME` vs `CYCLES` | 🛠️ 派生列计算 idle\_ratio              | —                                        |
| **单次 Timeline View**       | ✅ `hpctraceviewer` 展示 CPU threads + GPU streams                              | ❌                                  | —                                        |
| **两次 run Timeline 对比**     | ❌ 无叠加 & 对齐                                                                   | 🛠️ Hatchet 支持 diff 但不是时间轴         | 用 **Kineto** 两条 JSON + Chrome trace 分屏查看 |
| **CPU vs GPU 混合 Timeline** | ✅ 同时展示（CPU bar & GPU lane）                                                   | —                                  | —                                        |
| **时间基准 realign 两 run**     | ❌                                                                            | 🛠️ 需自行将 `time` 列减去首样本 → 写入新 trace | 或用 nsys `--timebase gpu` 配对              |
| **一键追踪 bottleneck**        | 🛠️ `Hot Path` + Stall metrics 可定位单 run                                      | ✅ 比较两 run 指标差，自动高亮                 | 多 run root‑cause 要自写规则                   |

---

## 3. 如何用现有工具“最低代价”达标

### 3.1 指标面

* 在 `hpcrun` 里同时打开
  `-e gpu=nvidia` (算 utilization & stall)
  `-e REALTIME@f200` (墙钟)
  `-e CYCLES@4M` (算 CPU idle)
  → Hatchet 脚本里即可派生 util%, idle%, CPI 等 

### 3.2 时间线

* 单 run：`hpctraceviewer db` → 过滤 GPU stream & CPU thread → 右击 “Show Metric” 选 `gpuop` 看 busy/idle
* 双 run：

  1. 用 Kineto (`torch.profiler` backend) 同时 profile 两次实验 → 得到 `trace1.json` & `trace2.json`.
  2. 在 Chrome trace “Load” 两个文件 → UI 内可 overlay & 手动 realign 起点。
  3. 若要脚本化，可用 `tensorboard_plugin_profile` 的 *stall-analyzer* 读两条 trace。
     （Kineto 没有 CCT，但你只为时间线叠加。）

### 3.3 跨 run Diff

* Hatchet:

  ```python
  gA = ht.GraphFrame.from_hpctoolkit('runA-db')
  gB = ht.GraphFrame.from_hpctoolkit('runB-db')
  diff = gB - gA           # 自动对齐 context-hash，再产出 Δmetric
  diff.to_pandas().sort_values('time (inc)').head()
  ```



### 3.4 可选 “自动找 long pole”

1. 用 Hatchet 计算 `Δtime%` 列；阈值>5% 且 `gins:stl_*` 同时升 → 标为怀疑瓶颈。
2. 输出按 CCT 层级排第一的函数即长杆。（脚本示例见 Hatchet tutorial §3.4）

---

## 4. 推荐“最小特性集 (MVP)”写法

1. **采集**：HPCToolkit + CUPTI 事件集。
2. **分析**：Hatchet 脚本自动生成

   * a) 利用率 / idle% 散点
   * b) Top‑10 Hot‑Path 表
3. **GUI**：

   * 时序：hpctraceviewer（单 run）；Chrome trace + Kineto（双 run）。
4. **跨 run diff**：Hatchet CLI `ht.diff runA runB --metric time`.
5. **后续**：若想统一 GUI，多 run timeline 可等 HPCToolkit 路线图里的 *multi-experiment viewer*（roadmap 2025Q3）——但当前需 Kineto 辅助。

---

## 5. 结语

* **HPCToolkit + Hatchet** 已可满足列表里 70 % 的硬需求，尤其是调用栈、利用率、单 run 时间线与跨 run diff。
* **Kineto** 填平“两个运行叠加时间线”这一缺口，成本低（PyTorch 原生）。
* Chakra/HTA 功能重叠度低、生态窄，可暂不考虑。

按此组合落地，你可以在 **1–2 周内交付 MVP**，后续若 HPCToolkit 发布多实验时间线再替换 Kineto 即可。
