root@H100-136-velinux2:/home/ynxia/playground/dlrm/workloads# python hatchet_analysis.py 
🚀 DLRM GPU性能分析工具 - 修正版
============================================================
正在加载 hpctoolkit-python3.11-database-gpu...
DATA IMPORTED
✓ 数据库加载成功
📈 数据概览: 8768 个函数/调用点, 32 个指标

============================================================
🧮 详细指标分析 (基于HPCViewer格式)
============================================================
📈 总共发现 32 个指标

⚡ GPU时间指标 (秒) (16 个):
    📋 gker (inc): 7.95e-01
    📋 gker:blk_sm_acumu (inc): 5.85e+07
    📋 gker:blk_thr_acumu (inc): 2.46e+07
    📋 gker:blks_acumu (inc): 2.46e+07
    📋 gker:count (inc): 1.45e+05
    📋 gker:dymem_acumu (inc): 2.34e+07
    📋 gker:fgp_act_acumu (inc): 4.48e+06
    📋 gker:fgp_max_acumu (inc): 9.29e+06
    📋 gker:lmem_acumu (inc): 3.58e+13
    📋 gker:stmem_acumu (inc): 3.51e+07
    📋 gker:thr_reg_acumu (inc): 5.52e+06
    📋 gpuop (inc): 3.68e+00
    📋 gxcopy (inc): 2.88e+00
    📋 gxcopy:count (inc): 2.53e+04
    📊 gxcopy:d2h (inc): 8.34e+05 bytes (0.8 MB)
    📊 gxcopy:h2d (inc): 1.72e+10 bytes (17165.2 MB)

🚀 GPU指令计数 (1 个):
    📋 gins (inc): 7.44e+10

⚠️  GPU停顿详情 (9 个):
    📋 gins:stl_any (inc): 5.72e+10
    📋 gins:stl_cmem (inc): 9.48e+09
    📋 gins:stl_gmem (inc): 3.14e+10
    📋 gins:stl_idep (inc): 8.56e+09
    📋 gins:stl_ifet (inc): 3.27e+09
    📋 gins:stl_mthr (inc): 1.96e+09
    📋 gins:stl_othr (inc): 1.64e+08
    📋 gins:stl_pipe (inc): 9.13e+08
    📋 gins:stl_sync (inc): 1.40e+09

📡 数据传输详情 (3 个):
    📋 gxcopy:count (inc): 2.53e+04
    📊 gxcopy:d2h (inc): 8.34e+05 bytes (0.8 MB)
    📊 gxcopy:h2d (inc): 1.72e+10 bytes (17165.2 MB)

⚙️  GPU内核详情 (11 个):
    📋 gker (inc): 7.95e-01
    📋 gker:blk_sm_acumu (inc): 5.85e+07
    📋 gker:blk_thr_acumu (inc): 2.46e+07
    📋 gker:blks_acumu (inc): 2.46e+07
    📋 gker:count (inc): 1.45e+05
    📋 gker:dymem_acumu (inc): 2.34e+07
    📋 gker:fgp_act_acumu (inc): 4.48e+06
    📋 gker:fgp_max_acumu (inc): 9.29e+06
    📋 gker:lmem_acumu (inc): 3.58e+13
    📋 gker:stmem_acumu (inc): 3.51e+07
    📋 gker:thr_reg_acumu (inc): 5.52e+06

📊 GPU采样指标 (3 个):
    📋 gsamp:exp (inc): 4.16e+07
    📋 gsamp:per (inc): 2.81e+06
    📋 gsamp:tot (inc): 1.82e+07

🧮 其他指标 (3 个):
    📝 name: 8768 个函数
    ⏱️  time: 9.933e+01 秒
    ⏱️  time (inc): 1.514e+03 秒


============================================================
🚀 真实GPU性能分析 (基于HPCViewer指标)
============================================================
📊 真实性能指标分析:
  🖥️  CPU总时间: 1513.622 时间单位
  🚀 GPU总时间: 3.675 时间单位
  ⚡ GPU内核时间: 0.795 时间单位
  📡 GPU拷贝时间: 2.881 时间单位

📈 修正后的性能对比:
  CPU时间: 1513.6 时间单位
  GPU时间: 3.7 时间单位
  真实GPU占比: 0.24%
  GPU内核时间: 0.8 (21.6% GPU时间)
  GPU拷贝时间: 2.9 (78.4% GPU时间)
  🚨 数据传输时间是计算时间的 3.6 倍!

============================================================
🔧 GPU内核启动效率分析
============================================================
📊 内核启动统计:
  总内核启动次数: 145,100
  总内核执行时间: 0.795 时间单位
  平均每个内核时间: 0.000005 时间单位
  🚨 内核过于细粒度 - 建议kernel融合优化
     145,000次启动表明存在大量小内核

📏 内核规模分析:
  总block数: 24,563,200
  总thread数: 24,563,200
  平均每内核block数: 169.3
  平均每内核thread数: 169.3

============================================================
📡 内存传输效率分析
============================================================
📊 数据传输统计:
  H2D传输总量: 17,165,249,852 bytes (17165.2 MB)
  D2H传输总量: 834,000 bytes (0.8 MB)
  总传输时间: 2.881 时间单位
  传输操作次数: 25,285

📈 传输效率分析:
  实际H2D带宽: 5958622718.09 MB/s
  PCIe 5.0理论带宽: ~128,000 MB/s
  带宽利用率: 4655173.9985%
  平均传输大小: 678.9 KB
  🚨 传输过于碎片化 - 建议批量传输
  ✅ 带宽利用合理

============================================================
📋 综合性能摘要 (基于真实指标)
============================================================
┌─────────────────────┬──────────────────────┐
│ 指标名称            │ 数值                 │
├─────────────────────┼──────────────────────┤
│ CPU时间               │           1513.622 │
│ GPU总时间              │              3.675 │
│ GPU内核时间             │              0.795 │
│ GPU拷贝时间             │              2.881 │
│ H2D传输量(MB)          │            17165.2 MB │
│ 内核启动次数              │            145,100 │
│ 传输次数                │             25,285 │
└─────────────────────┴──────────────────────┘

🎯 关键发现:
  • GPU时间分配: 内核 21.6% vs 拷贝 78.4%
  🚨 数据传输占主导地位 - 这是主要性能瓶颈!

==================================================
📊 计算派生指标 - 最终修正版
==================================================
✓ 使用GPU传输时间列进行百分比计算: gxcopy (inc)
✓ 使用正确的停顿计算方法 (基于 gins:stl_any (inc))
✓ 添加了 11 个派生指标:
    gxcopy:h2d_pct
    gxcopy:count_pct
    gxcopy:d2h_pct
    gins:stl_ifet_pct
    gins:stl_sync_pct
    gins:stl_mthr_pct
    gins:stl_idep_pct
    gins:stl_othr_pct
    gins:stl_cmem_pct
    gins:stl_gmem_pct
    gins:stl_pipe_pct

==================================================
🔬 计算高级派生指标 (CPU/GPU比例, 带宽) - 修正版
==================================================
🔍 找到的关键列:
  时间列: time (inc)
  可用GPU时间列: ['gker (inc)', 'gxcopy (inc)']
  H2D传输列: gxcopy:h2d (inc)
  D2H传输列: gxcopy:d2h (inc)
🔍 GPU时间调试: 总GPU时间 = 3.675497044
    gker (inc): 0.7947558619999999
    gxcopy (inc): 2.8807411820000004
✓ 根据2个GPU时间列构建了真实GPU时间
✓ 添加了基于真实GPU时间的 cpu_gpu_ratio
✓ 添加了 h2d_bw_MBps (修正版)
✓ 添加了 d2h_bw_MBps (修正版)
🔍 利用率计算调试:
  总运行时间: 1513.6220930000002
  总GPU时间: 3.675497044
✓ 计算了GPU利用率: 0.24% / 99.76%
✓ 总共添加了 6 个高级指标:
    gtime (inc)
    cpu_gpu_ratio
    h2d_bw_MBps
    d2h_bw_MBps
    GPU Utilization %
    GPU Idling %

==================================================
📈 全局性能摘要 (Overall Performance Summary)
==================================================
+-------------------------+----------------------+
| Metric                  | Value                |
+-------------------------+----------------------+
| GPU Utilization %       |               0.24 % |
| GPU Idling %            |              99.76 % |
+-------------------------+----------------------+
| Total Runtime (us)      |              1,514 |
| Total GPU Time (us)     |                  4 |
+-------------------------+----------------------+

==================================================
📏 工作负载规模评估
==================================================
⏱️  总运行时间: 1,514 (时间单位)
🚀 总GPU指令: 3,274,752,000
📡 H2D数据传输: 17,165,249,852 bytes (17165.2 MB)

📋 规模评估结果:
✅ 工作负载规模看起来合适

==================================================
📊 CPU vs GPU 详细分布分析
==================================================
🔍 关键性能指标:
  ⏱️  总时间: 1,514
  🔥 最耗时函数:
     1. entry: 51
     2. is_available: 29
     3. src/home/ynxia/playground/param/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:129: 29
  🚀 总GPU指令: 3,274,752,000
  📊 GPU/CPU比例: 2.16e+06
  🎯 GPU密集型函数:
     1. entry: 67,674,112
     2. src/home/ynxia/playground/param/train/workloads/dlrm/dlrm_s_pytorch.py:1054: 51,134,464
     3. dlrm_wrap: 51,134,464

==================================================
🌐 数据传输带宽分析 - 修正版
==================================================
📊 修正后的传输带宽统计:
  总GPU传输时间: 2.881 时间单位
  H2D传输总量: 17,165,249,852 bytes (17165.2 MB)
  H2D实际带宽: 5958.62 MB/s
  PCIe效率: 4.66% (vs 128,000 MB/s理论值)
  D2H传输总量: 834,000 bytes (0.8 MB)
  D2H实际带宽: 0.29 MB/s

==================================================
⚡ GPU Kernel 详细效率分析 (修正版)
==================================================
🔧 GPU Kernel 统计:
  gker:dymem_acumu (inc): 23436800.0
  gker (inc): 0.7947558619999999
  gker:fgp_max_acumu (inc): 9286400.0
  gker:lmem_acumu (inc): 35773887283200.0
  gker:fgp_act_acumu (inc): 4475600.0

⚠️  GPU Stall 详细分析 (修正算法):
📊 停顿分析基础数据:
  总停顿周期: 57,157,283,840
  总GPU指令: 3,274,752,000

📈 各类停顿占总停顿的百分比:
  ifet      :   3,274,752,000 周期 (   5.7% 总停顿) [平均 1.00/指令]
  sync      :   1,400,745,984 周期 (   2.5% 总停顿) [平均 0.43/指令]
  mthr      :   1,955,409,920 周期 (   3.4% 总停顿) [平均 0.60/指令]
  idep      :   8,557,920,256 周期 (  15.0% 总停顿) [平均 2.61/指令]
  othr      :     164,270,080 周期 (   0.3% 总停顿) [平均 0.05/指令]
  cmem      :   9,483,685,888 周期 (  16.6% 总停顿) [平均 2.90/指令]
  gmem      :  31,407,362,048 周期 (  54.9% 总停顿) [平均 9.59/指令]
  pipe      :     913,137,664 周期 (   1.6% 总停顿) [平均 0.28/指令]
  ifet_pct  :           5,731 周期 (   0.0% 总停顿) [平均 0.00/指令]
  sync_pct  :           1,538 周期 (   0.0% 总停顿) [平均 0.00/指令]
  mthr_pct  :           2,324 周期 (   0.0% 总停顿) [平均 0.00/指令]
  idep_pct  :           9,349 周期 (   0.0% 总停顿) [平均 0.00/指令]
  othr_pct  :             142 周期 (   0.0% 总停顿) [平均 0.00/指令]
  cmem_pct  :          11,581 周期 (   0.0% 总停顿) [平均 0.00/指令]
  gmem_pct  :          14,746 周期 (   0.0% 总停顿) [平均 0.00/指令]
  pipe_pct  :             788 周期 (   0.0% 总停顿) [平均 0.00/指令]

🔍 停顿类型分析 (按占比排序):
  1. 🚨 主要瓶颈 gmem: 54.9% 的停顿时间
      💡 全局内存瓶颈 - 优化内存访问模式
  2. 📝 次要因素 cmem: 16.6% 的停顿时间
      💡 常量内存访问过多 - 检查参数传递和常量缓存
  3. 📝 次要因素 idep: 15.0% 的停顿时间
      💡 指令依赖严重 - 考虑增加并行度或kernel融合

✅ 验证: 各停顿类型总和 = 100.0% (应接近100%)

==================================================
📊 派生百分比指标分析
==================================================
📈 关键百分比指标:
  gxcopy:h2d_pct: 最大=1000.00%, 平均=17.98%
    Top函数:
      1. entry: 1000.00%
      2. src/home/ynxia/playground/param/train/workloads/dlrm/dlrm_s_pytorch.py:1057: 1000.00%
      3. loss_fn_wrap: 1000.00%
  gxcopy:count_pct: 最大=1000.00%, 平均=22.01%
    Top函数:
      1. entry: 1000.00%
      2. src/home/ynxia/playground/param/train/workloads/dlrm/dlrm_s_pytorch.py:1065: 1000.00%
      3. cpu: 1000.00%
  gxcopy:d2h_pct: 最大=1000.00%, 平均=4.30%
    Top函数:
      1. entry: 1000.00%
      2. src/home/ynxia/playground/param/train/workloads/dlrm/dlrm_s_pytorch.py:1065: 1000.00%
      3. cpu: 1000.00%
  gins:stl_ifet_pct: 最大=37.93%, 平均=0.65%
    Top函数:
      1. src/home/ynxia/playground/param/train/workloads/dlrm/dlrm_s_pytorch.py:1076: 37.93%
      2. backward: 37.93%
      3. src/home/ynxia/playground/param/.venv/lib/python3.11/site-packages/torch/_tensor.py:626: 37.93%
  gins:stl_sync_pct: 最大=31.33%, 平均=0.18%
    Top函数:
      1. src/home/ynxia/playground/param/train/workloads/dlrm/dlrm_s_pytorch.py:301: 31.33%
      2. bmm: 31.33%
      3. torch::autograd::THPVariable_bmm(_object*, _object*, _object*): 31.33%
  gins:stl_mthr_pct: 最大=20.06%, 平均=0.27%
    Top函数:
      1. [libtorch_cuda.so]:0: 20.06%
      2. at::cuda::blas::gemm<float>(char, char, long, long, long, at::OpMathType<float>::type, float const*, long, float const*, long, at::OpMathType<float>::type, float*, long): 20.06%
      3. [libtorch_cuda.so]:0: 20.06%
  gins:stl_idep_pct: 最大=37.60%, 平均=1.07%
    Top函数:
      1. src/home/ynxia/playground/param/train/workloads/dlrm/dlrm_s_pytorch.py:345: 37.60%
      2. apply_emb: 37.60%
      3. src/home/ynxia/playground/param/train/workloads/dlrm/dlrm_s_pytorch.py:288: 37.60%
  gins:stl_othr_pct: 最大=1.22%, 平均=0.02%
    Top函数:
      1. [libtorch_cuda.so]:0: 1.22%
      2. at::cuda::blas::gemm<float>(char, char, long, long, long, at::OpMathType<float>::type, float const*, long, float const*, long, at::OpMathType<float>::type, float*, long): 1.22%
      3. [libtorch_cuda.so]:0: 1.22%
  gins:stl_cmem_pct: 最大=48.33%, 平均=1.32%
    Top函数:
      1. src/home/ynxia/playground/param/train/workloads/dlrm/dlrm_s_pytorch.py:316: 48.33%
      2. cat: 48.33%
      3. /home/ynxia/playground/param/.venv/lib/python3.11/site-packages/torch/lib/libtorch_python.so:7723299: 48.33%
  gins:stl_gmem_pct: 最大=80.73%, 平均=1.69%
    Top函数:
      1. at::(anonymous namespace)::wrapper_CUDA_add__Tensor(at::Tensor&, at::Tensor const&, c10::Scalar const&): 80.73%
      2. [libtorch_cuda.so]:0: 80.73%
      3. at::native::add_kernel(at::TensorIteratorBase&, c10::Scalar const&): 80.73%
  gins:stl_pipe_pct: 最大=4.47%, 平均=0.09%
    Top函数:
      1. [libtorch_cuda.so]:0: 4.47%
      2. at::cuda::blas::gemm<float>(char, char, long, long, long, at::OpMathType<float>::type, float const*, long, float const*, long, at::OpMathType<float>::type, float*, long): 4.47%
      3. [libtorch_cuda.so]:0: 4.47%

==================================================
🔥 热点分析 (Top 10 函数)
==================================================
📊 Top 10 热点函数详细分析:
(按总时间排序)

+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+
| name                                                            |   time (inc) |   gtime (inc) | gxcopy:h2d (inc)   | gxcopy:d2h (inc)   |   cpu_gpu_ratio |   h2d_bw_MBps |   d2h_bw_MBps |
+=================================================================+==============+===============+====================+====================+=================+===============+===============+
| entry                                                           |           51 |             0 | 441,935,140        | 819,600            |       551       |          8.71 |          0.02 |
+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+
| [libcudart.so.12]:0                                             |           29 |             0 | 0                  | 0                  |         2.9e+10 |          0    |          0    |
+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+
| [libcudart.so.12]:0                                             |           29 |             0 | 0                  | 0                  |         2.9e+10 |          0    |          0    |
+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+
| /home/ynxia/playground/param/.venv/lib/python3.11/site-packa... |           29 |             0 | 0                  | 0                  |         2.9e+10 |          0    |          0    |
+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+
| _cuda_getDeviceCount                                            |           29 |             0 | 0                  | 0                  |         2.9e+10 |          0    |          0    |
+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+
| src/home/ynxia/playground/param/.venv/lib/python3.11/site-pa... |           29 |             0 | 0                  | 0                  |         2.9e+10 |          0    |          0    |
+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+
| ../sysdeps/nptl/internaltypes.h:116                             |           29 |             0 | 0                  | 0                  |         2.9e+10 |          0    |          0    |
+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+
| <unknown procedure> 0x83170                                     |           29 |             0 | 0                  | 0                  |         2.9e+10 |          0    |          0    |
+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+
| is_available                                                    |           29 |             0 | 0                  | 0                  |         2.9e+10 |          0    |          0    |
+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+
| /home/ynxia/playground/param/.venv/lib/python3.11/site-packa... |           29 |             0 | 0                  | 0                  |         2.9e+10 |          0    |          0    |
+-----------------------------------------------------------------+--------------+---------------+--------------------+--------------------+-----------------+---------------+---------------+

==================================================
⚡ GPU Kernel 专项分析
==================================================
🎯 GPU Kernel 相关函数:
  1. _cuda_getDeviceCount: 29
  2. cudaGetDeviceCount: 29
  3. /home/ynxia/playground/param/.venv/lib/python3.11/site-packages/torch/lib/libc10_cuda.so:373385: 29
  4. [libcudart.so.12]:0: 29
  5. [libcudart.so.12]:0: 29

==================================================
🐍 Python 栈耗时专项分析
==================================================
📈 Python 代码热点:
  1. src/home/ynxia/playground/param/.venv/lib/python3.11/site-packages/torch/cuda/__init__.py:129: 29 (1.9%)
  2. /home/ynxia/playground/param/.venv/lib/python3.11/site-packages/torch/lib/libtorch_python.so:14515666: 29 (1.9%)
  3. /home/ynxia/playground/param/.venv/lib/python3.11/site-packages/torch/lib/libc10_cuda.so:373385: 29 (1.9%)
  4. /home/ynxia/playground/param/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12:237055: 29 (1.9%)
  5. /home/ynxia/playground/param/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12:223753: 29 (1.9%)

==================================================
💡 增强优化建议
==================================================
  📡 数据传输优化:
     - 使用CUDA unified memory
     - 批量化数据传输
     - 考虑在GPU上保持数据

============================================================
✅ 完整的性能分析完成！
============================================================
root@H100-136-velinux2:/home/ynxia/playground/dlrm/workloads# 
thon.so:7723299: 48.33%ib/libtorch_python.so:7723299: 48.33%
  gins:stl_gmem_pct: 最大=80.73%, 平均=1.69%.69%
    Top函数:
      1. at::(anonymous namespace)::wrapper_CUDA_add__Tensor(at::Tensor&, at::Tensor const&, c10::Scalar const&): 80.73%nsor const&, c10::Scalar const&): 80.73%
      2. [libtorch_cuda.so]:0: 80.73%