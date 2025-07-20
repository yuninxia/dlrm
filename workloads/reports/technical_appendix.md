
# DLRM GPU Performance Analysis - Technical Detailed Report

**Generated**: 2025-07-15 01:16:49  
**Data Source**: HPCToolkit Database  
**Analysis Framework**: Hatchet + Pandas  
**Function Count**: 8768  
**Metric Count**: 46  

## 🔍 Complete Metric Statistics

### CPU Performance Metrics
- **time (inc)**: Total=1.514e+03, Average=1.726e-01, Max=5.071e+01
- **time**: Total=9.933e+01, Average=6.255e-02, Max=2.333e+01
- **gtime (inc)**: Total=3.675e+00, Average=4.192e-04, Max=9.209e-02
- **cpu_gpu_ratio**: Total=1.196e+12, Average=1.364e+08, Max=2.899e+10

### GPU Compute Metrics
- **gpuop (inc)**: Total=3.675e+00, Average=5.761e-03, Active Functions=638
- **gins:stl_ifet (inc)**: Total=3.275e+09, Average=7.088e+06, Active Functions=462
- **gins:stl_sync (inc)**: Total=1.401e+09, Average=1.197e+07, Active Functions=117
- **gins:stl_mthr (inc)**: Total=1.955e+09, Average=4.326e+06, Active Functions=452
- **gker:dymem_acumu (inc)**: Total=2.344e+07, Average=2.150e+05, Active Functions=109
- **gins:stl_idep (inc)**: Total=8.558e+09, Average=1.852e+07, Active Functions=462
- **gker (inc)**: Total=7.948e-01, Average=1.720e-03, Active Functions=462
- **gker:fgp_max_acumu (inc)**: Total=9.286e+06, Average=2.010e+04, Active Functions=462
- **gker:lmem_acumu (inc)**: Total=3.577e+13, Average=7.743e+10, Active Functions=462
- **gker:fgp_act_acumu (inc)**: Total=4.476e+06, Average=9.687e+03, Active Functions=462
- **gins:stl_any (inc)**: Total=5.716e+10, Average=1.237e+08, Active Functions=462
- **gins:stl_othr (inc)**: Total=1.643e+08, Average=4.576e+05, Active Functions=359
- **gker:stmem_acumu (inc)**: Total=3.508e+07, Average=3.693e+05, Active Functions=95
- **gker:blk_thr_acumu (inc)**: Total=2.456e+07, Average=5.317e+04, Active Functions=462
- **gker:blks_acumu (inc)**: Total=2.456e+07, Average=5.317e+04, Active Functions=462
- **gins:stl_cmem (inc)**: Total=9.484e+09, Average=2.053e+07, Active Functions=462
- **gins:stl_gmem (inc)**: Total=3.141e+10, Average=7.304e+07, Active Functions=430
- **gins (inc)**: Total=7.438e+10, Average=1.610e+08, Active Functions=462
- **gker:thr_reg_acumu (inc)**: Total=5.518e+06, Average=1.194e+04, Active Functions=462
- **gins:stl_pipe (inc)**: Total=9.131e+08, Average=2.686e+06, Active Functions=340
- **gker:blk_sm_acumu (inc)**: Total=5.852e+07, Average=3.775e+05, Active Functions=155
- **gker:count (inc)**: Total=1.451e+05, Average=3.141e+02, Active Functions=462
- **gins:stl_ifet_pct**: Total=5.731e+03, Average=6.536e-01, Active Functions=462
- **gins:stl_sync_pct**: Total=1.538e+03, Average=1.826e-01, Active Functions=117
- **gins:stl_mthr_pct**: Total=2.324e+03, Average=2.654e-01, Active Functions=452
- **gins:stl_idep_pct**: Total=9.349e+03, Average=1.066e+00, Active Functions=462
- **gins:stl_othr_pct**: Total=1.424e+02, Average=1.643e-02, Active Functions=359
- **gins:stl_cmem_pct**: Total=1.158e+04, Average=1.321e+00, Active Functions=462
- **gins:stl_gmem_pct**: Total=1.475e+04, Average=1.688e+00, Active Functions=430
- **gins:stl_pipe_pct**: Total=7.877e+02, Average=9.110e-02, Active Functions=340

### GPU Memory Transfer Metrics
- **gxcopy:h2d (inc)**: Total=17165249852 bytes (17165.2 MB)
- **gxcopy (inc)**: Total=2.881e+00, Average=1.493e-02
- **gxcopy:count (inc)**: Total=2.528e+04, Average=1.310e+02
- **gxcopy:d2h (inc)**: Total=834000 bytes (0.8 MB)
- **gxcopy:h2d_pct**: Total=157000 bytes (0.2 MB)
- **gxcopy:count_pct**: Total=1.930e+05, Average=2.201e+01
- **gxcopy:d2h_pct**: Total=37000 bytes (0.0 MB)
- **h2d_bw_MBps**: Total=220019 bytes (0.2 MB)
- **d2h_bw_MBps**: Total=1 bytes (0.0 MB)

### GPU Stall Metrics
- **gins:stl_ifet (inc)**: Total=3.275e+09, Active Functions=462
- **gins:stl_sync (inc)**: Total=1.401e+09, Active Functions=117
- **gins:stl_mthr (inc)**: Total=1.955e+09, Active Functions=452
- **gins:stl_idep (inc)**: Total=8.558e+09, Active Functions=462
- **gins:stl_any (inc)**: Total=5.716e+10, Active Functions=462
- **gins:stl_othr (inc)**: Total=1.643e+08, Active Functions=359
- **gins:stl_cmem (inc)**: Total=9.484e+09, Active Functions=462
- **gins:stl_gmem (inc)**: Total=3.141e+10, Active Functions=430
- **gins:stl_pipe (inc)**: Total=9.131e+08, Active Functions=340
- **gins:stl_ifet_pct**: Total=5.731e+03, Active Functions=462
- **gins:stl_sync_pct**: Total=1.538e+03, Active Functions=117
- **gins:stl_mthr_pct**: Total=2.324e+03, Active Functions=452
- **gins:stl_idep_pct**: Total=9.349e+03, Active Functions=462
- **gins:stl_othr_pct**: Total=1.424e+02, Active Functions=359
- **gins:stl_cmem_pct**: Total=1.158e+04, Active Functions=462
- **gins:stl_gmem_pct**: Total=1.475e+04, Active Functions=430
- **gins:stl_pipe_pct**: Total=7.877e+02, Active Functions=340

## 🔥 Top 10 Hotspot Functions

| Rank | Function Name | CPU Time | GPU Time | Transfer Time |
|------|--------|---------|---------|----------|
| 1 | entry | 50.712 | 0.092 | 0.074 |
| 2 | is_available | 28.993 | nan | nan |
| 3 | src/home/ynxia/playground/param/.venv/lib/python3.... | 28.993 | nan | nan |
| 4 | _cuda_getDeviceCount | 28.993 | nan | nan |
| 5 | /home/ynxia/playground/param/.venv/lib/python3.11/... | 28.993 | nan | nan |
| 6 | /home/ynxia/playground/param/.venv/lib/python3.11/... | 28.993 | nan | nan |
| 7 | cudaGetDeviceCount | 28.993 | nan | nan |
| 8 | [libcudart.so.12]:0 | 28.993 | nan | nan |
| 9 | <unknown procedure> 0x2a3b0 | 28.993 | nan | nan |
| 10 | [libcudart.so.12]:0 | 28.993 | nan | nan |

    
## 📊 Data Files

- **Plot Directory**: `plots/`
- **Original Data**: `reports/metrics_data.json`
- **Analysis Script**: `hatchet_analysis.py`

## 🔧 Reproduction Steps

```bash
# 1. Data Collection
hpcrun -e gpu=nvidia python dlrm_main.py

# 2. Data Processing  
hpcstruct dlrm_binary
hpcprof -S dlrm_binary.hpcstruct hpctoolkit-measurements

# 3. Analysis and Visualization
python hatchet_analysis.py
```

---
*Technical detailed report includes complete metric statistics, supporting deep optimization analysis*
