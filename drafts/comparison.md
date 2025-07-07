## TLDR
*Nsight Systems* can completely capture **H2D/D2H transfer bytes, bandwidth, kernel duration, CPU thread time**;
*Nsight Compute* can provide **Warp Stall breakdown, instruction execution count, memory access bandwidth, occupancy** at the *single kernel* level;
*PyTorch Profiler* (torch.profiler) can only provide **Kernel timeline, cudaMemcpy events, memory usage**, currently unable to get Warp-stall details. Below, we'll explain whether each tool can capture the key columns appearing in your script, along with enabling methods and limitations.

---

## 1 Overall Comparison Table

| HPCToolkit Field                              | Nsight Systems                                                                                        | Nsight Compute                                                                                                                                                                | PyTorch Profiler                                                          | Notes / Enabling Method                        |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------- |
| `gxcopy:h2d (inc)`<br>H2D transfer bytes     | **✓** CUDA Memcpy row has Bytes, `nsys stats --report cuda-apis` can summarize ([forums.developer.nvidia.com][1]) | ✗ (only sees internal copies within single kernel)                                                                                                                           | **✓** `torch.profiler` captures cudaMemcpy callback `CUDAMemoryCopy`([github.com][2]) | Nsight Systems can also provide bandwidth; Profiler needs manual sum |
| `gxcopy:d2h (inc)`                           | Same as above                                                                                         | Same as above                                                                                                                                                                 | Same as above                                                             | —                                              |
| `gker (inc)`<br>Total kernel time            | **✓** Timeline & CUDA Kernel row ([docs.nvidia.com][3])                                             | **✓** Each kernel's `duration` / `launch__.*` metric ([developer.nvidia.com][4])                                                                                            | **✓** `self_cuda_time_total`                                              |                                                |
| `gins (inc)`<br>GPU instruction count        | ✗ (doesn't do PC sampling)                                                                           | **✓** `inst_executed.sum` metric; need to include *Launch Statistics* in report ([developer.nvidia.com][4])                                                                | ✗                                                                         |                                                |
| `gins:stl_gmem / stl_ifet …`<br>Warp Stall breakdown | ✗                                                                                                 | **✓** Warp State Stats → Stall Reasons: Long Scoreboard≈`stl_gmem`, Instr Fetch≈`stl_ifet`, Execution Dep≈`stl_idep`… ([forums.developer.nvidia.com][5], [stackoverflow.com][6]) | ✗                                                                         |                                                |
| `cpu time (inc)` vs `gtime (inc)`            | **✓** CPU Thread row (sampling period) and GPU row both have timeline, can read ratio in *Summary → GPU Busy* panel ([docs.nvidia.com][3]) | ✗                                                                                                                                                                         | Partial support (CPU events only at Python layer)                       |                                                |
| `LLC_MISSES` etc CPU PMU                     | **✓** Starting from 2025.3 `--event-sample` supports x86/Grace PMU sampling ([docs.nvidia.com][3])                                      | ✗                                                                                                                                                                         | ✗                                                                         |                                                |
| `gsync / gmem` etc sync, memory wait time    | **✓** View bar *CUDA Wait*                                                                           | **✓** `dram__throughput.avg`, `l2_subp0_read_transactions` etc can be used to calculate                                                                                      | ✗                                                                         |                                                |

---

## 2 How to Enable and Export Similar Metrics

### 2.1 Nsight Systems (System-level timeline & transfer statistics)

```bash
nsys profile -t cuda,osrt,cudnn,cublas \
             --gpu-metrics-device=all \
             --cuda-graph-trace=all \
             -o dlrm_nsys \
             python dlrm.py
# Summarize API bytes
nsys stats --report cuda-apis dlrm_nsys.qdrep
```

* Transfer bytes and bandwidth are in the **CUDA API Statistics** table, directly corresponding to your script's `gxcopy:h2d (inc)` numbers.
* To sample CPU PMU, add `--event-sample cpu-cycles,cache-misses` (requires 2025.3+ version). ([docs.nvidia.com][3])

### 2.2 Nsight Compute (Single kernel Warp Stall / instruction count)

```bash
ncu --set full  \
    --metrics smsp__sass_average_branch_divergence_per_warp, \
              smsp__warp_issue_stalled_long_scoreboard_per_warp, \
              gpu__time_duration.sum,inst_executed.sum \
    --kernel-name "embedding*"  \
    --target-processes all  \
    python dlrm.py
```

* The **Warp State Statistics → Issue Stall Reasons** in the report are the official counterparts to `gins:stl_*`. ([forums.developer.nvidia.com][5])
* `inst_executed.sum` ≈ HPCToolkit `gins (inc)`; can also get `dram__throughput` to calculate HBM bandwidth.
* Only aggregates within *selected kernels*, doesn't automatically sum totals across kernels—requires post-processing scripts.

### 2.3 PyTorch Profiler (High-level analysis)

```python
with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        record_shapes=True) as prof:
    run_dlrm()
prof.export_chrome_trace("dlrm_trace.json")
```

* `CUDAMemoryCopy` events in Chrome trace can be aggregated to get H2D/D2H bytes.
* No Warp-stall / instruction-level metrics—this is CUPTI *device profiling* path, currently not exposed to PyTorch Profiler; official issue #124547 still tracking. ([github.com][2])

---

## 3 Why Numbers Might Not Match

1. **HPCToolkit's `gins:stl_*` comes from PC-Sampling**, time span covers *entire program*; Nsight Compute defaults to single kernel *replay sampling*, need to merge multiple kernel reports.
2. **Bandwidth differences**: Nsight Systems calculates by event time periods, HPCToolkit gives *total bytes*; to compare bandwidth need manual `bytes / duration`.
3. **Retry copies** (Pinned ↔ Pageable) will make Nsight Systems show two memcpy entries, while HPCToolkit only records once—can cause 5–10% deviation.

---

## 4 Recommended Tool Combinations

| Use Case                          | Recommended Toolchain                                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Debug copy fragmentation/bandwidth** | **Nsight Systems**: Quick view of cudaMemcpy granularity and timing; use `nsys stats` for total bytes. |
| **Drill into kernel Stall/instruction distribution** | **Nsight Compute** (or `ncu-cli + csv`) → corresponds to hpctoolkit `gins:stl_*`, can map to source lines. |
| **End-to-end automated regression** | **HPCToolkit + Hatchet**: Single capture for full program+CCT, scriptable diff; Nsight series more for interactive analysis. |
| **Daily model development**       | **PyTorch Profiler**: Low overhead hot paths, memory usage, cudaMalloc calls, integrated TensorBoard UI, lowest barrier. |

> Using Nsight series to reproduce core metrics from your script is completely feasible; but for "one-click generation of cross-kernel, cross-iteration differential reports", HPCToolkit remains most convenient. Suggestions:
>
> 1. **Nsight Systems** to locate remaining H2D fragmentation → optimize data pipeline;
> 2. **Nsight Compute** to deep-dive hottest kernel's `Long Scoreboard`, `Global Memory` Stall;
> 3. Finally use **HPCToolkit** for reproducible benchmarks submitted to top conferences or CI, maintaining closed-loop toolchain.

[1]: https://forums.developer.nvidia.com/t/nsight-systems-dram-bandwidth-under-gpu-metrics/284223?utm_source=chatgpt.com "Nsight Systems, 'DRAM Bandwidth' under GPU Metrics"
[2]: https://github.com/pytorch/pytorch/issues/124547?utm_source=chatgpt.com "Profiler does not record CUDA times · Issue #124547 - GitHub"
[3]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html?utm_source=chatgpt.com "User Guide — nsight-systems 2025.3 documentation - NVIDIA Docs"
[4]: https://developer.nvidia.com/nsight-compute-2025_1-new-features2?utm_source=chatgpt.com "Nsight Compute 2025.1 - New Features - NVIDIA Developer"
[5]: https://forums.developer.nvidia.com/t/long-scoreboard-stall-meanings/230738?utm_source=chatgpt.com "Long scoreboard stall meanings? - NVIDIA Developer Forums"
[6]: https://stackoverflow.com/questions/14887807/what-are-other-issue-stall-reasons-displayed-by-the-nsight-profiler?utm_source=chatgpt.com "What are \"Other\" Issue Stall Reasons displayed by the Nsight profiler?"
