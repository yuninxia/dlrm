# HPCAnalysis GINS Metric Parsing Fix

## Problem
The `query_metric_descriptions` method in HPCAnalysis was incorrectly parsing GINS (GPU Instruction) metrics that contain multiple colons, such as `gins:stl_gmem:sum (i)`. The original implementation split on the first colon and treated everything after it as the aggregation, which failed for GINS stall metrics.

## Root Cause
GINS metrics have a different naming convention:
- Traditional metrics: `metric:aggregation` (e.g., `cputime:sum`)
- GINS metrics: `gins:stall_type:aggregation` (e.g., `gins:stl_gmem:sum`)

The original parser assumed only one colon separator between metric name and aggregation.

## Solution
Modified the `query_metric_descriptions` method in `/hpcanalysis/data_query.py` to:

1. **Parse the metric query more intelligently** based on the number of colons
2. **Special handling for GINS metrics** that start with "gins:"
3. **Preserve backward compatibility** with traditional metric formats

### Key Changes

```python
# Original problematic code:
if ":" in name:
    aggregation = name.split(":")[1]
    name = name.split(":")[0]

# Fixed code:
parts = main_part.split(":")

if len(parts) >= 3:
    # GINS metric with aggregation: "gins:stl_gmem:sum"
    if parts[0] == "gins":
        # Reconstruct the metric name
        name = ":".join(parts[:-1])  # "gins:stl_gmem"
        aggregation = parts[-1]       # "sum"
```

## Verification
The fix has been tested with:
- ✅ Traditional metrics (cputime, gpuop, etc.)
- ✅ GINS base metric (gins:sum)
- ✅ All GINS stall metrics (stl_gmem, stl_sync, stl_idep, etc.)
- ✅ Different aggregations (sum, prop, min, max)
- ✅ Different scopes (i, e, p, c)
- ✅ Edge cases and invalid queries

## Impact
This fix enables:
1. Access to GPU PC sampling stall data through HPCAnalysis
2. Proper analysis of GPU performance bottlenecks
3. Programmatic extraction of metrics visible in hpcviewer

## Note on Sampling Rate
GINS metrics use a 1% sampling rate. Raw values should be divided by 100 to match hpcviewer display values:

```python
raw_value = query_api.query_profile_slices('*', '*', 'gins:stl_gmem:sum (i)')
corrected_value = raw_value / 100  # Matches hpcviewer
```