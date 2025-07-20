# HPCAnalysis Comprehensive Capabilities Report

## Overview
After thoroughly examining the HPCAnalysis source code, I've identified all available APIs, metrics, data extraction methods, and capabilities. This report documents every feature found in the library.

## Core Data Structures and Classes

### 1. DataRead (data_read.py)
Primary class for reading HPCToolkit database files.

**Key Methods:**
- `read_cct()` - Reads the Calling Context Tree (CCT) with support for all entry points
- `read_metric_descriptions()` - Extracts all available metric metadata
- `read_profile_descriptions()` - Gets profile information including GPU streams/contexts
- `read_profile_slices()` - Reads metric values for specific profiles/CCT nodes
- `read_trace_slices()` - Reads trace data with timestamps

**Important Features:**
- Supports parallel reading with configurable `_tasks_count`
- Handles summary profiles (profile_id=0)
- Binary search optimization for large datasets
- Supports all entry points (not just main thread)

### 2. DataQuery (data_query.py)
Advanced querying interface built on top of DataRead.

**Query Methods:**
- `query_cct()` - Flexible CCT querying with pattern matching
- `query_profile_descriptions()` - Profile filtering with complex expressions
- `query_metric_descriptions()` - Metric selection with aggregation options
- `query_profile_slices()` - Combined profile/CCT/metric data extraction
- `query_trace_slices()` - Time-based trace data queries

**Query Expression Patterns:**
- CCT: `"entry|function|loop|line|instruction"`
- Profiles: `"node|rank|thread|gpudevice|gpucontext|gpustream|core"`
- Metrics: `"(metric_name):(aggregation) (scope)"`

### 3. DataAnalysis (data_analysis.py)
High-level analysis and visualization capabilities.

**Analysis Methods:**
- `to_hatchet()` - Converts to Hatchet GraphFrame format for advanced analysis
- `visualize_cct()` - Interactive CCT visualization (Jupyter/text modes)
- `hpcreport()` - Comprehensive performance report (CPU/GPU breakdown)
- `flat_profile()` - Function-level performance summary
- `gpu_idleness()` - GPU utilization analysis
- `load_imbalance()` - MPI rank load balance analysis

## Available Metrics

### Time Metrics (TIME_METRICS)
- `cputime` - CPU time
- `realtime` - Wall clock time
- `cycles` - CPU cycles

### GPU Metrics (GPU_METRICS)
- `gpuop` - GPU all operations
- `gker` - GPU kernel execution
- `gmem` - GPU memory allocation/deallocation
- `gmset` - GPU memory set
- `gxcopy` - GPU explicit data copy
- `gicopy` - GPU implicit data copy
- `gsync` - GPU synchronization

### Metric Aggregations
- `sum` - Sum aggregation
- `min` - Minimum value
- `max` - Maximum value
- `prop` - Proportional/inclusive metric

### Metric Scopes
- `i` - Execution/exclusive
- `e` - Function/inclusive
- `p` - Point
- `c` - Lexically aware

## Advanced Features

### 1. CCT Reduction Filters
Reduces CCT complexity by filtering nodes:
- `OpenMPReduction` - Filters OpenMP runtime functions
- `MPIReduction` - Filters MPI implementation details
- `FunctionReduction` - Shows only function-level nodes
- `TimeReduction` - Filters nodes below time threshold

### 2. MPI Function Categorization
Extensive categorization of 484 MPI functions into:
- Point-to-Point Communication
- One-Sided Communication
- Collective Communication
- Groups and Communicators
- Process Topologies
- Process Creation and Management
- Environmental Inquiry and Profiling
- Miscellaneous
- I/O Operations

### 3. OpenMP Function Categorization
- OpenMP Idle
- OpenMP Overhead
- OpenMP Wait (barrier, task, mutex)
- OpenMP Work (regions, tasks)

### 4. Profile Identifiers
Supports querying by:
- `node` - Compute node
- `rank` - MPI rank
- `thread` - Thread ID
- `core` - CPU core
- `gpudevice` - GPU device ID
- `gpucontext` - GPU context
- `gpustream` - GPU stream

### 5. CCT Node Types
- `entry` - Entry points (main thread, GPU threads, etc.)
- `function` - Function calls
- `loop` - Loop constructs
- `line` - Source line locations
- `instruction` - Assembly instructions

## Data Extraction Capabilities

### 1. Raw Data Access
- Direct binary file parsing (meta.db, prof.db, ctxt.db, trce.db)
- Custom binary search implementations for performance
- Efficient memory-mapped reading

### 2. Metadata Extraction
- Source file paths and line numbers
- Load module paths and offsets
- Function names with demangling
- Metric units and descriptions

### 3. Time-Series Data
- Trace data with microsecond timestamps
- Time frame queries for specific intervals
- Duration calculations (start/end timestamps)

### 4. Statistical Functions
- Mean, variance calculations (load_imbalance)
- Percentage calculations relative to total
- Aggregation across profiles/ranks

## Additional Utilities

### 1. Format Conversion
- Pandas DataFrame output for all queries
- Hatchet GraphFrame integration
- JSON-compatible data structures

### 2. Visualization Support
- IPython/Jupyter notebook detection
- Interactive tree visualization (ipytree)
- Text-based tree display (treelib)

### 3. Performance Optimizations
- Parallel processing with joblib
- Caching of read data
- Binary search for large datasets
- Lazy loading of profile/trace data

## Query Expression Examples

### Complex Profile Queries
```python
# Multiple ranks with ranges
"rank(0-15:2,20,25-30)"

# Combined dimensions
"node(0).rank(0-7).thread(0)"

# GPU-specific
"gpudevice(0).gpustream(1-4)"
```

### CCT Path Queries
```python
# Function path
"entry(main thread).function(main).function(MPI_*)"

# Line-based path
"function(compute).line(solver.cpp:45)"

# Instruction-level
"function(kernel).instruction(libcuda.so:0x1234)"
```

### Metric Queries
```python
# Time metrics with specific aggregation
"time:sum (i)"

# Multiple GPU metrics
["gker:sum (i)", "gmem:sum (i)", "gxcopy:sum (i)"]

# All metrics
"*"
```

## Features Not Previously Explored

1. **Multi-entry point support** - Reads all entry points, not just main thread
2. **GPU stream/context analysis** - Full GPU execution hierarchy
3. **Instruction-level profiling** - Assembly-level performance data
4. **Time-based filtering** - Trace analysis for specific time windows
5. **Load module offset tracking** - Binary-level performance attribution
6. **Parallel data loading** - Configurable parallel processing
7. **Summary profile caching** - Optimized repeated queries
8. **Flexible metric scoping** - Point, exclusive, inclusive, lexical
9. **MPI/OpenMP categorization** - Automatic function classification
10. **Hatchet integration** - Export to advanced analysis framework

## Recommendations for Complete Data Extraction

To extract ALL available information:

1. Use `query_metric_descriptions("*")` to get all metrics
2. Use `query_profile_descriptions("*")` to get all profiles
3. Use `query_cct("*")` for complete CCT
4. Enable trace reading for temporal analysis
5. Use DataAnalysis methods for derived metrics
6. Export to Hatchet for advanced graph algorithms
7. Access raw binary data for custom analysis