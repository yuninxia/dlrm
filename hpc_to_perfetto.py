import sys
import os

# Add the hpcanalysis-mirror directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
hpcanalysis_path = os.path.join(script_dir, "hpcanalysis-mirror")
if hpcanalysis_path not in sys.path:
    sys.path.insert(0, hpcanalysis_path)

import hpcanalysis
import pandas as pd
import json

# Correct imports based on our findings
from hpcanalysis.data_read import DataRead
from hpcanalysis.data_query import DataQuery

def main():
    """
    Main function to convert HPCToolkit data to Perfetto format.
    """
    # Path to the HPCToolkit database for GPU
    db_path = "workloads/hpctoolkit-python3.11-database-gpu"

    print(f"Opening HPCToolkit database at: {db_path}")
    
    # Correct way to open the database
    read_api = DataRead(db_path)
    query_api = DataQuery(read_api)

    # --- Step 2: Read HPCToolkit data ---
    print("Reading data from the database...")

    # Extract profile descriptions (process/thread info)
    profiles_df = query_api.query_profile_descriptions("*")
    print("\n--- Profile Descriptions (first 5 rows): ---")
    print(profiles_df.head())

    # Extract the Calling Context Tree (CCT)
    cct_df = query_api.query_cct("*")
    print("\n--- Calling Context Tree (first 5 rows): ---")
    print(cct_df.head())

    print("\nQuerying trace slices...")
    try:
        # First check if GPU profiles are included
        gpu_profiles = profiles_df[(profiles_df['gpustream'].notna()) | (profiles_df['gpucontext'].notna())]
        print(f"Found {len(gpu_profiles)} GPU profiles")
        if len(gpu_profiles) > 0:
            print("GPU profiles:")
            print(gpu_profiles[['node', 'thread', 'gpudevice', 'gpucontext', 'gpustream']].head())
        
        traces_df = query_api.query_trace_slices("*")
        print(f"Successfully retrieved {len(traces_df)} trace records.")
        print("\n--- Trace Slices (first 5 rows): ---")
        print(traces_df.head())
    except Exception as e:
        print(f"Could not get trace slices: {e}")
        traces_df = pd.DataFrame()


    print("\nData reading complete. Next, we will process this data.")

    # --- Step 4: Convert to Perfetto format ---
    perfetto_trace = convert_to_perfetto(query_api, profiles_df, cct_df, traces_df)

    # --- Step 5: Write to JSON file ---
    output_filename = "hpctoolkit_trace.json"
    print(f"\nWriting Perfetto trace to {output_filename}...")
    with open(output_filename, "w") as f:
        json.dump(perfetto_trace, f, indent=2)
    print("Done.")


def is_meaningful_function(raw_name):
    """
    Check if a function name is meaningful and should be included.
    Filter out unknown, mangled, profiler overhead, or meaningless function names.
    """
    name = str(raw_name).strip().lower()
    
    # Filter out functions with "unknown" in the name
    if 'unknown' in name:
        return False
    
    # Filter out profiler overhead and system monitoring functions
    profiler_patterns = [
        # System call monitoring
        '__gi___ioctl', '__gi___libc_malloc', '__gi___libc_write', '__gi___libc_read',
        '__gi___libc_open', '__gi___close', '__gi___mmap', '__gi___munmap', 
        '__gi___mprotect', '__gi___brk', '__gi___fstat', '__gi___lseek',
        
        # Memory allocation tracking
        '__malloc', '__free', '__calloc', '__realloc', '__libc_malloc', 
        '__libc_calloc', '__libc_free', '__libc_realloc', 'mid_memalign',
        'int_malloc', 'int_free', 'int_realloc', 'int_calloc',
        'malloc_consolidate', 'malloc_check', 'free_check',
        'grow_heap', 'shrink_heap', 'alloc_new_heap', 'new_heap', 
        'heap_trim', 'systrim', 'sbrk', 'brk', 'mmap', 'munmap',
        'mremap', 'mprotect', 'madvise', 'arena_get', 'arena_put',
        
        # String and memory operations (but keep GPU memcpy operations)
        'memmove_chk', 'memmove_chk_sse2', 'memmove_chk_avx',
        'strcpy', 'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp',
        'strlen', 'strchr', 'strrchr', 'strstr', 'strtok',
        '_sse2_', '_avx_', '_avx2_', '_avx512_', '_evex_', '_erms_',
        # Note: Keeping memcpy operations that might be GPU-related
        
        # Thread monitoring
        '__pthread_', 'pthread_create', 'pthread_exit', 'pthread_mutex_',
        'pthread_rwlock_', '__gi___pthread_',
        
        # Dynamic library loading tracking
        'dlopen', 'dlsym', 'dlerror', 'dl_open_', '__gi__dl_catch_',
        '_dl_open', '_dl_init', '_dl_relocate_', 'dl_map_object', 
        'dl_lookup_symbol', 'dl_audit_symbind', 'elf_dynamic_do_', 
        'elf_machine_', 'resolve_map', 'do_lookup_x', 'openaux',
        'create_dynamic', 'open_path', 'dl_map_object_deps',
        
        # CUDA profiling interface
        'cupti', 'cupti_pc_sampling', 'f_cuptiactivity', 'cuptiactivity',
        
        # Signal handling and error monitoring
        'pyerr_checksignals', 'pytracemalloc', '_tracemalloc', 'signal_handler',
        
        # Profiler infrastructure
        'hpctoolkit', 'hpcrun_', 'hpcrun', 'profiler', 'monitor', 'sampling', 'trace_', 'prof_',
        'frame_dummy', '__do_global_', '__static_initialization',
        
        # HPCToolkit specific internals
        'ip_normalized', 'cct_', 'splay', 'gpu_activity_channel', 
        'receive_activity', 'activity_channel', 'correlation_id',
        'cupti_buffer', 'cupti_correlation', 'cupti_event',
        'metric_set', 'metric_desc', 'sample_source',
        'hpcrun_sample', 'hpcrun_trace', 'hpcrun_metric',
        'hpcrun_cct', 'hpcrun_thread', 'hpcrun_process',
        'gpu_monitoring', 'gpu_trace', 'gpu_correlation',
        'loadmap', 'fnbounds', 'unwind', 'backtrace',
        
        # C++ runtime overhead
        '__cxa_', '__gcc_', '__gnu_', '__glibc_', 'rtld_',
        
        # Memory usage reporting (profiler feature)
        'reportmemoryusagetoprofiler',
        
        # Cryptographic/hashing utilities (not application logic)
        'crypto_compute_hash', 'crypto_compute_hash_string', 'md5_',
        'sha1_', 'sha256_', 'hash_', 'checksum_',
        
        # GPU binary management (driver/runtime internals)
        'gpu_binary_save', 'gpu_binary_load', 'gpu_binary_store',
        'gpu_module_', 'gpu_context_'
        # Note: Removed 'gpu_kernel_' as it might filter actual GPU kernels
    ]
    
    if any(pattern in name for pattern in profiler_patterns):
        return False
    
    # Filter out PyTorch/ATen framework internals
    pytorch_framework_patterns = [
        # ATen tensor library internals
        'at::', 'at_', 'c10::', 'c10_', 'c10d::', 
        'torch::jit::', 'torch::autograd::', 'torch::nn::',
        'caffe2::', 'aten::', '_aten_', 
        
        # PyTorch autograd internals (converted from :: to _)
        'torch_autograd_', 'torch_jit_', 'torch_nn_', 'torch_detail_',
        'torch_library_', 'torch_Library_', 'global_sub_i_', 'std_vector', 
        'torch_library_def', 'torch_Library_def', 'torch_library_fallback',
        'torch_Library_fallback', 'torchlibraryinit',
        
        # Common ATen operations that are just wrappers
        'empty_strided', 'empty_like', 'resize_', 'set_',
        'storage_offset', 'is_contiguous', 'suggest_memory_format',
        
        # C10 dispatcher and backend selection
        'computedispatchkey', 'dispatchkey', 'backendselect',
        'autogradcpu', 'autogradcuda', 'backendmeta',
        
        # Memory allocation internals
        'allocator::', 'cudacachingallocator', 'cpuallocator',
        'pinnedallocator', 'recordstream', 'emptycache',
        
        # CUDA internals
        'cuda::detail::', 'cudnn::', 'cublas::', 'cusparse::',
        'nccl::', 'cufft::', 'curand::',
        
        # CUDA runtime API calls (but these are not the actual kernels)
        # Removed: 'cudalaunchkernel', 'cudamemcpy' as these might be actual operations
        
        # cuBLAS/cuDNN API calls
        'cublassgemm', 'cublasltmatmul', 'cublaslt', 'cudnnconv',
        'cudnnbatchnorm', 'cudnnactivation', 'cudnnpooling',
        
        # Specific ATen internal patterns
        'at_ops_', 'at_native_', 'c10_impl_',
        
        # Generic wrapper functions and implementation details
        'wrap_kernel_functor', 'redispatch', '_call', '_impl', 'impl_',
        
        # THC legacy patterns
        'thc_', 'thcudnn_', 'thcublas_'
    ]
    
    # Also filter out very generic function names that don't provide value
    if name == 'at' or name == 'c10' or name == 'torch':
        return False
    
    if any(pattern.lower() in name.lower() for pattern in pytorch_framework_patterns):
        return False
    
    # Filter out Python interpreter overhead functions
    python_overhead_patterns = [
        # Python profiling overhead
        'python_profile', 'profile_trampoline', '_lsprof',
        
        # Python interpreter internals
        'cpu_main', 'cpu_main', 'gpu_main', 'decorator', 'dedent', 'dedent_lines', 
        'parse', 'parse_param_list', 'find_spec', 'get_spec', 'compile_bytecode', 
        'compile', 'exec_builtin', 'exec_dynamic', 'exec_module', 'load_module', 
        'create_module', 'pyrun_', 'pyx_pymod_exec_', 'classify_pyc',
        'verify_matching_signatures', 'code', 'loads', 'path_stats', 'path_stat',
        
        # Python import system
        'import_module', 'importlib', '_find_and_load', '_load_unlocked',
        '_find_spec', '_get_spec', 'find_module', 'load_module', '_import',
        'builtin_import', '__import__', '_handle_fromlist', 'get_data',
        'get_code', 'get_source', 'get_filename', 'is_package',
        
        # Python compilation and execution
        'compile_bytecode', 'get_code', 'source_to_code', 'compile_source',
        'exec_code', 'run_code', 'eval_code', 'eval_frame', 'evaluate_frame',
        
        # Python object management
        'pyobject_', 'pytype_', 'pygc_', 'pyarray_assignarray', 'pyarray_newfromdescr',
        'pyarray_flatten', 'pyarray_fromany', 'pyarray_discoverdtype',
        
        # Python function calls and evaluation
        'pyeval_', '_pyeval_', 'pyframe_', 'pycode_', 'pyfunction_',
        'pymethod_', 'pycall_', 'pyobject_call', 'pyobject_vectorcall',
        
        # Python imports and modules  
        'pyimport_', 'pymodule_', 'py_bytesmain', 'py_runmain', 'py_finalizeex',
        
        # Python data structures
        'pydict_', 'pylist_', 'pytuple_', 'pyset_', 'pyfrozenset_',
        'pystring_', 'pyunicode_', 'pybytes_', 'pyslice_',
        
        # Python error handling
        'pyerr_', 'pytraceback_',
        
        # Python memory management  
        'pymem_', 'pydatamem_', 'pytracemalloc_',
        
        # Python numbers and operations
        'pylong_', 'pyfloat_', 'pynumber_', 'pysequence_', 'pyiter_',
        'pymapping_', 'pybuffer_', 'pyweakref_',
        
        # Python internal utilities
        '_py_', 'py_decref', 'py_incref', 'py_xdecref', 'py_xincref',
        
        # Python inspection and introspection utilities
        'signature_from_callable', 'inspect_', 'getattr_', 'hasattr_', 'setattr_',
        
        # Python language constructs and generic operations
        'getitem', 'setitem', 'listcomp', 'dictcomp', 'setcomp', 'gencomp',
        'fetch', 'next_data', 'next', '__next__', '__iter__',
        'cpu_main', 'gpu_main', 'main_thread',
        
        # Test/synthetic data generation (not performance-relevant)
        'generate_uniform_input_batch', 'generate_random_', 'create_test_',
        'mock_data', 'synthetic_', 'dummy_',
        
        # NumPy infrastructure overhead (not actual computation)
        'ufunc_generic_fastcall', 'ufunc_generic_vectorcall', 'pyufunc_',
        'promote_and_get_ufuncimpl', 'get_wrapped_legacy_ufunc_loop',
        'pyufunc_defaultlegacyinnerloopselector', 'check_ufunc_fperr',
        
        # Import/module loading infrastructure
        'find_and_load', 'import_module', 'exec_module', 'gcd_import',
        'find_and_load_unlocked', 'load_unlocked', 'module_from_spec',
        
        # Generic wrappers and dispatchers (not actual computation)
        'wrapper', 'wrapfunc', 'wrapped_call_impl', '_call', '_redispatch',
        
        # PyTorch autograd thread management (infrastructure)
        'torch_autograd_engine_thread_main', 'torch_autograd_python_pythonengine_thread_init',
        'engine_thread_main', 'thread_main', 'thread_init'
    ]
    
    if any(pattern in name for pattern in python_overhead_patterns):
        return False
    
    # Filter out heavily mangled names that don't provide useful info
    if name.startswith('_z') and len(name) > 30 and '::' not in name:
        return False
    
    # Filter out pure hex addresses
    if name.startswith('0x') or (name.startswith('lt_') and name.endswith('_gt_') and any(c in name for c in '0123456789abcdef')):
        return False
    
    # Filter out very short or meaningless names
    if len(name) < 3 or name in ['???', 'null', 'none', 'void']:
        return False
    
    # Filter out implementation detail functions
    if 'impl' in name.lower() and not any(keep in name.lower() for keep in ['implement', 'simple']):
        return False
    
    # Special handling for GPU kernels - they often have specific patterns
    # Keep functions that look like GPU kernels
    gpu_kernel_patterns = [
        'kernel<',  # CUDA kernel template
        'memcpyasync',  # Async memory operations
        'memcpyh2d', 'memcpyd2h', 'memcpyd2d',  # GPU memory transfers
        'cutlass::',  # CUTLASS library kernels
        '_kernel_',  # Common kernel naming
        'launch_kernel',
        'gemm_kernel', 'conv_kernel', 'pool_kernel'
    ]
    
    if any(pattern in name.lower() for pattern in gpu_kernel_patterns):
        return True  # Definitely keep GPU kernels
    
    # Also keep functions with angle brackets that might be templated kernels
    if '<' in name and '>' in name and 'kernel' in name.lower():
        return True
    
    # Filter out overly generic single-word function names that don't provide context
    generic_names = [
        'round', 'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
        'set', 'array_empty', 'empty', 'copy', 'get', 'set', 'call', 'init',
        'new', 'delete', 'create', 'destroy', 'start', 'stop', 'run', 'exec',
        'array', 'vector', 'matrix', 'tensor', 'decorator', 'wrapper',
        'parse', 'compile', 'eval', 'apply', 'reduce', 'filter', 'map',
        'zip', 'enumerate', 'range', 'slice', 'property', 'staticmethod',
        'classmethod', 'super', 'type', 'object', 'getattr', 'setattr',
        'hasattr', 'delattr', 'isinstance', 'issubclass', 'callable',
        'repr', 'hash', 'id', 'help', 'print', 'input', 'open', 'close',
        'read', 'write', 'flush', 'seek', 'tell', 'truncate',
        # Low-level math operations (but keep important BLAS operations)
        'add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'exp', 'log', 
        'sin', 'cos', 'tan', 'sum', 'mean', 'std', 'var', 'max', 'min', 
        'argmax', 'argmin', 'cat', 'stack', 'split', 'chunk', 'squeeze', 
        'unsqueeze', 'view', 'reshape', 'transpose', 'permute', 'flatten', 'unfold'
        # Note: Keeping 'gemm', 'gemv', 'bmm', 'mm', 'mv', 'dot', 'ger' as they are important BLAS operations
    ]
    if name in generic_names:
        return False
    
    # Filter out Python standard library modules and their functions
    stdlib_patterns = [
        # Standard library modules
        'importlib.', 'sys.', 'os.', 'io.', 'abc.', 'collections.',
        'itertools.', 'functools.', 'operator.', 'types.', 'typing.',
        'dataclasses.', 'enum.', 'inspect.', 'ast.', 'dis.', 'pickle.',
        'json.', 'csv.', 'configparser.', 'logging.', 'warnings.',
        'traceback.', 'linecache.', 'tokenize.', 'keyword.', 'builtins.',
        
        # Common standard library function patterns
        '_bootstrap', '_install', '_setup', '_init', '_load', '_compile',
        '_parse', '_build', '_create', '_make', '_get', '_set', '_del',
        '_check', '_verify', '_validate', '_process', '_handle',
        
        # Descriptor and metaclass internals
        '__get__', '__set__', '__delete__', '__set_name__', '__prepare__',
        '__instancecheck__', '__subclasscheck__', '__subclasshook__',
        
        # Context manager internals
        '__enter__', '__exit__', '__aenter__', '__aexit__',
        
        # Iteration internals
        '__iter__', '__next__', '__reversed__', '__length_hint__',
        
        # Numeric internals
        '__add__', '__sub__', '__mul__', '__div__', '__mod__', '__pow__',
        '__lshift__', '__rshift__', '__and__', '__or__', '__xor__',
        '__radd__', '__rsub__', '__rmul__', '__rdiv__', '__rmod__',
        
        # Comparison internals
        '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
        '__cmp__', '__rcmp__', '__hash__', '__bool__', '__nonzero__',
        
        # String/representation internals
        '__str__', '__repr__', '__format__', '__bytes__', '__unicode__',
        
        # Attribute access internals
        '__getattr__', '__setattr__', '__delattr__', '__getattribute__',
        '__dir__', '__slots__', '__dict__', '__class__',
        
        # Special method internals
        '__new__', '__init__', '__del__', '__copy__', '__deepcopy__',
        '__reduce__', '__reduce_ex__', '__getstate__', '__setstate__',
        '__getnewargs__', '__getnewargs_ex__', '__getinitargs__'
    ]
    
    if any(pattern in name for pattern in stdlib_patterns):
        return False
    
    # Filter out low-level system calls that are usually not the performance bottleneck
    system_overhead = [
        'clock_gettime', 'gettimeofday', 'time', 'nanosleep', 'usleep',
        'getpid', 'gettid', 'getuid', 'getgid', 'getenv', 'setenv',
        'sigaction', 'sigprocmask', 'pthread_sigmask', 'stat', 'fstat',
        'lstat', 'access', 'open64_nocancel', 'gi_open64_nocancel',
        'checked_request2size', 'new_exitfn', 'internal_atexit',
        'lambda', 'version'
    ]
    if name.lower() in [s.lower() for s in system_overhead]:
        return False
    
    # Filter out compiler-generated target/thunk functions
    # Matches patterns like targ15680, targ12e60, targ1b3c0, etc.
    if name.startswith('targ') and len(name) > 4:
        # Check if the rest contains only hex characters (0-9, a-f)
        suffix = name[4:].lower()
        if all(c in '0123456789abcdef' for c in suffix):
            return False
    
    # Filter out other compiler artifacts
    compiler_artifacts = [
        'thunk', '_thunk', 'stub', '_stub', 'trampoline',
        'plt', '_plt', 'got', '_got', 'veneer'
    ]
    if any(artifact in name for artifact in compiler_artifacts):
        return False
    
    return True


def clean_function_name(raw_name):
    """
    Clean up function names for better readability in Perfetto.
    Remove mangling, templates, and other noise.
    """
    name = str(raw_name).strip()
    
    # Handle C++ mangled names
    if name.startswith('_Z'):
        # Try to extract class::function patterns
        if '::' in name:
            parts = name.split('::')
            # Take the last meaningful part
            name = parts[-1] if len(parts[-1]) > 3 else '::'.join(parts[-2:])
        else:
            # For heavily mangled names, try to make them more readable
            # But still keep some identifying information
            name = f"mangled_{name[2:22]}"  # Skip _Z prefix, take next 20 chars
    
    # Clean up template and operator syntax
    name = name.replace('<', '_').replace('>', '_')
    name = name.replace('(', '').replace(')', '')  # Remove parentheses completely
    name = name.replace('[', '_').replace(']', '_')
    name = name.replace(' ', '_')
    name = name.replace(',', '_')
    name = name.replace('::', '_')
    
    # Remove multiple underscores
    while '__' in name:
        name = name.replace('__', '_')
    
    # Remove trailing underscores
    name = name.strip('_')
    
    # Limit length for readability
    if len(name) > 60:
        name = name[:57] + "..."
    
    return name if name else "filtered_function"


def get_function_call_stack(cct_df, node_id):
    """
    Traverses the CCT from a given node up to the root and returns only the function call stack.
    Filters out line numbers, loops, and other non-function contexts.
    """
    function_stack = []
    curr_id = node_id
    
    # Handle special cases for missing CCT IDs
    if curr_id not in cct_df.index:
        if curr_id == 0:
            return []  # Root context - no function stack
        else:
            return []  # Unknown context
    
    # Build the complete path first
    path = []
    while curr_id in cct_df.index:
        path.append(curr_id)
        parent_id = cct_df.loc[curr_id]['parent']
        if pd.isna(parent_id) or parent_id in path:
            break
        curr_id = int(parent_id)
    
    # Filter to only include functions and entry points
    for cct_id in reversed(path):  # From root to leaf
        try:
            cct_node = cct_df.loc[cct_id]
            node_type = cct_node.get('type', 'unknown')
            
            # Only include functions and entry points in the call stack
            if node_type in ['function', 'entry']:
                function_stack.append(cct_id)
        except KeyError:
            continue
    
    return function_stack


def is_computational_thread(tid, traces_df, cct_df, functions_df):
    """
    Check if a thread contains meaningful computational work vs just infrastructure.
    """
    if tid not in traces_df['profile_id'].values:
        return False
    
    profile_traces = traces_df[traces_df['profile_id'] == tid]
    
    # Collect unique function names for this thread
    unique_functions = set()
    
    for _, row in profile_traces.iterrows():
        cct_id = int(row['cct_id'])
        function_stack = get_function_call_stack(cct_df, cct_id)
        
        for func_cct_id in function_stack:
            try:
                cct_node = cct_df.loc[func_cct_id]
                node_type = cct_node.get('type', 'unknown')
                
                if node_type == 'function' and pd.notna(cct_node['name']):
                    func_id = int(cct_node['name'])
                    if functions_df is not None and func_id in functions_df.index:
                        raw_name = str(functions_df.loc[func_id]['name'])
                        if is_meaningful_function(raw_name):
                            unique_functions.add(raw_name.lower())
            except KeyError:
                continue
    
    # Filter out threads with only infrastructure functions
    infrastructure_only = [
        'gpu_context', 'blas_thread_server', 'blast_thread_server',
        'pthread_create', 'thread_start', 'worker_thread', 'background_thread',
        'cleanup', 'monitor', 'watchdog'
    ]
    
    # Remove infrastructure functions
    computational_functions = unique_functions - set(infrastructure_only)
    
    # Thread is computational if it has:
    # 1. More than 10 unique functions, OR
    # 2. At least 3 non-infrastructure functions
    return len(unique_functions) > 10 or len(computational_functions) >= 3


def convert_to_perfetto(query_api, profiles_df, cct_df, traces_df):
    """
    Converts HPCToolkit dataframes to a Perfetto trace event list.
    """
    perfetto_trace = []

    # Get the functions table properly
    functions_df = query_api._functions if hasattr(query_api, '_functions') and not query_api._functions.empty else None

    # First pass: collect which threads will actually have trace events
    # Each thread becomes its own top-level process for better visibility
    # Also filter out infrastructure-only threads
    active_threads = set()
    
    if not traces_df.empty:
        # Pre-process to find which profiles will generate meaningful events
        total_threads = 0
        computational_threads = 0
        
        for profile_id in traces_df['profile_id'].unique():
            if profile_id in profiles_df.index:
                profile_row = profiles_df.loc[profile_id]
                tid = int(profile_id)
                total_threads += 1
                
                # Filter out infrastructure-only threads
                if is_computational_thread(tid, traces_df, cct_df, functions_df):
                    # Each thread becomes its own process (pid = tid)
                    # This flattens the hierarchy for better Perfetto navigation
                    pid = tid
                    active_threads.add((pid, tid))
                    computational_threads += 1
        
        print(f"Filtered threads: {computational_threads} computational out of {total_threads} total")
    
    print(f"Found {len(active_threads)} computational threads (each as top-level process).")
    
    # --- Create Metadata: each thread as its own top-level process ---
    for pid, tid in active_threads:
        if tid in profiles_df.index:
            row = profiles_df.loc[tid]
            
            # Determine thread type based on profile data
            thread_name = "Thread"
            
            # Check if this is a GPU stream/context profile
            gpustream = row.get('gpustream', '<NA>')
            gpucontext = row.get('gpucontext', '<NA>')
            gpudevice = row.get('gpudevice', '<NA>')
            
            # Determine profile type based on HPCToolkit's schema
            if pd.notna(gpustream) and gpustream != '<NA>':
                # This is a GPU stream
                thread_name = f"GPU Stream {gpustream}"
                if pd.notna(gpudevice) and gpudevice != '<NA>':
                    thread_name += f" (Device {gpudevice})"
            elif pd.notna(gpucontext) and gpucontext != '<NA>':
                # This is a GPU context
                thread_name = f"GPU Context {gpucontext}"
                if pd.notna(gpudevice) and gpudevice != '<NA>':
                    thread_name += f" (Device {gpudevice})"
            else:
                # This is a CPU thread
                thread_id = row.get('thread', tid)
                core_id = row.get('core', '<NA>')
                
                if tid in traces_df['profile_id'].values:
                    profile_traces = traces_df[traces_df['profile_id'] == tid]
                    num_events = len(profile_traces)
                    
                    # Classify CPU threads based on characteristics
                    if thread_id == 0:
                        thread_name = "Main Thread"
                    elif num_events > 1000:
                        thread_name = f"Worker Thread {thread_id}"
                    else:
                        thread_name = f"Thread {thread_id}"
                    
                    if pd.notna(core_id) and core_id != '<NA>':
                        thread_name += f" (Core {core_id})"
                else:
                    thread_name = f"Thread {thread_id}"
            
            process_name = f"{thread_name} (Profile {tid})"

            # Each thread becomes its own process in Perfetto
            perfetto_trace.append({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "ts": 0,
                "args": { "name": process_name }
            })
            perfetto_trace.append({
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "ts": 0,
                "args": { "sort_index": pid }
            })
            
            # Create a single thread within each process
            perfetto_trace.append({
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "ts": 0,
                "args": { "name": "Main" }
            })
            perfetto_trace.append({
                "name": "thread_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "ts": 0,
                "args": { "sort_index": 0 }
            })

    print(f"Created {len(perfetto_trace)} metadata events for active processes and threads only.")

    # --- Create Trace Events with full call stack hierarchy ---
    if not traces_df.empty:
        # --- Normalize timestamps to start from 0 and convert to microseconds ---
        min_ts = traces_df['start_timestamp'].min()
        traces_df = traces_df.copy()  # Avoid modifying the original dataframe
        traces_df['start_timestamp'] = ((traces_df['start_timestamp'] - min_ts) // 1000).astype(int)
        traces_df['end_timestamp'] = ((traces_df['end_timestamp'] - min_ts) // 1000).astype(int)

        # Group traces by thread/profile to process each one independently
        # Only process computational threads
        computational_thread_ids = {tid for pid, tid in active_threads}
        traces_by_tid = traces_df[traces_df['profile_id'].isin(computational_thread_ids)].groupby('profile_id')
        total_events = 0
        filtered_functions = 0

        profile_count = 0
        for tid, group in traces_by_tid:
            tid = int(tid)
            profile_count += 1
            
            # Find pid for this tid using the same logic as metadata creation
            try:
                profile_row = profiles_df.loc[tid]
                # Each thread is its own process (flattened hierarchy)
                pid = tid
                    
                profile_events_added = 0
                    
            except KeyError:
                continue

            active_stack = []  # The stack of currently open CCT nodes for this thread
            
            # Sort events by time to process them in order
            sorted_group = group.sort_values(by='start_timestamp')

            for _, row in sorted_group.iterrows():
                start_ts = int(row['start_timestamp'])
                end_ts = int(row['end_timestamp'])
                duration = end_ts - start_ts
                
                # Get the function call stack from the CCT
                cct_id = int(row['cct_id'])
                new_stack = get_function_call_stack(cct_df, cct_id)
                
                # Find how much of the new stack is common with the old one
                common_depth = 0
                while (common_depth < len(active_stack) and
                       common_depth < len(new_stack) and
                       active_stack[common_depth] == new_stack[common_depth]):
                    common_depth += 1
                
                # 1. Close frames that are no longer on the stack (from leaf to common ancestor)
                for i in range(len(active_stack) - 1, common_depth - 1, -1):
                    cct_id_to_close = active_stack[i]
                    # We don't need the name for an "E" event, but it's good practice
                    perfetto_trace.append({"ph": "E", "ts": start_ts, "pid": pid, "tid": tid})
                    total_events += 1
                    profile_events_added += 1

                # 2. Open new frames on the stack (from common ancestor to leaf)
                # Collect valid events first
                valid_frames = []
                for i in range(common_depth, len(new_stack)):
                    cct_id_to_open = new_stack[i]
                    
                    event_name = None
                    category = "function"
                    event_args = {"cct_id": int(cct_id_to_open)}
                    
                    try:
                        cct_node = cct_df.loc[cct_id_to_open]
                        node_type = cct_node.get('type', 'unknown')
                        
                        if node_type == 'function' and pd.notna(cct_node['name']):
                            func_id = int(cct_node['name'])
                            if functions_df is not None and func_id in functions_df.index:
                                # Check if this is a meaningful function
                                raw_name = str(functions_df.loc[func_id]['name'])
                                
                                if not is_meaningful_function(raw_name):
                                    filtered_functions += 1
                                    continue  # Skip unknown/meaningless functions
                                
                                # Clean up function names for better readability
                                event_name = clean_function_name(raw_name)
                                
                                # Add source file information if available
                                func_data = functions_df.loc[func_id]
                                if 'file_id' in func_data and pd.notna(func_data['file_id']):
                                    event_args["source"] = f"file_{int(func_data['file_id'])}"
                                if 'line' in func_data and pd.notna(func_data['line']):
                                    event_args["line"] = int(func_data['line'])
                            else:
                                continue  # Skip functions without names
                            category = "function"
                        elif node_type == 'entry':
                            # Skip entry points - they're just execution context markers
                            continue
                        else:
                            # Skip non-function nodes
                            continue
                    except KeyError:
                        # Skip unknown contexts
                        continue
                    
                    if event_name:
                        valid_frames.append({
                            "name": str(event_name), 
                            "ph": "B", 
                            "ts": start_ts, 
                            "pid": pid, 
                            "tid": tid,
                            "cat": category,
                            "args": event_args
                        })
                
                # Only add events if we have valid frames after filtering
                if valid_frames:
                    for frame in valid_frames:
                        perfetto_trace.append(frame)
                        total_events += 1
                        profile_events_added += 1
                    active_stack = new_stack
                else:
                    # If no valid frames, keep the previous stack state
                    pass

            # At the very end for this thread, close all remaining open frames
            if not sorted_group.empty:
                last_ts = int(sorted_group.iloc[-1]['end_timestamp'])
                for _ in range(len(active_stack)):
                    perfetto_trace.append({"ph": "E", "ts": last_ts, "pid": pid, "tid": tid})
                    total_events += 1
                    profile_events_added += 1

        print(f"Added {total_events} begin/end trace events from {profile_count} profiles.")
        print(f"Filtered out {filtered_functions} unknown/meaningless functions.")

    return {"traceEvents": perfetto_trace}


if __name__ == "__main__":
    main() 