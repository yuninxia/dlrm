#!/usr/bin/env python3
"""
DLRM CPU+GPU Performance Analysis Tool - Management Enhanced Version
Specialized performance analysis for DLRM workloads
Enhanced based on expert recommendations
"""

import hatchet as ht
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Try to import matplotlib, disable visualization if it fails
try:
    import matplotlib.pyplot as plt
    import matplotlib
    # Set font for better compatibility
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # Use simple style to avoid seaborn dependency
    plt.style.use('default')
    VISUALIZATION_ENABLED = True
except ImportError:
    print("⚠️  matplotlib not installed, will skip chart generation")
    VISUALIZATION_ENABLED = False

PLOT_DIR = Path("plots")
REPORTS_DIR = Path("reports") 
PLOT_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# 全局变量存储关键指标用于ROI计算
GLOBAL_METRICS = {}

def load_hpctoolkit_database(db_path):
    """Load HPCToolkit database"""
    try:
        print(f"Loading {db_path}...")
        gf = ht.GraphFrame.from_hpctoolkit_latest(db_path)
        print("✓ Database loaded successfully")
        return gf
    except Exception as e:
        print(f"✗ Loading failed: {e}")
        return None

def plot_cpu_gpu_overview(gf):
    """Management Core Chart 1: CPU vs GPU Time Distribution Pie Chart"""
    if not VISUALIZATION_ENABLED:
        print("⚠️  Visualization disabled, skipping CPU vs GPU chart")
        return
        
    print("\n📊 Generating management chart 1/4: CPU vs GPU time distribution...")
    
    df = gf.dataframe
    
    # Find actual metrics
    cpu_time_col = next((col for col in df.columns if 'time (inc)' in col), None)
    gpu_total_col = next((col for col in df.columns if 'gpuop (inc)' in col), None)
    
    if not cpu_time_col or not gpu_total_col:
        print("⚠️  Missing key time metrics, skipping CPU vs GPU chart")
        return
    
    cpu_time = df[cpu_time_col].sum()
    gpu_time = df[gpu_total_col].sum()
    
    # Calculate GPU utilization percentage
    total_time = cpu_time + gpu_time
    gpu_utilization = (gpu_time / total_time * 100) if total_time > 0 else 0
    
    # Store key metrics
    GLOBAL_METRICS.update({
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'gpu_utilization_pct': gpu_utilization
    })
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = ['CPU Time', 'GPU Time']
    sizes = [cpu_time, gpu_time]
    colors = ['#ff9999', '#66b3ff']
    
    # Highlight GPU underutilization issues
    if gpu_utilization < 5:
        colors = ['#ff6b6b', '#ffd93d']  # Warning colors
        title_color = 'red'
        title_suffix = f" - ⚠️ GPU Severely Underutilized ({gpu_utilization:.2f}%)"
    elif gpu_utilization < 20:
        title_color = 'orange'
        title_suffix = f" - ⚠️ GPU Utilization Low ({gpu_utilization:.2f}%)"
    else:
        title_color = 'green'
        title_suffix = f" - ✅ GPU Utilization Reasonable ({gpu_utilization:.2f}%)"
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                     autopct='%1.1f%%', startangle=90,
                                     explode=(0.05, 0.05))
    
    # Beautify text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_weight('bold')
    
    ax.set_title(f'CPU vs GPU Time Distribution{title_suffix}', 
                fontsize=16, fontweight='bold', color=title_color, pad=20)
    
    # Add value annotations
    plt.figtext(0.02, 0.02, f'CPU Time: {cpu_time:.3f}s  |  GPU Time: {gpu_time:.3f}s', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'cpu_gpu_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ CPU vs GPU pie chart saved: {PLOT_DIR / 'cpu_gpu_overview.png'}")

def plot_gpu_time_breakdown(gf):
    """Management Core Chart 2: GPU Time Breakdown Bar Chart"""
    if not VISUALIZATION_ENABLED:
        print("⚠️  Visualization disabled, skipping GPU breakdown chart")
        return
        
    print("\n📊 Generating management chart 2/4: GPU time breakdown...")
    
    df = gf.dataframe
    
    # Find GPU breakdown metrics
    gpu_kernel_col = next((col for col in df.columns if 'gker (inc)' in col), None)
    gpu_copy_col = next((col for col in df.columns if 'gxcopy (inc)' in col), None)
    
    if not gpu_kernel_col or not gpu_copy_col:
        print("⚠️  Missing GPU breakdown metrics, skipping GPU breakdown chart")
        return
    
    kernel_time = df[gpu_kernel_col].sum()
    copy_time = df[gpu_copy_col].sum()
    total_gpu = kernel_time + copy_time
    
    # Calculate percentages
    kernel_pct = (kernel_time / total_gpu * 100) if total_gpu > 0 else 0
    copy_pct = (copy_time / total_gpu * 100) if total_gpu > 0 else 0
    
    # Store key metrics
    GLOBAL_METRICS.update({
        'kernel_time': kernel_time,
        'copy_time': copy_time,
        'copy_percentage': copy_pct
    })
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['GPU Kernel Execution', 'GPU Data Transfer']
    values = [kernel_time, copy_time]
    percentages = [kernel_pct, copy_pct]
    
    # Choose colors based on data transfer percentage
    if copy_pct > 60:
        colors = ['#2ecc71', '#e74c3c']  # Green compute, red transfer (problem)
        title_suffix = f" - Data Transfer Dominates ({copy_pct:.1f}%)"
        title_color = 'red'
    elif copy_pct > 30:
        colors = ['#2ecc71', '#f39c12']  # Green compute, orange transfer (caution)
        title_suffix = f" - Data Transfer High ({copy_pct:.1f}%)"
        title_color = 'orange'
    else:
        colors = ['#2ecc71', '#3498db']  # Green compute, blue transfer (normal)
        title_suffix = f" - Compute Dominates ({copy_pct:.1f}%)"
        title_color = 'green'
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    
    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}s\n({pct:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'GPU Time Composition Analysis{title_suffix}', 
                fontsize=16, fontweight='bold', color=title_color, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'gpu_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ GPU breakdown bar chart saved: {PLOT_DIR / 'gpu_breakdown.png'}")

def plot_gpu_stall_analysis(gf):
    """Management Core Chart 3: GPU Stall Analysis Pie Chart"""
    if not VISUALIZATION_ENABLED:
        print("⚠️  Visualization disabled, skipping GPU stall analysis chart")
        return
        
    print("\n📊 Generating management chart 3/4: GPU stall analysis...")
    
    df = gf.dataframe
    
    # Find stall related metrics
    stall_cols = [col for col in df.columns if 'gins:stl_' in col and col != 'gins:stl_any (inc)']
    
    if not stall_cols:
        print("⚠️  GPU stall metrics not found, skipping stall analysis chart")
        return
    
    # Calculate each stall type
    stall_data = {}
    total_stall = 0
    
    for col in stall_cols:
        stall_value = df[col].sum()
        if stall_value > 0:
            # Simplify stall type names
            stall_type = col.replace('gins:stl_', '').replace(' (inc)', '')
            stall_data[stall_type] = stall_value
            total_stall += stall_value
    
    if total_stall == 0:
        print("⚠️  No valid GPU stall data detected")
        return
    
    # Only show stall types with >5% contribution
    significant_stalls = {k: v for k, v in stall_data.items() 
                         if (v/total_stall*100) > 5}
    
    if not significant_stalls:
        print("⚠️  No significant GPU stall types found")
        return
    
    # Store key metrics
    top_stall = max(significant_stalls.items(), key=lambda x: x[1])
    GLOBAL_METRICS.update({
        'total_stall_cycles': total_stall,
        'dominant_stall_type': top_stall[0],
        'dominant_stall_pct': top_stall[1]/total_stall*100
    })
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = list(significant_stalls.keys())
    sizes = list(significant_stalls.values())
    
    # Use highlighting colors for the largest stall type
    colors = plt.cm.Set3(np.arange(len(labels)))
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90,
                                     explode=[0.1 if i == 0 else 0 for i in range(len(labels))])
    
    # Beautify text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    # Title with suggestions based on dominant stall type
    dominant_pct = GLOBAL_METRICS['dominant_stall_pct']
    dominant_type = GLOBAL_METRICS['dominant_stall_type']
    
    if 'gmem' in dominant_type and dominant_pct > 40:
        title_suffix = f" - Memory Access Bottleneck ({dominant_pct:.1f}%)"
        suggestion = "Recommendation: Optimize memory access patterns, use shared memory"
    elif 'idep' in dominant_type and dominant_pct > 30:
        title_suffix = f" - Instruction Dependency Bottleneck ({dominant_pct:.1f}%)"
        suggestion = "Recommendation: Increase parallelism, kernel fusion optimization"
    else:
        title_suffix = f" - Main Stall: {dominant_type} ({dominant_pct:.1f}%)"
        suggestion = "Recommendation: Target optimization for main stall type"
    
    ax.set_title(f'GPU Stall Type Analysis{title_suffix}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add optimization suggestion
    plt.figtext(0.02, 0.02, suggestion, fontsize=10, style='italic', color='blue')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'gpu_stall_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ GPU stall analysis pie chart saved: {PLOT_DIR / 'gpu_stall_analysis.png'}")

def plot_kernel_launch_efficiency(gf):
    """Management Core Chart 4: Kernel Launch Efficiency Analysis"""
    if not VISUALIZATION_ENABLED:
        print("⚠️  Visualization disabled, skipping kernel launch efficiency chart")
        return
        
    print("\n📊 Generating management chart 4/4: Kernel launch efficiency...")
    
    df = gf.dataframe
    
    # Find kernel related metrics
    kernel_count_col = next((col for col in df.columns if 'gker:count (inc)' in col), None)
    kernel_time_col = next((col for col in df.columns if 'gker (inc)' in col), None)
    
    if not kernel_count_col or not kernel_time_col:
        print("⚠️  Missing kernel metrics, skipping kernel efficiency chart")
        return
    
    total_kernels = df[kernel_count_col].sum()
    total_kernel_time = df[kernel_time_col].sum()
    
    if total_kernels == 0:
        print("⚠️  No kernel launches detected")
        return
    
    # Calculate average kernel time (convert to microseconds)
    avg_kernel_time_us = (total_kernel_time / total_kernels) * 1e6
    
    # Estimate kernel launch overhead (assume 5 microseconds per launch)
    kernel_launch_overhead_us = 5  # Based on NVIDIA documentation
    estimated_overhead_s = (total_kernels * kernel_launch_overhead_us) / 1e6
    overhead_percentage = (estimated_overhead_s / total_kernel_time * 100) if total_kernel_time > 0 else 0
    
    # Store key metrics
    GLOBAL_METRICS.update({
        'total_kernels': total_kernels,
        'avg_kernel_time_us': avg_kernel_time_us,
        'kernel_overhead_pct': overhead_percentage
    })
    
    # Create histogram showing kernel time distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left chart: Kernel time distribution histogram
    kernel_times_per_func = df[kernel_time_col] / df[kernel_count_col].replace(0, 1)
    valid_times = kernel_times_per_func[kernel_times_per_func > 0] * 1e6  # Convert to microseconds
    
    if len(valid_times) > 0:
        ax1.hist(valid_times, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.axvline(avg_kernel_time_us, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_kernel_time_us:.1f}μs')
        ax1.axvline(kernel_launch_overhead_us, color='orange', linestyle='--', linewidth=2,
                   label=f'Launch overhead: {kernel_launch_overhead_us}μs')
        
        ax1.set_xlabel('Kernel Average Execution Time (μs)')
        ax1.set_ylabel('Number of Kernels')
        ax1.set_title('Kernel Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Right chart: Efficiency analysis bar chart
    categories = ['Compute Time', 'Launch Overhead']
    values = [total_kernel_time - estimated_overhead_s, estimated_overhead_s]
    colors = ['green', 'red'] if overhead_percentage > 20 else ['green', 'orange']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.8)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{val:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Time (seconds)')
    
    # Set title and suggestions based on overhead percentage
    if overhead_percentage > 30:
        title_color = 'red'
        title_suffix = f" - Launch Overhead Too High ({overhead_percentage:.1f}%)"
        suggestion = f"Recommendation: Kernel fusion optimization, reduce {total_kernels:,.0f} launches"
    elif overhead_percentage > 15:
        title_color = 'orange' 
        title_suffix = f" - Launch Overhead High ({overhead_percentage:.1f}%)"
        suggestion = "Recommendation: Consider kernel fusion and batch processing optimization"
    else:
        title_color = 'green'
        title_suffix = f" - Launch Efficiency Reasonable ({overhead_percentage:.1f}%)"
        suggestion = "Kernel launch efficiency within acceptable range"
    
    ax2.set_title(f'Kernel Launch Efficiency{title_suffix}', color=title_color)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add overall title and suggestions
    fig.suptitle(f'Kernel Launch Efficiency Analysis - Total {total_kernels:,.0f} launches', 
                fontsize=16, fontweight='bold')
    plt.figtext(0.02, 0.02, suggestion, fontsize=10, style='italic', color='blue')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'kernel_launch_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Kernel launch efficiency chart saved: {PLOT_DIR / 'kernel_launch_efficiency.png'}")

def plot_transfer_size_distribution(gf):
    """Additional Chart 1: Memory Transfer Size Distribution Analysis"""
    if not VISUALIZATION_ENABLED:
        print("⚠️  Visualization disabled, skipping transfer size distribution chart")
        return
        
    print("\n📊 Generating additional chart 1/3: Memory transfer size distribution...")
    
    df = gf.dataframe
    
    # Find transfer related metrics
    h2d_col = next((col for col in df.columns if 'gxcopy:h2d' in col and 'inc' in col), None)
    d2h_col = next((col for col in df.columns if 'gxcopy:d2h' in col and 'inc' in col), None)
    copy_count_col = next((col for col in df.columns if 'gxcopy:count (inc)' in col), None)
    
    if not h2d_col or not copy_count_col:
        print("⚠️  Missing transfer metrics, skipping transfer size distribution chart")
        return
    
    # Calculate transfer sizes per function
    h2d_sizes = []
    d2h_sizes = []
    
    for idx, row in df.iterrows():
        if row[copy_count_col] > 0:
            # Calculate average transfer size per function
            avg_h2d_size = row[h2d_col] / row[copy_count_col] if row[h2d_col] > 0 else 0
            if avg_h2d_size > 0:
                h2d_sizes.extend([avg_h2d_size] * int(row[copy_count_col]))
            
            if d2h_col and row[d2h_col] > 0:
                avg_d2h_size = row[d2h_col] / row[copy_count_col]
                if avg_d2h_size > 0:
                    d2h_sizes.extend([avg_d2h_size] * int(row[copy_count_col]))
    
    if not h2d_sizes:
        print("⚠️  No valid transfer size data found")
        return
    
    # Convert to KB for better readability
    h2d_sizes_kb = [size / 1024 for size in h2d_sizes]
    d2h_sizes_kb = [size / 1024 for size in d2h_sizes] if d2h_sizes else []
    
    # Store key metrics
    total_transfers = len(h2d_sizes)
    avg_transfer_size_kb = np.mean(h2d_sizes_kb)
    median_transfer_size_kb = np.median(h2d_sizes_kb)
    
    GLOBAL_METRICS.update({
        'total_transfers': total_transfers,
        'avg_transfer_size_kb': avg_transfer_size_kb,
        'median_transfer_size_kb': median_transfer_size_kb
    })
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left chart: H2D transfer size distribution
    ax1.hist(h2d_sizes_kb, bins=50, color='lightblue', alpha=0.7, edgecolor='black')
    ax1.axvline(avg_transfer_size_kb, color='red', linestyle='--', linewidth=2, 
               label=f'Average: {avg_transfer_size_kb:.1f} KB')
    ax1.axvline(median_transfer_size_kb, color='orange', linestyle='--', linewidth=2,
               label=f'Median: {median_transfer_size_kb:.1f} KB')
    
    # Add optimal transfer size guidelines
    ax1.axvline(1024, color='green', linestyle=':', linewidth=2, alpha=0.7,
               label='Optimal: 1MB+')
    
    ax1.set_xlabel('Transfer Size (KB)')
    ax1.set_ylabel('Number of Transfers')
    ax1.set_title('H2D Transfer Size Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visibility
    
    # Right chart: Transfer efficiency analysis
    small_transfers = sum(1 for size in h2d_sizes_kb if size < 64)  # < 64KB
    medium_transfers = sum(1 for size in h2d_sizes_kb if 64 <= size < 1024)  # 64KB-1MB
    large_transfers = sum(1 for size in h2d_sizes_kb if size >= 1024)  # >= 1MB
    
    categories = ['Small\n(<64KB)', 'Medium\n(64KB-1MB)', 'Large\n(≥1MB)']
    counts = [small_transfers, medium_transfers, large_transfers]
    percentages = [count/total_transfers*100 for count in counts]
    
    # Color code based on efficiency
    colors = ['red', 'orange', 'green']  # Red for inefficient small transfers
    
    bars = ax2.bar(categories, counts, color=colors, alpha=0.8)
    
    # Add percentage labels
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{pct:.1f}%\n({int(height):,})', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Number of Transfers')
    ax2.set_title('Transfer Size Categories')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Analysis and recommendations
    fragmentation_percentage = (small_transfers / total_transfers) * 100
    
    if fragmentation_percentage > 50:
        title_color = 'red'
        title_suffix = f" - High Fragmentation ({fragmentation_percentage:.1f}%)"
        suggestion = "Critical: Batch small transfers into larger chunks (target >1MB)"
    elif fragmentation_percentage > 25:
        title_color = 'orange'
        title_suffix = f" - Moderate Fragmentation ({fragmentation_percentage:.1f}%)"
        suggestion = "Recommended: Consolidate transfers to improve bandwidth utilization"
    else:
        title_color = 'green'
        title_suffix = f" - Good Transfer Sizes ({fragmentation_percentage:.1f}% small)"
        suggestion = "Transfer sizes are reasonably optimized"
    
    # Add overall title and analysis
    fig.suptitle(f'Memory Transfer Size Analysis{title_suffix}', 
                fontsize=16, fontweight='bold', color=title_color)
    plt.figtext(0.02, 0.02, 
                f'{suggestion} | Total: {total_transfers:,} transfers, Avg: {avg_transfer_size_kb:.1f}KB', 
                fontsize=10, style='italic', color='blue')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'transfer_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Transfer size distribution chart saved: {PLOT_DIR / 'transfer_size_distribution.png'}")

def plot_gpu_occupancy_timeline(gf):
    """Additional Chart 2: GPU Occupancy Timeline Analysis"""
    if not VISUALIZATION_ENABLED:
        print("⚠️  Visualization disabled, skipping GPU occupancy timeline chart")
        return
        
    print("\n📊 Generating additional chart 2/3: GPU occupancy timeline...")
    
    df = gf.dataframe
    
    # Find GPU occupancy and utilization related metrics
    gpu_total_col = next((col for col in df.columns if 'gpuop (inc)' in col), None)
    cpu_time_col = next((col for col in df.columns if 'time (inc)' in col), None)
    gpu_kernel_col = next((col for col in df.columns if 'gker (inc)' in col), None)
    gpu_copy_col = next((col for col in df.columns if 'gxcopy (inc)' in col), None)
    
    if not gpu_total_col or not cpu_time_col:
        print("⚠️  Missing occupancy metrics, skipping GPU occupancy timeline chart")
        return
    
    # Get functions with significant GPU activity
    gpu_active_funcs = df[(df[gpu_total_col] > 0) | (df[gpu_kernel_col] > 0) | (df[gpu_copy_col] > 0)].copy()
    
    if len(gpu_active_funcs) == 0:
        print("⚠️  No GPU-active functions found")
        return
    
    # Sort by CPU time to create a rough timeline
    gpu_active_funcs = gpu_active_funcs.sort_values(by=cpu_time_col, ascending=False).head(20)
    
    # Calculate occupancy metrics for each function
    gpu_active_funcs['gpu_utilization_pct'] = (gpu_active_funcs[gpu_total_col] / gpu_active_funcs[cpu_time_col] * 100).fillna(0)
    gpu_active_funcs['kernel_ratio'] = (gpu_active_funcs[gpu_kernel_col] / gpu_active_funcs[gpu_total_col] * 100).fillna(0)
    gpu_active_funcs['copy_ratio'] = (gpu_active_funcs[gpu_copy_col] / gpu_active_funcs[gpu_total_col] * 100).fillna(0)
    
    # Store key metrics
    avg_utilization = gpu_active_funcs['gpu_utilization_pct'].mean()
    max_utilization = gpu_active_funcs['gpu_utilization_pct'].max()
    idle_functions = len(df) - len(gpu_active_funcs)
    
    GLOBAL_METRICS.update({
        'avg_gpu_utilization': avg_utilization,
        'max_gpu_utilization': max_utilization,
        'idle_functions': idle_functions,
        'active_gpu_functions': len(gpu_active_funcs)
    })
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Top chart: GPU utilization timeline
    x_positions = range(len(gpu_active_funcs))
    function_names = [name[:30] + "..." if len(name) > 30 else name for name in gpu_active_funcs['name']]
    
    # Color code based on utilization level
    colors = []
    for util in gpu_active_funcs['gpu_utilization_pct']:
        if util < 1:
            colors.append('red')      # Severely underutilized
        elif util < 10:
            colors.append('orange')   # Low utilization
        elif util < 50:
            colors.append('yellow')   # Moderate utilization  
        else:
            colors.append('green')    # Good utilization
    
    bars = ax1.bar(x_positions, gpu_active_funcs['gpu_utilization_pct'], color=colors, alpha=0.8)
    
    # Add average line
    ax1.axhline(y=avg_utilization, color='blue', linestyle='--', linewidth=2, 
               label=f'Average: {avg_utilization:.2f}%')
    
    # Add target line
    ax1.axhline(y=50, color='green', linestyle=':', linewidth=2, alpha=0.7,
               label='Target: 50%+')
    
    ax1.set_xlabel('Functions (Top 20 GPU-Active)')
    ax1.set_ylabel('GPU Utilization %')
    ax1.set_title('GPU Utilization by Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(x_positions[::2])  # Show every other label to avoid crowding
    ax1.set_xticklabels([function_names[i] for i in range(0, len(function_names), 2)], rotation=45, ha='right')
    
    # Bottom chart: Compute vs Transfer breakdown
    width = 0.8
    bottom_kernel = gpu_active_funcs['kernel_ratio']
    bottom_copy = gpu_active_funcs['copy_ratio']
    
    bars1 = ax2.bar(x_positions, bottom_kernel, width, label='Kernel Execution', color='lightblue', alpha=0.8)
    bars2 = ax2.bar(x_positions, bottom_copy, width, bottom=bottom_kernel, label='Data Transfer', color='lightcoral', alpha=0.8)
    
    ax2.set_xlabel('Functions (Top 20 GPU-Active)')
    ax2.set_ylabel('GPU Time Distribution %')
    ax2.set_title('GPU Time: Compute vs Transfer Breakdown')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(x_positions[::2])
    ax2.set_xticklabels([function_names[i] for i in range(0, len(function_names), 2)], rotation=45, ha='right')
    
    # Analysis and recommendations
    underutilized_funcs = sum(1 for util in gpu_active_funcs['gpu_utilization_pct'] if util < 10)
    underutilization_pct = (underutilized_funcs / len(gpu_active_funcs)) * 100
    
    if underutilization_pct > 75:
        title_color = 'red'
        title_suffix = f" - Severe Underutilization ({underutilization_pct:.0f}% functions <10%)"
        suggestion = "Critical: GPU sits idle most of the time - increase workload size or parallelism"
    elif underutilization_pct > 50:
        title_color = 'orange'
        title_suffix = f" - High Underutilization ({underutilization_pct:.0f}% functions <10%)"
        suggestion = "Important: Significant GPU idle time - optimize workload distribution"
    else:
        title_color = 'green'
        title_suffix = f" - Reasonable Utilization ({underutilization_pct:.0f}% functions <10%)"
        suggestion = "GPU utilization within acceptable range"
    
    # Add overall title and analysis
    fig.suptitle(f'GPU Occupancy Timeline Analysis{title_suffix}', 
                fontsize=16, fontweight='bold', color=title_color)
    plt.figtext(0.02, 0.01, 
                f'{suggestion} | Avg Utilization: {avg_utilization:.2f}%, Max: {max_utilization:.2f}%, {idle_functions:,} idle functions', 
                fontsize=10, style='italic', color='blue')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'gpu_occupancy_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ GPU occupancy timeline chart saved: {PLOT_DIR / 'gpu_occupancy_timeline.png'}")




def calculate_roi_estimates():
    """Calculate return on investment estimates"""
    print("\n💰 Calculating ROI estimates...")
    
    if not GLOBAL_METRICS:
        print("⚠️  Missing key metrics, cannot calculate ROI")
        return {}
    
    roi_data = {}
    
    # 1. GPU utilization improvement potential
    current_gpu_util = GLOBAL_METRICS.get('gpu_utilization_pct', 0)
    if current_gpu_util < 50:
        potential_speedup = 50 / max(current_gpu_util, 1)  # Assume 50% utilization achievable
        roi_data['gpu_utilization'] = {
            'current': f"{current_gpu_util:.2f}%",
            'target': "50%",
            'potential_speedup': f"{potential_speedup:.1f}x",
            'description': "Optimize CPU-GPU workload balance"
        }
    
    # 2. Data transfer optimization potential
    copy_pct = GLOBAL_METRICS.get('copy_percentage', 0)
    if copy_pct > 40:
        # Assume Unified Memory and batching can reduce 60% of transfer time
        transfer_reduction = 0.6
        speedup_from_transfer = 1 / (1 - copy_pct/100 * transfer_reduction)
        roi_data['data_transfer'] = {
            'current': f"{copy_pct:.1f}% GPU time on transfers",
            'target': f"{copy_pct * (1-transfer_reduction):.1f}% GPU time on transfers",
            'potential_speedup': f"{speedup_from_transfer:.1f}x",
            'description': "Unified Memory and batch transfer optimization"
        }
    
    # 3. Kernel launch optimization potential
    kernel_overhead = GLOBAL_METRICS.get('kernel_overhead_pct', 0)
    if kernel_overhead > 15:
        # Assume kernel fusion can reduce 80% of launch count
        launch_reduction = 0.8
        speedup_from_fusion = 1 / (1 - kernel_overhead/100 * launch_reduction)
        roi_data['kernel_fusion'] = {
            'current': f"{kernel_overhead:.1f}% time on launch overhead",
            'target': f"{kernel_overhead * (1-launch_reduction):.1f}% time on launch overhead",
            'potential_speedup': f"{speedup_from_fusion:.1f}x",
            'description': "Kernel fusion to reduce launch count"
        }
    
    # 4. Combined performance improvement estimate
    total_speedup = 1.0
    for optimization in roi_data.values():
        if 'potential_speedup' in optimization:
            speedup_val = float(optimization['potential_speedup'].replace('x', ''))
            total_speedup *= speedup_val
    
    roi_data['total_estimate'] = {
        'combined_speedup': f"{total_speedup:.1f}x",
        'description': "Expected performance improvement after combined optimization"
    }
    
    return roi_data

def generate_executive_summary():
    """Generate management executive summary"""
    print("\n📋 Generating management executive summary...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"""
# DLRM GPU Performance Analysis - Executive Summary

**Generated**: {timestamp}  
**Analysis Tool**: HPCToolkit + Hatchet  
**Scope**: CPU + GPU Unified Performance Profiling

## 🎯 Key Findings

### 1. GPU Utilization Status
- **Current GPU Utilization**: {GLOBAL_METRICS.get('gpu_utilization_pct', 0):.2f}%
- **ROI Issue**: GPU hardware idle most of the time
- **Direct Impact**: Low hardware ROI, high operational costs

### 2. Performance Bottleneck Analysis
- **Data Transfer Ratio**: {GLOBAL_METRICS.get('copy_percentage', 0):.1f}% of GPU time spent on data transfers
- **Main Stall Type**: {GLOBAL_METRICS.get('dominant_stall_type', 'N/A')} ({GLOBAL_METRICS.get('dominant_stall_pct', 0):.1f}%)
- **Kernel Launch Overhead**: {GLOBAL_METRICS.get('kernel_overhead_pct', 0):.1f}% ({GLOBAL_METRICS.get('total_kernels', 0):,.0f} launches)

### 3. Quantified Improvement Opportunities
"""
    
    # Add ROI estimates
    roi_data = calculate_roi_estimates()
    if roi_data:
        summary += "\n| Optimization Direction | Current State | Target State | Expected Improvement |\n"
        summary += "|------------------------|---------------|--------------|---------------------|\n"
        
        for opt_name, opt_data in roi_data.items():
            if opt_name != 'total_estimate':
                summary += f"| {opt_data['description']} | {opt_data['current']} | {opt_data['target']} | {opt_data['potential_speedup']} |\n"
        
        if 'total_estimate' in roi_data:
            summary += f"\n**🚀 Combined Optimization Expected**: {roi_data['total_estimate']['combined_speedup']} performance improvement\n"
    
    summary += f"""

## 💡 Specific Action Recommendations

### Immediate (1-2 weeks)
1. **Batch Data Transfers**: Reduce CPU-GPU transfer frequency
2. **Use Pinned Memory**: Improve transfer bandwidth
3. **Adjust Batch Size**: Increase GPU workload

### Medium-term (1-2 months)  
1. **Kernel Fusion**: Reduce {GLOBAL_METRICS.get('total_kernels', 0):,.0f} launch overhead
2. **Unified Memory**: Simplify memory management
3. **Async Execution**: CPU-GPU parallelization

### Long-term Strategy (3-6 months)
1. **Algorithm Optimization**: Target {GLOBAL_METRICS.get('dominant_stall_type', 'N/A')} stall optimization
2. **Hardware Upgrade Assessment**: Based on performance analysis data
3. **Automated Monitoring**: Integrate HPCToolkit into CI/CD

## 📊 Key Chart Explanations

1. **CPU vs GPU Time Distribution**: Shows GPU utilization issues
2. **GPU Time Composition**: Identifies data transfer bottlenecks  
3. **GPU Stall Analysis**: Identifies compute efficiency issues
4. **Kernel Launch Efficiency**: Quantifies launch overhead impact

---
*This report is based on HPCToolkit analysis*
"""
    
    # Save executive summary
    with open(REPORTS_DIR / 'executive_summary.md', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"✓ Management executive summary saved: {REPORTS_DIR / 'executive_summary.md'}")
    
    return summary

def generate_technical_appendix(gf):
    """Generate technical appendix (engineer detailed version)"""
    print("\nGenerating technical detailed report...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = gf.dataframe
    
    # 获取所有可用指标的统计
    metrics_summary = {}
    for col in df.columns:
        if col != 'name':
            metrics_summary[col] = {
                'sum': df[col].sum(),
                'mean': df[col].mean(), 
                'max': df[col].max(),
                'non_zero_count': (df[col] > 0).sum()
            }
    
    appendix = f"""
# DLRM GPU Performance Analysis - Technical Detailed Report

**Generated**: {timestamp}  
**Data Source**: HPCToolkit Database  
**Analysis Framework**: Hatchet + Pandas  
**Function Count**: {len(df)}  
**Metric Count**: {len(df.columns)-1}  

## 🔍 Complete Metric Statistics

### CPU Performance Metrics
"""
    
    # CPU metrics
    cpu_metrics = [col for col in df.columns if 'time' in col.lower() or 'cpu' in col.lower()]
    for metric in cpu_metrics:
        if metric in metrics_summary:
            stats = metrics_summary[metric]
            appendix += f"- **{metric}**: Total={stats['sum']:.3e}, Average={stats['mean']:.3e}, Max={stats['max']:.3e}\n"
    
    appendix += "\n### GPU Compute Metrics\n"
    gpu_compute_metrics = [col for col in df.columns if any(x in col for x in ['gker', 'gins', 'gpuop'])]
    for metric in gpu_compute_metrics:
        if metric in metrics_summary:
            stats = metrics_summary[metric]
            appendix += f"- **{metric}**: Total={stats['sum']:.3e}, Average={stats['mean']:.3e}, Active Functions={stats['non_zero_count']}\n"
    
    appendix += "\n### GPU Memory Transfer Metrics\n"
    gpu_memory_metrics = [col for col in df.columns if any(x in col for x in ['gxcopy', 'h2d', 'd2h'])]
    for metric in gpu_memory_metrics:
        if metric in metrics_summary:
            stats = metrics_summary[metric]
            if 'bytes' in metric or 'h2d' in metric or 'd2h' in metric:
                appendix += f"- **{metric}**: Total={stats['sum']:.0f} bytes ({stats['sum']/1e6:.1f} MB)\n"
            else:
                appendix += f"- **{metric}**: Total={stats['sum']:.3e}, Average={stats['mean']:.3e}\n"
    
    appendix += "\n### GPU Stall Metrics\n"
    stall_metrics = [col for col in df.columns if 'stl' in col]
    for metric in stall_metrics:
        if metric in metrics_summary:
            stats = metrics_summary[metric]
            appendix += f"- **{metric}**: Total={stats['sum']:.3e}, Active Functions={stats['non_zero_count']}\n"
    
    # Top hotspot functions analysis
    if 'time (inc)' in df.columns:
        appendix += "\n## 🔥 Top 10 Hotspot Functions\n\n"
        appendix += "| Rank | Function Name | CPU Time | GPU Time | Transfer Time |\n"
        appendix += "|------|--------|---------|---------|----------|\n"
        
        hot_functions = df.nlargest(10, 'time (inc)')
        for i, (idx, row) in enumerate(hot_functions.iterrows()):
            func_name = row['name'][:50] + "..." if len(row['name']) > 50 else row['name']
            cpu_time = row.get('time (inc)', 0)
            gpu_time = row.get('gpuop (inc)', 0) 
            copy_time = row.get('gxcopy (inc)', 0)
            appendix += f"| {i+1} | {func_name} | {cpu_time:.3f} | {gpu_time:.3f} | {copy_time:.3f} |\n"
    
    # Save key metrics to JSON
    metrics_json = {
        'timestamp': timestamp,
        'global_metrics': GLOBAL_METRICS,
        'top_functions': df.nlargest(5, 'time (inc)')[['name', 'time (inc)']].to_dict('records') if 'time (inc)' in df.columns else [],
        'optimization_recommendations': calculate_roi_estimates()
    }
    
    with open(REPORTS_DIR / 'metrics_data.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False, default=str)
    
    appendix += f"""
    
## 📊 Data Files

- **Plot Directory**: `{PLOT_DIR}/`
- **Original Data**: `{REPORTS_DIR}/metrics_data.json`
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
"""
    
    with open(REPORTS_DIR / 'technical_appendix.md', 'w', encoding='utf-8') as f:
        f.write(appendix)
    
    print(f"✓ Technical detailed report saved: {REPORTS_DIR / 'technical_appendix.md'}")

# ================== 保留原有分析函数 ==================

def enhanced_list_all_metrics(gf):
    """Enhanced metrics list - based on real HPCViewer metrics"""
    print("\n" + "="*60)
    print("🧮 Detailed Metrics Analysis (HPCViewer format)")
    print("="*60)
    
    df = gf.dataframe
    all_columns = sorted(df.columns)
    
    # New categorization method - based on HPCViewer metrics
    enhanced_categories = {
        '🕒 CPU Time Metrics': [col for col in all_columns if col.upper().startswith('CPUTIME')],
        '⚡ GPU Time Metrics (seconds)': [col for col in all_columns if any(x in col.upper() for x in ['GKER', 'GXCOPY', 'GPUOP'])],
        '🚀 GPU Instruction Count': [col for col in all_columns if col.upper().startswith('GINS') and 'STL' not in col.upper()],
        '⚠️  GPU Stall Details': [col for col in all_columns if 'STL' in col.upper() and col.upper().startswith('GINS')],
        '📡 Data Transfer Details': [col for col in all_columns if 'GXCOPY' in col.upper() and any(x in col.upper() for x in ['H2D', 'D2H', 'COUNT'])],
        '⚙️  GPU Kernel Details': [col for col in all_columns if col.upper().startswith('GKER') and 'SEC' not in col.upper()],
        '📊 GPU Sampling Metrics': [col for col in all_columns if col.upper().startswith('GSAMP')],
        '🎯 GPU Utilization': [col for col in all_columns if 'UTIL' in col.upper() or 'OCC' in col.upper()],
        '🧮 Other Metrics': [col for col in all_columns if col not in sum([
            [col for col in all_columns if col.upper().startswith('CPUTIME')],
            [col for col in all_columns if any(x in col.upper() for x in ['GKER', 'GXCOPY', 'GPUOP'])],
            [col for col in all_columns if col.upper().startswith('GINS') and 'STL' not in col.upper()],
            [col for col in all_columns if 'STL' in col.upper() and col.upper().startswith('GINS')],
            [col for col in all_columns if 'GXCOPY' in col.upper() and any(x in col.upper() for x in ['H2D', 'D2H', 'COUNT'])],
            [col for col in all_columns if col.upper().startswith('GKER') and 'SEC' not in col.upper()],
            [col for col in all_columns if col.upper().startswith('GSAMP')],
            [col for col in all_columns if 'UTIL' in col.upper() or 'OCC' in col.upper()]
        ], [])]
    }
    
    print(f"📈 Found {len(all_columns)} total metrics\n")
    
    for category, columns in enhanced_categories.items():
        if columns:
            print(f"{category} ({len(columns)} metrics):")
            for col in columns:
                # Display metric and its total value
                total_value = df[col].sum() if col != 'name' else len(df)
                if col == 'name':
                    print(f"    {col}: {total_value} functions")
                elif any(x in col.upper() for x in ['SEC', 'TIME']):
                    print(f"    {col}: {total_value:.3e} seconds")
                elif any(x in col.upper() for x in ['BYTES', 'H2D', 'D2H']):
                    print(f"    {col}: {total_value:.2e} bytes ({total_value/1e6:.1f} MB)")
                elif 'UTIL' in col.upper() or 'OCC' in col.upper():
                    avg_value = df[col].mean() if df[col].sum() > 0 else 0
                    print(f"    {col}: average {avg_value:.2f}%")
                else:
                    print(f"    {col}: {total_value:.2e}")
            print()

def add_derived_metrics(gf):
    """Add derived metrics (percentages etc.) - final corrected version"""
    print("\n" + "="*50)
    print("📊 Computing Derived Metrics - Final Corrected Version")
    print("="*50)
    
    df = gf.dataframe
    
    # Find key metrics
    time_col = next((col for col in df.columns if 'time' in col.lower()), None)
    copy_cols = [col for col in df.columns if 'gxcopy' in col.lower()]
    stall_cols = [col for col in df.columns if 'stl' in col.lower()]
    
    derived_metrics = []
    
    # Completely redesigned data transfer percentage calculation
    if time_col and copy_cols:
        gxcopy_time_col = next((col for col in copy_cols if 'h2d' not in col and 'd2h' not in col and 'count' not in col), None)
        if gxcopy_time_col:  # Use GPU transfer time instead of CPU time
            print(f"✓ Using GPU transfer time column for percentage calculation: {gxcopy_time_col}")
            for copy_col in copy_cols:
                if 'h2d' in copy_col or 'd2h' in copy_col or 'count' in copy_col:
                    pct_col = f"{copy_col.replace(' (inc)', '')}_pct"
                    # Use GPU transfer time as denominator
                    df[pct_col] = np.where(
                        df[gxcopy_time_col] > 1e-9,
                        np.clip(100 * df[copy_col] / df[gxcopy_time_col], 0, 1000),  # Limit max 1000%
                        0
                    )
                    derived_metrics.append(pct_col)
        else:
            print("⚠️  GPU transfer time column not found, skipping transfer percentage calculation")
    
    # GPU stall percentage maintains corrected calculation
    total_stall_col = next((col for col in df.columns if 'stl_any' in col and 'inc' in col), None)
    if total_stall_col and stall_cols:
        print(f"✓ Using correct stall calculation method (based on {total_stall_col})")
        for stall_col in stall_cols:
            if stall_col != total_stall_col and stall_col in df.columns:
                pct_col = f"{stall_col.replace(' (inc)', '')}_pct"
                df[pct_col] = np.where(df[total_stall_col] > 0,
                                     100 * df[stall_col] / df[total_stall_col], 0)
                derived_metrics.append(pct_col)
    
    print(f"✓ Added {len(derived_metrics)} derived metrics:")
    for metric in derived_metrics:
        print(f"    {metric}")
    
    return derived_metrics

# 在 add_advanced_derived_metrics 函数中，修改GPU利用率计算部分
def add_advanced_derived_metrics(gf):
    """Add advanced derived metrics - CPU/GPU ratios and bandwidth (corrected version)"""
    print("\n" + "="*50)
    print("🔬 Computing Advanced Derived Metrics (CPU/GPU ratios, bandwidth) - Corrected Version")
    print("="*50)
    
    df = gf.dataframe
    
    # Find actual column names
    time_col = next((col for col in df.columns if 'time (inc)' in col), None)
    h2d_col = next((col for col in df.columns if 'gxcopy:h2d' in col and 'inc' in col), None)
    d2h_col = next((col for col in df.columns if 'gxcopy:d2h' in col and 'inc' in col), None)
    
    advanced_metrics = []
    
    # ====  Build GPU time according to HPCToolkit manual Table 8.1  ==============
    GPU_TIME_COLS = ["gker (inc)", "gxcopy (inc)", "gsync (inc)", "gmem (inc)", "gmset (inc)"]
    available_gpu_cols = [col for col in GPU_TIME_COLS if col in df.columns]
    
    print(f"🔍 Found key columns:")
    print(f"  Time column: {time_col}")
    print(f"  Available GPU time columns: {available_gpu_cols}")
    print(f"  H2D transfer column: {h2d_col}")
    print(f"  D2H transfer column: {d2h_col}")
    
    # Build comprehensive GPU time
    if available_gpu_cols and time_col:
        # Ensure column names match dataframe exactly
        valid_gpu_cols = [col for col in available_gpu_cols if col in df.columns]
        df["gtime (inc)"] = df[valid_gpu_cols].sum(axis=1).fillna(0)
        
        # Debug information
        total_gpu_time = df["gtime (inc)"].sum()
        print(f"🔍 GPU time debugging: Total GPU time = {total_gpu_time}")
        for col in valid_gpu_cols:
            col_sum = df[col].sum()
            print(f"    {col}: {col_sum}")
        
        # Calculate real CPU/GPU time ratio
        df["cpu_gpu_ratio"] = df[time_col] / (df["gtime (inc)"] + 1e-9)
        advanced_metrics.extend(["gtime (inc)", "cpu_gpu_ratio"])
        print(f"✓ Built real GPU time from {len(valid_gpu_cols)} GPU time columns")
        print("✓ Added cpu_gpu_ratio based on real GPU time")
    elif time_col:
        # If no GPU time columns, use gins as substitute
        gins_col = next((col for col in df.columns if col.startswith('gins') and 'inc' in col), None)
        if gins_col:
            df["cpu_gpu_ratio"] = np.where(df[gins_col] > 0,
                                          df[time_col] / (df[gins_col] / 1e9),
                                          float('inf'))
            advanced_metrics.append("cpu_gpu_ratio")
            print("✓ Added cpu_gpu_ratio (based on GPU instruction count)")
    
    # ====  Corrected bandwidth calculation, avoid outliers  =================
    if time_col and h2d_col:
        # Add minimum time threshold to avoid division by zero and extreme values
        df["h2d_bw_MBps"] = np.where(df[time_col] > 1e-6,  # Minimum 1 microsecond
                                    (df[h2d_col] / 1e6) / df[time_col],
                                    0)
        advanced_metrics.append("h2d_bw_MBps")
        print("✓ Added h2d_bw_MBps (corrected version)")
    
    if time_col and d2h_col:
        df["d2h_bw_MBps"] = np.where(df[time_col] > 1e-6,  # Minimum 1 microsecond
                                    (df[d2h_col] / 1e6) / df[time_col],
                                    0)
        advanced_metrics.append("d2h_bw_MBps")
        print("✓ Added d2h_bw_MBps (corrected version)")

    # ====  Corrected GPU utilization calculation - use correct sum method  =================
    if "gtime (inc)" in df.columns and time_col:
        # Correction: use sum() to calculate total time, not max()
        total_runtime = df[time_col].sum()    # Total CPU time of all functions
        total_gputime = df["gtime (inc)"].sum()  # Total GPU time of all functions
        
        print(f"🔍 Utilization calculation debugging:")
        print(f"  Total runtime: {total_runtime}")
        print(f"  Total GPU time: {total_gputime}")

        if total_runtime > 0:
            gpu_utilization_pct = (total_gputime / total_runtime) * 100
            gpu_idling_pct = 100 - gpu_utilization_pct
            
            # Attach these metrics to GraphFrame object
            gf.global_metrics = {
                "GPU Utilization %": gpu_utilization_pct,
                "GPU Idling %": gpu_idling_pct,
                "Total Runtime (us)": total_runtime,
                "Total GPU Time (us)": total_gputime  # This value should now be correct
            }
            
            print(f"✓ Calculated GPU utilization: {gpu_utilization_pct:.2f}% / {gpu_idling_pct:.2f}%")
            advanced_metrics.extend(["GPU Utilization %", "GPU Idling %"])
    else:
        print("⚠️  Cannot calculate GPU utilization: missing gtime column")

    print(f"✓ Added {len(advanced_metrics)} total advanced metrics:")
    for metric in advanced_metrics:
        print(f"    {metric}")
    
    return advanced_metrics

def analyze_hotspots_with_markdown(gf):
    """Hotspot Analysis - Focus on GPU kernel, H2D/D2H transfer, Python stack time"""
    print("\n" + "="*50)
    print("🔥 Hotspot Analysis (Top 10 Functions)")
    print("="*50)
    
    df = gf.dataframe
    
    # Find actual column names
    time_col = next((col for col in df.columns if 'time (inc)' in col), None)
    gtime_col = "gtime (inc)" if "gtime (inc)" in df.columns else None
    h2d_col = next((col for col in df.columns if 'gxcopy:h2d' in col and 'inc' in col), None)
    d2h_col = next((col for col in df.columns if 'gxcopy:d2h' in col and 'inc' in col), None)
    
    if not time_col:
        print("Time column not found, cannot perform hotspot analysis")
        return
    
    # Sort by time, take top 10 hotspots
    hot = df.sort_values(by=time_col, ascending=False).head(10)
    
    # Build columns to display
    display_cols = ["name", time_col]
    
    if gtime_col:
        display_cols.append(gtime_col)
    
    if h2d_col:
        display_cols.append(h2d_col)
        
    if d2h_col:
        display_cols.append(d2h_col)
    
    # Add derived metrics columns
    derived_cols = ["cpu_gpu_ratio", "h2d_bw_MBps", "d2h_bw_MBps"]
    for col in derived_cols:
        if col in df.columns:
            display_cols.append(col)
    
    # Only keep existing columns
    available_cols = [col for col in display_cols if col in hot.columns]
    
    print("📊 Top 10 Hotspot Functions Detailed Analysis:")
    print("(Sorted by total time)\n")
    
    # Create display dataframe
    display_df = hot[available_cols].copy()
    
    # Format numeric columns for better display
    for col in display_df.columns:
        if col != "name":
            if 'bw_MBps' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "0.00")
            elif 'ratio' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2e}" if pd.notna(x) and x != float('inf') else "inf")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
    
    # Truncate function name for better display
    display_df["name"] = display_df["name"].apply(lambda x: x[:60] + "..." if len(x) > 60 else x)
    
    # Output Markdown table
    try:
        markdown_table = display_df.to_markdown(index=False, tablefmt="grid")
        print(markdown_table)
    except Exception as e:
        print(f"Markdown output failed, using plain format: {e}")
        print(display_df.to_string(index=False))
    
    return hot

def analyze_gpu_kernel_focus(gf):
    """Focus on GPU kernel performance analysis"""
    print("\n" + "="*50)
    print("⚡ GPU Kernel Focused Analysis")
    print("="*50)
    
    df = gf.dataframe
    
    # Find GPU kernel-related functions
    gpu_kernel_funcs = df[df['name'].str.contains('kernel|cuda|gpu|CUDA', case=False, na=False)]
    
    if len(gpu_kernel_funcs) > 0:
        print("🎯 GPU Kernel Related Functions:")
        
        time_col = next((col for col in df.columns if 'time (inc)' in col), None)
        if time_col:
            gpu_kernel_sorted = gpu_kernel_funcs.sort_values(by=time_col, ascending=False).head(5)
            
            for i, (idx, row) in enumerate(gpu_kernel_sorted.iterrows()):
                print(f"  {i+1}. {row['name']}: {row[time_col]:,.0f}")
                
                # If there is bandwidth information, also display it
                if 'h2d_bw_MBps' in row and pd.notna(row['h2d_bw_MBps']):
                    print(f"     H2D Bandwidth: {row['h2d_bw_MBps']:.2f} MB/s")
                if 'd2h_bw_MBps' in row and pd.notna(row['d2h_bw_MBps']):
                    print(f"     D2H Bandwidth: {row['d2h_bw_MBps']:.2f} MB/s")
    else:
        print("ℹ️  No obvious GPU kernel functions found")

def analyze_python_stack_focus(gf):
    """Focus on Python stack time analysis"""
    print("\n" + "="*50)
    print("🐍 Python Stack Time Focused Analysis")
    print("="*50)
    
    df = gf.dataframe
    
    # Find Python-related functions
    python_funcs = df[df['name'].str.contains('\.py:|python|torch|numpy', case=False, na=False)]
    
    if len(python_funcs) > 0:
        print("📈 Python Code Hotspot:")
        
        time_col = next((col for col in df.columns if 'time (inc)' in col), None)
        if time_col:
            python_sorted = python_funcs.sort_values(by=time_col, ascending=False).head(5)
            
            total_time = df[time_col].sum()
            for i, (idx, row) in enumerate(python_sorted.iterrows()):
                pct = (row[time_col] / total_time * 100) if total_time > 0 else 0
                print(f"  {i+1}. {row['name']}: {row[time_col]:,.0f} ({pct:.1f}%)")
    else:
        print("ℹ️  No Python stack information found")

def display_global_summary_table(gf):
    """Display global performance summary in a nice table"""
    if hasattr(gf, 'global_metrics') and gf.global_metrics:
        print("\n" + "="*50)
        print("📈 Global Performance Summary")
        print("="*50)
        
        data = gf.global_metrics
        
        print("+-------------------------+----------------------+")
        print("| Metric                  | Value                |")
        print("+-------------------------+----------------------+")
        print(f"| GPU Utilization %       | {data.get('GPU Utilization %', 0):>18.2f} % |")
        print(f"| GPU Idling %            | {data.get('GPU Idling %', 0):>18.2f} % |")
        print("+-------------------------+----------------------+")
        print(f"| Total Runtime (us)      | {data.get('Total Runtime (us)', 0):>18,.0f} |")
        print(f"| Total GPU Time (us)     | {data.get('Total GPU Time (us)', 0):>18,.0f} |")
        print("+-------------------------+----------------------+")

def assess_workload_scale(gf):
    """Assess workload scale"""
    print("\n" + "="*50)
    print("📏 Workload Scale Assessment")
    print("="*50)
    
    df = gf.dataframe
    
    # 查找关键指标
    time_col = next((col for col in df.columns if 'time' in col.lower()), None)
    gpu_col = next((col for col in df.columns if col.startswith('gins') and 'inc' in col), None)
    copy_h2d = next((col for col in df.columns if 'h2d' in col.lower()), None)
    
    issues = []
    recommendations = []
    
    if time_col:
        total_time = df[time_col].sum()
        print(f"⏱️  Total Runtime: {total_time:,.0f} (time unit)")
        
    if gpu_col:
        total_gpu = df[gpu_col].sum()
        print(f"🚀 Total GPU Instructions: {total_gpu:,.0f}")
        
        # 检查GPU利用率
        if total_gpu < 1e6:  # 少于100万条指令
            issues.append("GPU instructions too few (< 1M)")
            recommendations.append("Increase batch size or model complexity")
    
    if copy_h2d:
        total_h2d = df[copy_h2d].sum()
        print(f"📡 H2D Data Transfer: {total_h2d:,.0f} bytes ({total_h2d/1e6:.1f} MB)")
        
        # 检查数据传输量
        if total_h2d < 1e9:  # 少于1GB
            issues.append("Data transfer too few (< 1GB)")
            recommendations.append("Increase embedding table size or batch size")
    
    # CPU vs GPU 比例检查
    if time_col and gpu_col:
        cpu_time = df[time_col].sum()
        gpu_ops = df[gpu_col].sum()
        
        # 简单的不平衡检测（这里的比例判断需要根据具体情况调整）
        if gpu_ops < cpu_time / 1e6:  # GPU操作相对CPU时间太少
            issues.append("CPU-GPU workload imbalance")
            recommendations.append("Consider moving more computation to GPU")
    
    # 总结评估
    print(f"\n📋 Scale Assessment Results:")
    if issues:
        print("⚠️  Found Issues:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n💡 Improvement Suggestions:")
        for rec in recommendations:
            print(f"    - {rec}")
    else:
        print("✅ Workload scale looks appropriate")

def analyze_cpu_gpu_distribution(gf):
    """Analyze CPU vs GPU time distribution - enhanced version"""
    print("\n" + "="*50)
    print("📊 CPU vs GPU Detailed Distribution Analysis")
    print("="*50)
    
    df = gf.dataframe
    
    # 查找指标
    time_metrics = [col for col in df.columns if 'time' in col.lower()]
    gpu_metrics = [col for col in df.columns if col.startswith('gins')]
    cycles_metrics = [col for col in df.columns if 'cycles' in col.lower()]
    
    print("🔍 Key Performance Metrics:")
    
    # CPU时间/周期
    if time_metrics:
        time_col = time_metrics[0]
        total_time = df[time_col].sum()
        print(f"  ⏱️  Total Time: {total_time:,.0f}")
        
        # 找出时间最长的函数
        top_time = df.nlargest(3, time_col)
        print(f"  🔥 Most Time-Consuming Functions:")
        for i, (idx, row) in enumerate(top_time.iterrows()):
            print(f"     {i+1}. {row['name']}: {row[time_col]:,.0f}")
    
    if cycles_metrics:
        cycles_col = cycles_metrics[0]
        total_cycles = df[cycles_col].sum()
        print(f"  🔄 Total CPU Cycles: {total_cycles:,.0f}")
    
    # GPU指令
    if gpu_metrics:
        gpu_col = gpu_metrics[0]
        total_gpu = df[gpu_col].sum()
        print(f"  🚀 Total GPU Instructions: {total_gpu:,.0f}")
        
        # CPU vs GPU 比例
        if time_metrics:
            ratio = total_gpu / (total_time if total_time > 0 else 1)
            print(f"  📊 GPU/CPU Ratio: {ratio:.2e}")
        
        # GPU密集型函数
        top_gpu = df.nlargest(3, gpu_col)
        print(f"  🎯 GPU-Intensive Functions:")
        for i, (idx, row) in enumerate(top_gpu.iterrows()):
            if row[gpu_col] > 0:
                print(f"     {i+1}. {row['name']}: {row[gpu_col]:,.0f}")

def analyze_data_bandwidth(gf):
    """Analyze data transfer bandwidth - corrected version"""
    print("\n" + "="*50)
    print("🌐 Data Transfer Bandwidth Analysis - Corrected Version")
    print("="*50)
    
    df = gf.dataframe
    
    # 使用真实的GPU传输时间而非CPU时间
    gxcopy_time_col = next((col for col in df.columns if 'gxcopy' in col and 'inc' in col and 'h2d' not in col and 'd2h' not in col), None)
    h2d_col = next((col for col in df.columns if 'gxcopy:h2d' in col and 'inc' in col), None)
    d2h_col = next((col for col in df.columns if 'gxcopy:d2h' in col and 'inc' in col), None)
    
    if not gxcopy_time_col:
        print("⚠️  No GPU transfer time column found")
        return
    
    total_transfer_time = df[gxcopy_time_col].sum()
    
    print("📊 Corrected Transfer Bandwidth Statistics:")
    print(f"  Total GPU Transfer Time: {total_transfer_time:.3f} time unit")
    
    if h2d_col and total_transfer_time > 0:
        total_h2d_bytes = df[h2d_col].sum()
        h2d_bandwidth = (total_h2d_bytes / 1e6) / total_transfer_time
        print(f"  H2D Transfer Total: {total_h2d_bytes:,.0f} bytes ({total_h2d_bytes/1e6:.1f} MB)")
        print(f"  H2D Actual Bandwidth: {h2d_bandwidth:.2f} MB/s")
        
        # PCIe理论带宽对比
        pcie_theoretical = 128000  # MB/s for PCIe 5.0
        efficiency = (h2d_bandwidth / pcie_theoretical) * 100
        print(f"  PCIe Efficiency: {efficiency:.2f}% (vs {pcie_theoretical:,} MB/s theoretical value)")
    
    if d2h_col and total_transfer_time > 0:
        total_d2h_bytes = df[d2h_col].sum()
        d2h_bandwidth = (total_d2h_bytes / 1e6) / total_transfer_time
        print(f"  D2H Transfer Total: {total_d2h_bytes:,.0f} bytes ({total_d2h_bytes/1e6:.1f} MB)")
        print(f"  D2H Actual Bandwidth: {d2h_bandwidth:.2f} MB/s")

def analyze_gpu_kernel_efficiency(gf):
    """Analyze GPU kernel efficiency - corrected version (based on user suggestions)"""
    print("\n" + "="*50)
    print("⚡ GPU Kernel Detailed Efficiency Analysis (Corrected)")
    print("="*50)
    
    df = gf.dataframe
    
    # GPU kernel related metrics
    kernel_metrics = [col for col in df.columns if 'gker' in col.lower()]
    stall_metrics = [col for col in df.columns if 'stl' in col.lower()]
    occupancy_metrics = [col for col in df.columns if 'occ' in col.lower()]
    
    if kernel_metrics:
        print("🔧 GPU Kernel Statistics:")
        for metric in kernel_metrics[:5]:
            total_value = df[metric].sum()
            print(f"  {metric}: {total_value}")
    
    if occupancy_metrics:
        print(f"\n📈 GPU Occupancy Metrics:")
        for metric in occupancy_metrics:
            mean_occ = df[metric].mean()
            max_occ = df[metric].max()
            print(f"  {metric}: average={mean_occ:.1f}%, max={max_occ:.1f}%")
    
    if stall_metrics:
        print(f"\n⚠️  GPU Stall Detailed Analysis (Corrected Algorithm):")
        
        # Find total stall metric (user's correct suggestion!)
        total_stall_col = next((col for col in df.columns if 'stl_any' in col and 'inc' in col), None)
        gpu_instruction_col = next((col for col in df.columns if col.startswith('gins') and 'inc' in col), None)
        
        if total_stall_col and df[total_stall_col].sum() > 0:
            total_stall_cycles = df[total_stall_col].sum()
            total_instructions = df[gpu_instruction_col].sum() if gpu_instruction_col else 0
            
            print(f"📊 Stall Analysis Base Data:")
            print(f"  Total stall cycles: {total_stall_cycles:,.0f}")
            print(f"  Total GPU instructions: {total_instructions:,.0f}")
            
            stall_summary = []
            print(f"\n📈 Each stall type as percentage of total stalls:")
            
            for metric in stall_metrics:
                if 'stl_any' not in metric:  # Exclude total stall itself
                    specific_stall = df[metric].sum()
                    
                    # Correct calculation method (user suggestion)
                    stall_percentage = 100 * specific_stall / total_stall_cycles
                    
                    # Additional analysis dimension
                    avg_stall_per_instruction = specific_stall / total_instructions if total_instructions > 0 else 0
                    
                    stall_summary.append((metric, specific_stall, stall_percentage, avg_stall_per_instruction))
                    
                    # Simplified stall type names
                    short_name = metric.replace('gins:stl_', '').replace(' (inc)', '')
                    
                    print(f"  {short_name:<10}: {specific_stall:>15,.0f} cycles ({stall_percentage:>6.1f}% total stall) [avg {avg_stall_per_instruction:.2f}/instruction]")
            
            # Sort by percentage and analyze
            stall_summary.sort(key=lambda x: x[2], reverse=True)
            print(f"\n🔍 Stall Type Analysis (sorted by percentage):")
            
            for i, (metric, cycles, pct, avg_per_inst) in enumerate(stall_summary[:3]):
                short_name = metric.replace('gins:stl_', '').replace(' (inc)', '')
                if pct > 50:
                    severity = "🚨 Major Bottleneck"
                elif pct > 20:
                    severity = "⚠️  Important Factor"
                else:
                    severity = "📝 Minor Factor"
                
                print(f"  {i+1}. {severity} {short_name}: {pct:.1f}% of stall time")
                
                # Give specific suggestions based on stall type
                if 'cmem' in short_name:
                    print(f"      💡 Excessive constant memory access - check parameter passing and constant cache")
                elif 'idep' in short_name:
                    print(f"      💡 Serious instruction dependency - consider increasing parallelism or kernel fusion")
                elif 'gmem' in short_name:
                    print(f"      💡 Global memory bottleneck - optimize memory access patterns")
                elif 'sync' in short_name:
                    print(f"      💡 High synchronization overhead - reduce unnecessary sync points")
                    
            # Verify total (should be close to 100%)
            total_percentage = sum(x[2] for x in stall_summary)
            print(f"\n✅ Verification: Total of all stall types = {total_percentage:.1f}% (should be close to 100%)")
            
        else:
            print("⚠️  Total stall metric not found, using traditional method...")
            # Fall back to old method but with improved descriptions
            gpu_col = next((col for col in df.columns if col.startswith('gins') and 'inc' in col), None)
            if gpu_col:
                total_instructions = df[gpu_col].sum()
                print(f"📊 Total GPU instructions: {total_instructions:,.0f}")
                
                for metric in stall_metrics[:5]:
                    stall_cycles = df[metric].sum()
                    avg_per_instruction = stall_cycles / total_instructions
                    short_name = metric.replace('gins:stl_', '').replace(' (inc)', '')
                    print(f"  {short_name}: {stall_cycles:,.0f} stall cycles (avg {avg_per_instruction:.2f} cycles/instruction)")
                    
def analyze_derived_percentages(gf, derived_metrics):
    """Analyze derived percentage metrics"""
    print("\n" + "="*50)
    print("📊 Derived Percentage Metrics Analysis")
    print("="*50)
    
    df = gf.dataframe
    
    if not derived_metrics:
        print("⚠️  No derived metrics available")
        return
    
    print("📈 Key Percentage Metrics:")
    for metric in derived_metrics:
        if metric in df.columns:
            max_pct = df[metric].max()
            mean_pct = df[metric].mean()
            print(f"  {metric}: Max={max_pct:.2f}%, Avg={mean_pct:.2f}%")
            
            # 显示百分比最高的函数
            if max_pct > 0:
                top_pct = df.nlargest(3, metric)
                print(f"    Top Functions:")
                for i, (idx, row) in enumerate(top_pct.iterrows()):
                    if row[metric] > 0:
                        print(f"      {i+1}. {row['name']}: {row[metric]:.2f}%")

def generate_enhanced_recommendations(gf):
    """Generate enhanced optimization recommendations"""
    print("\n" + "="*50)
    print("💡 Enhanced Optimization Recommendations")
    print("="*50)
    
    df = gf.dataframe
    
    recommendations = []
    
    # 工作负载规模建议
    gpu_col = next((col for col in df.columns if col.startswith('gins') and 'inc' in col), None)
    if gpu_col and df[gpu_col].sum() < 1e6:
        recommendations.append("📏 Workload Scale Recommendations:")
        recommendations.append("   - Increase embedding table size to >100万条目")
        recommendations.append("   - Increase batch size to >512")
        recommendations.append("   - Increase MLP layers and width")
        recommendations.append("   - Consider running multiple iterations")
    
    # CPU-GPU 平衡建议
    time_col = next((col for col in df.columns if 'time' in col.lower()), None)
    if time_col and gpu_col:
        cpu_time = df[time_col].sum()
        gpu_ops = df[gpu_col].sum()
        if gpu_ops < cpu_time / 1e6:
            recommendations.append("⚖️  CPU-GPU Balance Optimization:")
            recommendations.append("   - Move embedding lookup to GPU")
            recommendations.append("   - Use GPU-optimized embedding library")
            recommendations.append("   - Consider asynchronous execution of CPU and GPU tasks")
    
    # 数据传输优化
    copy_cols = [col for col in df.columns if 'gxcopy' in col.lower()]
    if copy_cols:
        total_transfer = sum(df[col].sum() for col in copy_cols)
        if total_transfer > 0:
            recommendations.append("📡 Data Transfer Optimization:")
            recommendations.append("   - Use CUDA unified memory")
            recommendations.append("   - Batch data transfer")
            recommendations.append("   - Consider keeping data on GPU")
    
    if recommendations:
        for rec in recommendations:
            print(f"  {rec}")

def analyze_real_gpu_performance(gf):
    """GPU performance analysis based on real HPCToolkit metrics"""
    print("\n" + "="*60)
    print("🚀 Real GPU Performance Analysis (HPCViewer metrics)")
    print("="*60)
    
    df = gf.dataframe
    
    # Find real timing metrics
    real_metrics = {
        'cpu_time': 'time (inc)',
        'gpu_total': 'gpuop (inc)', 
        'gpu_kernel': 'gker (inc)',
        'gpu_copy': 'gxcopy (inc)',
        'kernel_count': 'gker:count (inc)',
        'copy_count': 'gxcopy:count (inc)',
        'h2d_bytes': 'gxcopy:h2d (inc)',
        'd2h_bytes': 'gxcopy:d2h (inc)',
        'gpu_instructions': 'gins (inc)'
    }
    
    print("📊 Real Performance Metrics Analysis:")
    
    # Extract real values
    results = {}
    for key, metric in real_metrics.items():
        if metric in df.columns:
            results[key] = df[metric].sum()
            
    # CPU time analysis
    if 'cpu_time' in results:
        print(f"  🖥️  Total CPU time: {results['cpu_time']:.3f} time units")
    
    # GPU time analysis - use real metrics
    if 'gpu_total' in results:
        print(f"  🚀 Total GPU time: {results['gpu_total']:.3f} time units")
        
    if 'gpu_kernel' in results:
        print(f"  ⚡ GPU kernel time: {results['gpu_kernel']:.3f} time units")
        
    if 'gpu_copy' in results:
        print(f"  📡 GPU copy time: {results['gpu_copy']:.3f} time units")
    
    # Calculate real performance ratios
    if 'cpu_time' in results and 'gpu_total' in results:
        cpu_time = results['cpu_time']
        gpu_time = results['gpu_total']
        
        print(f"\n📈 Corrected Performance Comparison:")
        print(f"  CPU time: {cpu_time:.1f} time units")
        print(f"  GPU time: {gpu_time:.1f} time units")
        print(f"  Real GPU ratio: {(gpu_time/(cpu_time+gpu_time)*100):.2f}%")
        
        if 'gpu_kernel' in results and 'gpu_copy' in results:
            kernel_time = results['gpu_kernel']
            copy_time = results['gpu_copy']
            
            print(f"  GPU kernel time: {kernel_time:.1f} ({kernel_time/gpu_time*100:.1f}% of GPU time)")
            print(f"  GPU copy time: {copy_time:.1f} ({copy_time/gpu_time*100:.1f}% of GPU time)")
            
            # Key findings
            if copy_time > kernel_time:
                ratio = copy_time / kernel_time
                print(f"  🚨 Data transfer time is {ratio:.1f}x computation time!")
            else:
                print(f"  ✅ Computation time exceeds transfer time as expected")

def analyze_kernel_launch_efficiency(gf):
    """Analyze kernel launch efficiency"""
    print("\n" + "="*60)
    print("🔧 GPU Kernel Launch Efficiency Analysis")
    print("="*60)
    
    df = gf.dataframe
    
    # Find kernel related metrics
    kernel_metrics = {
        'count': 'gker:count (inc)',
        'time': 'gker (inc)',
        'blocks': 'gker:blks_acumu (inc)',
        'threads': 'gker:blk_thr_acumu (inc)'
    }
    
    results = {}
    for key, metric in kernel_metrics.items():
        if metric in df.columns:
            results[key] = df[metric].sum()
    
    if 'count' in results and 'time' in results:
        kernel_count = results['count']
        kernel_time = results['time']
        
        print(f"📊 Kernel Launch Statistics:")
        print(f"  Total kernel launches: {kernel_count:,.0f}")
        print(f"  Total kernel execution time: {kernel_time:.3f} time units")
        
        if kernel_count > 0:
            avg_kernel_time = kernel_time / kernel_count
            print(f"  Average kernel time: {avg_kernel_time:.6f} time units")
            
            # Judge kernel efficiency - thresholds adjusted based on data
            if avg_kernel_time < 1e-5:  
                print(f"  🚨 Kernels too fine-grained - recommend kernel fusion")
                print(f"     {kernel_count:,.0f} launches indicate many small kernels")
            elif avg_kernel_time < 1e-4:  
                print(f"  ⚠️  Kernel granularity small - consider optimization")
            else:
                print(f"  ✅ Kernel granularity reasonable")
                
        # Kernel scale analysis
        if 'blocks' in results and 'threads' in results:
            total_blocks = results['blocks']
            total_threads = results['threads']
            
            print(f"\n📏 Kernel Scale Analysis:")
            print(f"  Total blocks: {total_blocks:,.0f}")
            print(f"  Total threads: {total_threads:,.0f}")
            
            if kernel_count > 0:
                avg_blocks = total_blocks / kernel_count
                avg_threads = total_threads / kernel_count
                print(f"  Average blocks per kernel: {avg_blocks:.1f}")
                print(f"  Average threads per kernel: {avg_threads:.1f}")
    else:
        print("⚠️  Kernel count metrics not found")

def analyze_memory_transfer_efficiency(gf):
    """Analyze memory transfer efficiency"""
    print("\n" + "="*60)
    print("📡 Memory Transfer Efficiency Analysis")
    print("="*60)
    
    df = gf.dataframe
    
    # Find transfer related metrics
    transfer_metrics = {
        'h2d_bytes': 'gxcopy:h2d (inc)',
        'd2h_bytes': 'gxcopy:d2h (inc)', 
        'copy_time': 'gxcopy (inc)',
        'copy_count': 'gxcopy:count (inc)'
    }
    
    results = {}
    for key, metric in transfer_metrics.items():
        if metric in df.columns:
            results[key] = df[metric].sum()
    
    print("📊 Data Transfer Statistics:")
    
    if 'h2d_bytes' in results:
        h2d_mb = results['h2d_bytes'] / 1e6
        print(f"  H2D total transfer: {results['h2d_bytes']:,.0f} bytes ({h2d_mb:.1f} MB)")
    
    if 'd2h_bytes' in results:
        d2h_mb = results['d2h_bytes'] / 1e6  
        print(f"  D2H total transfer: {results['d2h_bytes']:,.0f} bytes ({d2h_mb:.1f} MB)")
    
    if 'copy_time' in results:
        print(f"  Total transfer time: {results['copy_time']:.3f} time units")
    
    if 'copy_count' in results:
        print(f"  Transfer operations: {results['copy_count']:,.0f}")
    
    # Calculate transfer efficiency (FIX: Better time unit handling)
    if 'h2d_bytes' in results and 'copy_time' in results and results['copy_time'] > 0:
        h2d_bytes = results['h2d_bytes']
        copy_time = results['copy_time']
        
        # More conservative bandwidth calculation - assume time is in seconds
        # If bandwidth seems too high, time units might be microseconds
        bandwidth_mbps = (h2d_bytes / 1e6) / copy_time  # MB/s assuming seconds
        
        # Check if this gives unrealistic values and adjust
        if bandwidth_mbps > 500000:  # If > 500 GB/s, likely time is in microseconds
            bandwidth_mbps = bandwidth_mbps / 1e6  # Convert from μs to s
            time_unit = "μs"
        else:
            time_unit = "s"
        
        print(f"\n📈 Transfer Efficiency Analysis:")
        print(f"  Actual H2D bandwidth: {bandwidth_mbps:.2f} MB/s")
        print(f"  Time units interpreted as: {time_unit}")
        print(f"  PCIe 4.0 theoretical bandwidth: ~64,000 MB/s")
        print(f"  PCIe 5.0 theoretical bandwidth: ~128,000 MB/s")
        
        # More realistic efficiency calculation
        theoretical_bw = 64000  # Use PCIe 4.0 as baseline
        efficiency = (bandwidth_mbps / theoretical_bw) * 100
        print(f"  Bandwidth efficiency: {efficiency:.2f}%")
        
        # Transfer frequency analysis
        if 'copy_count' in results and results['copy_count'] > 0:
            avg_transfer_size = h2d_bytes / results['copy_count']
            print(f"  Average transfer size: {avg_transfer_size/1e3:.1f} KB")
            
            if avg_transfer_size < 1e6:  # Less than 1MB
                print(f"  🚨 Transfers too fragmented - recommend batching")
            else:
                print(f"  ✅ Transfer size reasonable")
        
        if efficiency < 1:
            print(f"  🚨 Bandwidth severely underutilized - optimize transfer patterns")
        elif efficiency < 10:
            print(f"  ⚠️  Bandwidth low - consider pinned memory and async transfers")
        else:
            print(f"  ✅ Bandwidth utilization reasonable")
    else:
        print("⚠️  Cannot calculate transfer bandwidth")

def display_comprehensive_summary(gf):
    """Display comprehensive performance summary"""
    print("\n" + "="*60)
    print("📋 Comprehensive Performance Summary (Based on Real Metrics)")
    print("="*60)
    
    df = gf.dataframe
    
    # 提取关键指标
    key_metrics = {
        'CPU Time': df.get('time (inc)', pd.Series([0])).sum(),
        'GPU Total Time': df.get('gpuop (inc)', pd.Series([0])).sum(), 
        'GPU Kernel Time': df.get('gker (inc)', pd.Series([0])).sum(),
        'GPU Copy Time': df.get('gxcopy (inc)', pd.Series([0])).sum(),
        'H2D Transfer (MB)': df.get('gxcopy:h2d (inc)', pd.Series([0])).sum() / 1e6,
        'Kernel Launch Count': df.get('gker:count (inc)', pd.Series([0])).sum(),
        'Transfer Count': df.get('gxcopy:count (inc)', pd.Series([0])).sum()
    }
    
    print("┌─────────────────────┬──────────────────────┐")
    print("│ Metric Name         │ Value                │")
    print("├─────────────────────┼──────────────────────┤")
    
    for metric, value in key_metrics.items():
        if 'MB' in metric:
            print(f"│ {metric:<19} │ {value:>18.1f} MB │")
        elif 'Count' in metric:
            print(f"│ {metric:<19} │ {value:>18,.0f} │")
        else:
            print(f"│ {metric:<19} │ {value:>18.3f} │")
    
    print("└─────────────────────┴──────────────────────┘")
    
    # 关键结论
    gpu_total = key_metrics['GPU Total Time']
    gpu_kernel = key_metrics['GPU Kernel Time'] 
    gpu_copy = key_metrics['GPU Copy Time']
    
    if gpu_total > 0:
        kernel_pct = gpu_kernel / gpu_total * 100
        copy_pct = gpu_copy / gpu_total * 100
        
        print(f"\n🎯 Key Findings:")
        print(f"  • GPU Time Distribution: Kernel {kernel_pct:.1f}% vs Copy {copy_pct:.1f}%")
        
        if copy_pct > 60:
            print(f"  🚨 Data Transfer Dominates - This is the main performance bottleneck!")
        elif copy_pct > 30:
            print(f"  ⚠️  Data Transfer Overhead High - Need Optimization")
        else:
            print(f"  ✅ Computation Time Dominates - Expected")

def main():
    """Main function - Enhanced version"""
    print("🚀 DLRM GPU Performance Analysis Tool - Enhanced Version")
    print("="*60)
    
    # Load database
    gf = load_hpctoolkit_database("hpctoolkit-python3.11-database-gpu")
    if not gf:
        return
    
    print(f"📈 Data overview: {gf.dataframe.shape[0]} functions/call sites, {gf.dataframe.shape[1]} metrics")
    
    # Enhanced metrics list
    enhanced_list_all_metrics(gf)
    
    # 🆕 Corrected real GPU performance analysis  
    analyze_real_gpu_performance(gf)
    
    # 🆕 Corrected kernel launch efficiency analysis
    analyze_kernel_launch_efficiency(gf)
    
    # 🆕 Corrected memory transfer efficiency analysis
    analyze_memory_transfer_efficiency(gf)
    
    # 🆕 Comprehensive summary
    display_comprehensive_summary(gf)
    
    # Keep original analysis functions
    derived_metrics = add_derived_metrics(gf)
    advanced_metrics = add_advanced_derived_metrics(gf)
    
    # Display global utilization summary
    display_global_summary_table(gf)
    
    # Assess workload scale
    assess_workload_scale(gf)
    
    # Execute various analyses
    analyze_cpu_gpu_distribution(gf)
    analyze_data_bandwidth(gf)
    analyze_gpu_kernel_efficiency(gf)
    
    # Analyze derived metrics
    if derived_metrics:
        analyze_derived_percentages(gf, derived_metrics)
    
    # ===== New specialized analyses =====
    # Hotspot analysis (with markdown tables)
    analyze_hotspots_with_markdown(gf)
    
    # GPU kernel specialized analysis
    analyze_gpu_kernel_focus(gf)
    
    # Python stack specialized analysis
    analyze_python_stack_focus(gf)
    
    # Generate enhanced recommendations
    generate_enhanced_recommendations(gf)

    # ================== New: Management Report Generation ==================
    print("\n" + "="*60)
    print("📊 Generating Management Visualization Reports")
    print("="*60)
    
    # Generate management charts
    plot_cpu_gpu_overview(gf)
    plot_gpu_time_breakdown(gf)
    plot_gpu_stall_analysis(gf)
    plot_kernel_launch_efficiency(gf)
    plot_transfer_size_distribution(gf)
    plot_gpu_occupancy_timeline(gf)

    # Generate management executive summary
    generate_executive_summary()

    # Generate technical detailed report
    generate_technical_appendix(gf)
    
    print("\n" + "="*60)
    print("✅ Complete performance analysis finished!")
    print("📁 Output files:")
    print(f"  📊 Charts directory: {PLOT_DIR}")
    print(f"  📋 Reports directory: {REPORTS_DIR}")
    if VISUALIZATION_ENABLED:
        print("  🎯 Management core charts:")
        print(f"    - CPU vs GPU time distribution: {PLOT_DIR}/cpu_gpu_overview.png")
        print(f"    - GPU time breakdown: {PLOT_DIR}/gpu_breakdown.png") 
        print(f"    - GPU stall analysis: {PLOT_DIR}/gpu_stall_analysis.png")
        print(f"    - Kernel launch efficiency: {PLOT_DIR}/kernel_launch_efficiency.png")
        print(f"    - Transfer size distribution: {PLOT_DIR}/transfer_size_distribution.png")
        print(f"    - GPU occupancy timeline: {PLOT_DIR}/gpu_occupancy_timeline.png")
    print("  📄 Management reports:")
    print(f"    - Executive summary: {REPORTS_DIR}/executive_summary.md")
    print(f"    - Technical details: {REPORTS_DIR}/technical_appendix.md")
    print(f"    - Data file: {REPORTS_DIR}/metrics_data.json")
    print("\n💡 Tip: Use executive summary and charts for management briefing, technical details for engineer analysis")
    print("="*60)

if __name__ == "__main__":
    main()