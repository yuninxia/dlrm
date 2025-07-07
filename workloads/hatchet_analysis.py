#!/usr/bin/env python3
"""
DLRM CPU+GPU 特性分析工具
专门针对DLRM workload的性能分析
根据专家建议进行了增强
"""

import hatchet as ht
import pandas as pd
import numpy as np

def load_hpctoolkit_database(db_path):
    """加载HPCToolkit数据库"""
    try:
        print(f"正在加载 {db_path}...")
        gf = ht.GraphFrame.from_hpctoolkit_latest(db_path)
        print("✓ 数据库加载成功")
        return gf
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return None

def list_all_metrics(gf):
    """列出所有可用的指标列"""
    print("\n" + "="*50)
    print("🧮 所有可用的指标列")
    print("="*50)
    
    all_columns = sorted(gf.dataframe.columns)
    print(f"总共 {len(all_columns)} 个指标:")
    
    # 按类别分组显示
    categories = {
        '🕒 时间相关': [col for col in all_columns if any(x in col.lower() for x in ['time', 'sec'])],
        '🔥 CPU相关': [col for col in all_columns if any(x in col.lower() for x in ['cycles', 'llc'])],
        '🚀 GPU指令': [col for col in all_columns if col.startswith('gins')],
        '⚙️  GPU Kernel': [col for col in all_columns if col.startswith('gker')],
        '📡 数据传输': [col for col in all_columns if any(x in col.lower() for x in ['copy', 'transfer'])],
        '🔄 同步操作': [col for col in all_columns if 'sync' in col.lower()],
        '🧮 其他': [col for col in all_columns if col not in sum([
            [col for col in all_columns if any(x in col.lower() for x in ['time', 'sec'])],
            [col for col in all_columns if any(x in col.lower() for x in ['cycles', 'llc'])],
            [col for col in all_columns if col.startswith('gins')],
            [col for col in all_columns if col.startswith('gker')],
            [col for col in all_columns if any(x in col.lower() for x in ['copy', 'transfer'])],
            [col for col in all_columns if 'sync' in col.lower()]
        ], [])]
    }
    
    for category, columns in categories.items():
        if columns:
            print(f"\n{category}:")
            for col in columns:
                print(f"    {col}")

def add_derived_metrics(gf):
    """添加派生指标（百分比等）"""
    print("\n" + "="*50)
    print("📊 计算派生指标")
    print("="*50)
    
    df = gf.dataframe
    
    # 查找关键指标
    time_col = next((col for col in df.columns if 'time' in col.lower()), None)
    gpu_col = next((col for col in df.columns if col.startswith('gins') and 'inc' in col), None)
    copy_cols = [col for col in df.columns if 'gxcopy' in col.lower()]
    stall_cols = [col for col in df.columns if 'stl' in col.lower()]
    
    derived_metrics = []
    
    # 数据传输百分比
    if time_col and copy_cols:
        for copy_col in copy_cols:
            if copy_col in df.columns and time_col in df.columns:
                pct_col = f"{copy_col.replace(' (inc)', '')}_pct"
                # 避免除零错误
                df[pct_col] = np.where(df[time_col] > 0, 
                                     100 * df[copy_col] / df[time_col], 0)
                derived_metrics.append(pct_col)
    
    # GPU stall百分比
    if gpu_col and stall_cols:
        for stall_col in stall_cols:
            if stall_col in df.columns and gpu_col in df.columns:
                pct_col = f"{stall_col.replace(' (inc)', '')}_pct"
                df[pct_col] = np.where(df[gpu_col] > 0,
                                     100 * df[stall_col] / df[gpu_col], 0)
                derived_metrics.append(pct_col)
    
    print(f"✓ 添加了 {len(derived_metrics)} 个派生指标:")
    for metric in derived_metrics:
        print(f"    {metric}")
    
    return derived_metrics

def add_advanced_derived_metrics(gf):
    """添加高级派生指标 - CPU/GPU比例和带宽"""
    print("\n" + "="*50)
    print("🔬 计算高级派生指标 (CPU/GPU比例, 带宽)")
    print("="*50)
    
    df = gf.dataframe
    
    # 查找实际的列名
    time_col = next((col for col in df.columns if 'time (inc)' in col), None)
    h2d_col = next((col for col in df.columns if 'gxcopy:h2d' in col and 'inc' in col), None)
    d2h_col = next((col for col in df.columns if 'gxcopy:d2h' in col and 'inc' in col), None)
    
    advanced_metrics = []
    
    # ====  根据HPCToolkit手册Table 8.1构建GPU时间  ==============
    # 识别 GPU 时间列 —— HPCToolkit 在 Coarse-grain Profiling 时
    # 会生成 5 类 GPU operation time，我们使用实际可用的列
    GPU_TIME_COLS = ["gker (inc)", "gxcopy (inc)", "gsync (inc)", "gmem (inc)", "gmset (inc)"]
    available_gpu_cols = [col for col in GPU_TIME_COLS if col in df.columns]
    
    print(f"🔍 找到的关键列:")
    print(f"  时间列: {time_col}")
    print(f"  可用GPU时间列: {available_gpu_cols}")
    print(f"  H2D传输列: {h2d_col}")
    print(f"  D2H传输列: {d2h_col}")
    
    # 构建综合GPU时间
    if available_gpu_cols and time_col:
        # 确保列名与dataframe完全匹配
        valid_gpu_cols = [col for col in available_gpu_cols if col in df.columns]
        df["gtime (inc)"] = df[valid_gpu_cols].sum(axis=1).fillna(0)
        
        # 调试信息
        total_gpu_time = df["gtime (inc)"].sum()
        print(f"🔍 GPU时间调试: 总GPU时间 = {total_gpu_time:,.0f}")
        for col in valid_gpu_cols:
            col_sum = df[col].sum()
            print(f"    {col}: {col_sum:,.0f}")
        
        # 计算真实的CPU/GPU时间比例
        df["cpu_gpu_ratio"] = df[time_col] / (df["gtime (inc)"] + 1e-9)
        advanced_metrics.extend(["gtime (inc)", "cpu_gpu_ratio"])
        print(f"✓ 根据{len(valid_gpu_cols)}个GPU时间列构建了真实GPU时间")
        print("✓ 添加了基于真实GPU时间的 cpu_gpu_ratio")
    elif time_col:
        # 如果没有GPU时间列，使用gins作为替代
        gins_col = next((col for col in df.columns if col.startswith('gins') and 'inc' in col), None)
        if gins_col:
            df["cpu_gpu_ratio"] = np.where(df[gins_col] > 0,
                                          df[time_col] / (df[gins_col] / 1e9),  # 归一化GPU指令
                                          float('inf'))
            advanced_metrics.append("cpu_gpu_ratio")
            print("✓ 添加了 cpu_gpu_ratio (基于GPU指令数)")
    
    # ====  H2D／D2H 带宽（MB/s） =================
    if time_col and h2d_col:
        df["h2d_bw_MBps"] = np.where(df[time_col] > 0,
                                    df[h2d_col] / df[time_col] / 1e6,
                                    0)
        advanced_metrics.append("h2d_bw_MBps")
        print("✓ 添加了 h2d_bw_MBps")
    
    if time_col and d2h_col:
        df["d2h_bw_MBps"] = np.where(df[time_col] > 0,
                                    df[d2h_col] / df[time_col] / 1e6,
                                    0)
        advanced_metrics.append("d2h_bw_MBps")
        print("✓ 添加了 d2h_bw_MBps")
    
    print(f"✓ 总共添加了 {len(advanced_metrics)} 个高级指标:")
    for metric in advanced_metrics:
        print(f"    {metric}")
    
    return advanced_metrics

def analyze_hotspots_with_markdown(gf):
    """热点分析 - 重点关注GPU kernel、H2D/D2H传输、Python栈耗时"""
    print("\n" + "="*50)
    print("🔥 热点分析 (Top 10 函数)")
    print("="*50)
    
    df = gf.dataframe
    
    # 查找实际的列名
    time_col = next((col for col in df.columns if 'time (inc)' in col), None)
    gtime_col = "gtime (inc)" if "gtime (inc)" in df.columns else None
    h2d_col = next((col for col in df.columns if 'gxcopy:h2d' in col and 'inc' in col), None)
    d2h_col = next((col for col in df.columns if 'gxcopy:d2h' in col and 'inc' in col), None)
    
    if not time_col:
        print("⚠️  未找到时间列，无法进行热点分析")
        return
    
    # 按时间排序，取前10个热点
    hot = df.sort_values(by=time_col, ascending=False).head(10)
    
    # 构建要显示的列
    display_cols = ["name", time_col]
    
    if gtime_col:
        display_cols.append(gtime_col)
    
    if h2d_col:
        display_cols.append(h2d_col)
        
    if d2h_col:
        display_cols.append(d2h_col)
    
    # 添加派生指标列
    derived_cols = ["cpu_gpu_ratio", "h2d_bw_MBps", "d2h_bw_MBps"]
    for col in derived_cols:
        if col in df.columns:
            display_cols.append(col)
    
    # 只保留存在的列
    available_cols = [col for col in display_cols if col in hot.columns]
    
    print("📊 Top 10 热点函数详细分析:")
    print("(按总时间排序)\n")
    
    # 创建显示用的数据框
    display_df = hot[available_cols].copy()
    
    # 格式化数值列以便更好显示
    for col in display_df.columns:
        if col != "name":
            if 'bw_MBps' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "0.00")
            elif 'ratio' in col:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2e}" if pd.notna(x) and x != float('inf') else "inf")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "0")
    
    # 截断函数名以便更好显示
    display_df["name"] = display_df["name"].apply(lambda x: x[:60] + "..." if len(x) > 60 else x)
    
    # 输出Markdown表格
    try:
        markdown_table = display_df.to_markdown(index=False, tablefmt="grid")
        print(markdown_table)
    except Exception as e:
        print(f"Markdown输出失败，使用普通格式: {e}")
        print(display_df.to_string(index=False))
    
    return hot

def analyze_gpu_kernel_focus(gf):
    """专注分析GPU kernel性能"""
    print("\n" + "="*50)
    print("⚡ GPU Kernel 专项分析")
    print("="*50)
    
    df = gf.dataframe
    
    # 查找GPU kernel相关的函数
    gpu_kernel_funcs = df[df['name'].str.contains('kernel|cuda|gpu|CUDA', case=False, na=False)]
    
    if len(gpu_kernel_funcs) > 0:
        print("🎯 GPU Kernel 相关函数:")
        
        time_col = next((col for col in df.columns if 'time (inc)' in col), None)
        if time_col:
            gpu_kernel_sorted = gpu_kernel_funcs.sort_values(by=time_col, ascending=False).head(5)
            
            for i, (idx, row) in enumerate(gpu_kernel_sorted.iterrows()):
                print(f"  {i+1}. {row['name']}: {row[time_col]:,.0f}")
                
                # 如果有bandwidth信息也显示
                if 'h2d_bw_MBps' in row and pd.notna(row['h2d_bw_MBps']):
                    print(f"     H2D带宽: {row['h2d_bw_MBps']:.2f} MB/s")
                if 'd2h_bw_MBps' in row and pd.notna(row['d2h_bw_MBps']):
                    print(f"     D2H带宽: {row['d2h_bw_MBps']:.2f} MB/s")
    else:
        print("ℹ️  未找到明显的GPU kernel函数")

def analyze_python_stack_focus(gf):
    """专注分析Python栈耗时"""
    print("\n" + "="*50)
    print("🐍 Python 栈耗时专项分析")
    print("="*50)
    
    df = gf.dataframe
    
    # 查找Python相关的函数
    python_funcs = df[df['name'].str.contains('\.py:|python|torch|numpy', case=False, na=False)]
    
    if len(python_funcs) > 0:
        print("📈 Python 代码热点:")
        
        time_col = next((col for col in df.columns if 'time (inc)' in col), None)
        if time_col:
            python_sorted = python_funcs.sort_values(by=time_col, ascending=False).head(5)
            
            total_time = df[time_col].sum()
            for i, (idx, row) in enumerate(python_sorted.iterrows()):
                pct = (row[time_col] / total_time * 100) if total_time > 0 else 0
                print(f"  {i+1}. {row['name']}: {row[time_col]:,.0f} ({pct:.1f}%)")
    else:
        print("ℹ️  未找到Python栈信息")

def assess_workload_scale(gf):
    """评估工作负载规模是否足够大"""
    print("\n" + "="*50)
    print("📏 工作负载规模评估")
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
        print(f"⏱️  总运行时间: {total_time:,.0f} (时间单位)")
        
    if gpu_col:
        total_gpu = df[gpu_col].sum()
        print(f"🚀 总GPU指令: {total_gpu:,.0f}")
        
        # 检查GPU利用率
        if total_gpu < 1e6:  # 少于100万条指令
            issues.append("GPU指令数过少 (< 1M)")
            recommendations.append("增加batch size或模型复杂度")
    
    if copy_h2d:
        total_h2d = df[copy_h2d].sum()
        print(f"📡 H2D数据传输: {total_h2d:,.0f} bytes ({total_h2d/1e6:.1f} MB)")
        
        # 检查数据传输量
        if total_h2d < 1e9:  # 少于1GB
            issues.append("数据传输量过少 (< 1GB)")
            recommendations.append("增加embedding表大小或batch size")
    
    # CPU vs GPU 比例检查
    if time_col and gpu_col:
        cpu_time = df[time_col].sum()
        gpu_ops = df[gpu_col].sum()
        
        # 简单的不平衡检测（这里的比例判断需要根据具体情况调整）
        if gpu_ops < cpu_time / 1e6:  # GPU操作相对CPU时间太少
            issues.append("CPU-GPU工作负载严重不平衡")
            recommendations.append("考虑将更多计算移到GPU上")
    
    # 总结评估
    print(f"\n📋 规模评估结果:")
    if issues:
        print("⚠️  发现的问题:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n💡 改进建议:")
        for rec in recommendations:
            print(f"    - {rec}")
    else:
        print("✅ 工作负载规模看起来合适")

def analyze_cpu_gpu_distribution(gf):
    """分析CPU vs GPU时间分布 - 增强版"""
    print("\n" + "="*50)
    print("📊 CPU vs GPU 详细分布分析")
    print("="*50)
    
    df = gf.dataframe
    
    # 查找指标
    time_metrics = [col for col in df.columns if 'time' in col.lower()]
    gpu_metrics = [col for col in df.columns if col.startswith('gins')]
    cycles_metrics = [col for col in df.columns if 'cycles' in col.lower()]
    
    print("🔍 关键性能指标:")
    
    # CPU时间/周期
    if time_metrics:
        time_col = time_metrics[0]
        total_time = df[time_col].sum()
        print(f"  ⏱️  总时间: {total_time:,.0f}")
        
        # 找出时间最长的函数
        top_time = df.nlargest(3, time_col)
        print(f"  🔥 最耗时函数:")
        for i, (idx, row) in enumerate(top_time.iterrows()):
            print(f"     {i+1}. {row['name']}: {row[time_col]:,.0f}")
    
    if cycles_metrics:
        cycles_col = cycles_metrics[0]
        total_cycles = df[cycles_col].sum()
        print(f"  🔄 总CPU周期: {total_cycles:,.0f}")
    
    # GPU指令
    if gpu_metrics:
        gpu_col = gpu_metrics[0]
        total_gpu = df[gpu_col].sum()
        print(f"  🚀 总GPU指令: {total_gpu:,.0f}")
        
        # CPU vs GPU 比例
        if time_metrics:
            ratio = total_gpu / (total_time if total_time > 0 else 1)
            print(f"  📊 GPU/CPU比例: {ratio:.2e}")
        
        # GPU密集型函数
        top_gpu = df.nlargest(3, gpu_col)
        print(f"  🎯 GPU密集型函数:")
        for i, (idx, row) in enumerate(top_gpu.iterrows()):
            if row[gpu_col] > 0:
                print(f"     {i+1}. {row['name']}: {row[gpu_col]:,.0f}")

def analyze_data_bandwidth(gf):
    """分析数据传输带宽"""
    print("\n" + "="*50)
    print("🌐 数据传输带宽分析")
    print("="*50)
    
    df = gf.dataframe
    
    time_col = next((col for col in df.columns if 'time' in col.lower()), None)
    copy_cols = [col for col in df.columns if 'gxcopy' in col.lower()]
    
    if not time_col or not copy_cols:
        print("⚠️  缺少时间或数据传输指标")
        return
    
    total_time = df[time_col].sum()
    
    print("📊 传输带宽统计:")
    for copy_col in copy_cols:
        if 'h2d' in copy_col.lower() or 'd2h' in copy_col.lower():
            total_bytes = df[copy_col].sum()
            if total_time > 0 and total_bytes > 0:
                bandwidth_mbps = (total_bytes / 1e6) / (total_time if total_time < 1e6 else total_time / 1e6)
                print(f"  {copy_col}: {total_bytes:,.0f} bytes")
                print(f"    带宽: {bandwidth_mbps:.2f} MB/s")
                
                # 找出传输最多的函数
                top_transfer = df.nlargest(3, copy_col)
                print(f"    Top传输函数:")
                for i, (idx, row) in enumerate(top_transfer.iterrows()):
                    if row[copy_col] > 0:
                        print(f"      {i+1}. {row['name']}: {row[copy_col]:,.0f}")

def analyze_gpu_kernel_efficiency(gf):
    """分析GPU kernel效率 - 增强版"""
    print("\n" + "="*50)
    print("⚡ GPU Kernel 详细效率分析")
    print("="*50)
    
    df = gf.dataframe
    
    # GPU kernel相关指标
    kernel_metrics = [col for col in df.columns if 'gker' in col.lower()]
    stall_metrics = [col for col in df.columns if 'stl' in col.lower()]
    occupancy_metrics = [col for col in df.columns if 'occ' in col.lower()]
    
    if kernel_metrics:
        print("🔧 GPU Kernel 统计:")
        for metric in kernel_metrics[:5]:
            total_value = df[metric].sum()
            print(f"  {metric}: {total_value}")
    
    if occupancy_metrics:
        print(f"\n📈 GPU占用率指标:")
        for metric in occupancy_metrics:
            mean_occ = df[metric].mean()
            max_occ = df[metric].max()
            print(f"  {metric}: 平均={mean_occ:.1f}%, 最大={max_occ:.1f}%")
    
    if stall_metrics:
        print(f"\n⚠️  GPU Stall 详细分析:")
        gpu_col = next((col for col in df.columns if col.startswith('gins') and 'inc' in col), None)
        
        stall_summary = []
        for metric in stall_metrics[:5]:
            total_stall = df[metric].sum()
            if gpu_col and df[gpu_col].sum() > 0:
                stall_pct = 100 * total_stall / df[gpu_col].sum()
                stall_summary.append((metric, total_stall, stall_pct))
                print(f"  {metric}: {total_stall:,.0f} ({stall_pct:.1f}%)")
        
        # 按stall百分比排序，显示最严重的
        if stall_summary:
            worst_stall = max(stall_summary, key=lambda x: x[2])
            print(f"\n🚨 最严重的stall类型: {worst_stall[0]} ({worst_stall[2]:.1f}%)")

def analyze_derived_percentages(gf, derived_metrics):
    """分析派生的百分比指标"""
    print("\n" + "="*50)
    print("📊 派生百分比指标分析")
    print("="*50)
    
    df = gf.dataframe
    
    if not derived_metrics:
        print("⚠️  没有可用的派生指标")
        return
    
    print("📈 关键百分比指标:")
    for metric in derived_metrics:
        if metric in df.columns:
            max_pct = df[metric].max()
            mean_pct = df[metric].mean()
            print(f"  {metric}: 最大={max_pct:.2f}%, 平均={mean_pct:.2f}%")
            
            # 显示百分比最高的函数
            if max_pct > 0:
                top_pct = df.nlargest(3, metric)
                print(f"    Top函数:")
                for i, (idx, row) in enumerate(top_pct.iterrows()):
                    if row[metric] > 0:
                        print(f"      {i+1}. {row['name']}: {row[metric]:.2f}%")

def generate_enhanced_recommendations(gf):
    """生成增强的优化建议"""
    print("\n" + "="*50)
    print("💡 增强优化建议")
    print("="*50)
    
    df = gf.dataframe
    
    recommendations = []
    
    # 工作负载规模建议
    gpu_col = next((col for col in df.columns if col.startswith('gins') and 'inc' in col), None)
    if gpu_col and df[gpu_col].sum() < 1e6:
        recommendations.append("📏 工作负载规模建议:")
        recommendations.append("   - 增加embedding表大小到 >100万条目")
        recommendations.append("   - 增加batch size到 >512")
        recommendations.append("   - 增加MLP层数和宽度")
        recommendations.append("   - 考虑运行多次迭代")
    
    # CPU-GPU 平衡建议
    time_col = next((col for col in df.columns if 'time' in col.lower()), None)
    if time_col and gpu_col:
        cpu_time = df[time_col].sum()
        gpu_ops = df[gpu_col].sum()
        if gpu_ops < cpu_time / 1e6:
            recommendations.append("⚖️  CPU-GPU平衡优化:")
            recommendations.append("   - 将embedding查表移到GPU上")
            recommendations.append("   - 使用GPU-optimized embedding库")
            recommendations.append("   - 考虑异步执行CPU和GPU任务")
    
    # 数据传输优化
    copy_cols = [col for col in df.columns if 'gxcopy' in col.lower()]
    if copy_cols:
        total_transfer = sum(df[col].sum() for col in copy_cols)
        if total_transfer > 0:
            recommendations.append("📡 数据传输优化:")
            recommendations.append("   - 使用CUDA unified memory")
            recommendations.append("   - 批量化数据传输")
            recommendations.append("   - 考虑在GPU上保持数据")
    
    # HPCToolkit 采样建议
    recommendations.append("🔬 HPCToolkit采样建议:")
    recommendations.append("   - 添加 LLC_MISSES 采样: -e LLC_MISSES@f400000")
    recommendations.append("   - 增加采样频率获得更细粒度数据")
    recommendations.append("   - 考虑添加 DRAM 带宽指标")
    
    if recommendations:
        for rec in recommendations:
            print(f"  {rec}")

def main():
    """主函数 - 增强版"""
    print("🚀 DLRM CPU+GPU 性能分析工具 - 专家增强版")
    print("="*60)
    
    # 加载数据库
    gf = load_hpctoolkit_database("hpctoolkit-python3.11-database")
    if not gf:
        return
    
    print(f"📈 数据概览: {gf.dataframe.shape[0]} 个函数/调用点, {gf.dataframe.shape[1]} 个指标")
    
    # 列出所有指标
    list_all_metrics(gf)
    
    # 添加派生指标
    derived_metrics = add_derived_metrics(gf)
    
    # 添加高级派生指标
    advanced_metrics = add_advanced_derived_metrics(gf)
    
    # 评估工作负载规模
    assess_workload_scale(gf)
    
    # 执行各种分析
    analyze_cpu_gpu_distribution(gf)
    analyze_data_bandwidth(gf)
    analyze_gpu_kernel_efficiency(gf)
    
    # 分析派生指标
    if derived_metrics:
        analyze_derived_percentages(gf, derived_metrics)
    
    # ===== 新增的专项分析 =====
    # 热点分析（包含markdown表格）
    analyze_hotspots_with_markdown(gf)
    
    # GPU kernel专项分析
    analyze_gpu_kernel_focus(gf)
    
    # Python栈专项分析
    analyze_python_stack_focus(gf)
    
    # 生成增强建议
    generate_enhanced_recommendations(gf)
    
    print("\n" + "="*60)
    print("✅ 专家级分析完成！")
    print("="*60)

if __name__ == "__main__":
    main()