#!/usr/bin/env python3
"""
🔬 DLRM Thicket 高级性能分析工具
基于 LLNL Thicket 的多维度、探索性性能数据分析
专门针对 HPCToolkit + GPU 性能分析
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add thicket to path if not already available
thicket_path = "/home/ynxia/playground/dlrm/thirdparty/thicket"
if thicket_path not in sys.path:
    sys.path.insert(0, thicket_path)

import thicket as th
import thicket.stats as th_stats
import hatchet as ht

def load_hpctoolkit_with_thicket(db_path):
    """使用 Thicket 加载 HPCToolkit 数据库"""
    print(f"\n🔬 正在使用 Thicket 加载 HPCToolkit 数据库...")
    print(f"   数据库路径: {db_path}")
    
    try:
        # 使用 Hatchet 最新方法读取，然后转换为 Thicket
        gf = ht.GraphFrame.from_hpctoolkit_latest(db_path)
        thicket_obj = th.Thicket.thicketize_graphframe(gf, db_path)
        print(f"✅ Thicket 对象创建成功")
        
        # 打印基本信息
        print(f"📊 数据概览:")
        print(f"   节点数量: {len(list(thicket_obj.graph.traverse()))}")
        print(f"   指标数量: {len(thicket_obj.performance_cols)}")
        print(f"   数据维度: {thicket_obj.dataframe.shape}")
        
        return thicket_obj
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

def explore_thicket_structure(tk):
    """探索 Thicket 对象的结构"""
    print("\n" + "="*60)
    print("🔍 THICKET 对象结构分析")
    print("="*60)
    
    print(f"\n📈 Performance Columns ({len(tk.performance_cols)}):")
    for i, col in enumerate(tk.performance_cols[:15]):  # 只显示前15个
        print(f"   {i+1:2d}. {col}")
    if len(tk.performance_cols) > 15:
        print(f"   ... 还有 {len(tk.performance_cols) - 15} 个指标")
    
    print(f"\n🔢 Exclusive Metrics ({len(tk.exc_metrics)}):")
    for metric in tk.exc_metrics[:10]:
        print(f"   • {metric}")
    
    print(f"\n📊 Inclusive Metrics ({len(tk.inc_metrics)}):")
    for metric in tk.inc_metrics[:10]:
        print(f"   • {metric}")
    
    # 显示数据框的结构
    print(f"\n🗂️  DataFrame 结构:")
    print(f"   Shape: {tk.dataframe.shape}")
    print(f"   Index levels: {tk.dataframe.index.names}")
    print(f"   Columns sample:")
    print(tk.dataframe.head(3))

def gpu_performance_analysis(tk):
    """GPU 性能专项分析"""
    print("\n" + "="*60)
    print("🚀 GPU 性能深度分析 (基于 Thicket)")
    print("="*60)
    
    # 识别 GPU 相关指标
    gpu_metrics = [col for col in tk.performance_cols 
                   if any(keyword in col.lower() for keyword in 
                         ['gpu', 'gker', 'gxcopy', 'gins', 'cuda'])]
    
    print(f"\n🎯 发现 {len(gpu_metrics)} 个 GPU 相关指标:")
    for metric in gpu_metrics:
        print(f"   • {metric}")
    
    if len(gpu_metrics) == 0:
        print("⚠️  未发现 GPU 指标，跳过 GPU 分析")
        return
    
    # 使用 Thicket 统计功能分析 GPU 指标
    print(f"\n📊 GPU 指标统计分析:")
    
    try:
        # 计算关键统计量
        for metric in gpu_metrics[:5]:  # 分析前5个最重要的
            print(f"\n🔬 分析指标: {metric}")
            
            # 使用 Thicket 的统计功能
            if metric in tk.dataframe.columns:
                # 计算均值
                th_stats.mean(tk, columns=[metric])
                
                # 计算最大值、最小值
                th_stats.maximum(tk, columns=[metric])
                th_stats.minimum(tk, columns=[metric])
                
                # 基本统计信息
                data = tk.dataframe[metric].dropna()
                if len(data) > 0:
                    print(f"   📈 总和: {data.sum():,.0f}")
                    print(f"   📊 均值: {data.mean():,.2f}")
                    print(f"   📏 中位数: {data.median():,.2f}")
                    print(f"   📐 标准差: {data.std():,.2f}")
                    print(f"   🔺 最大值: {data.max():,.0f}")
                    print(f"   🔻 最小值: {data.min():,.0f}")
                    
                    # 热点分析：找出该指标最高的函数
                    top_funcs = tk.dataframe.nlargest(3, metric)
                    print(f"   🔥 热点函数 (Top 3):")
                    for i, (idx, row) in enumerate(top_funcs.iterrows()):
                        func_name = str(idx[0]) if isinstance(idx, tuple) else str(idx)
                        # 截断长函数名
                        if len(func_name) > 50:
                            func_name = func_name[:47] + "..."
                        print(f"      {i+1}. {func_name}: {row[metric]:,.0f}")
            
    except Exception as e:
        print(f"⚠️  统计分析失败: {e}")

def bandwidth_efficiency_analysis(tk):
    """数据传输带宽效率分析"""
    print("\n" + "="*60)
    print("📡 数据传输带宽效率分析")
    print("="*60)
    
    # 查找传输相关指标
    transfer_metrics = [col for col in tk.performance_cols 
                       if any(keyword in col.lower() for keyword in 
                             ['gxcopy', 'h2d', 'd2h', 'copy', 'transfer'])]
    
    print(f"\n🔍 发现 {len(transfer_metrics)} 个传输相关指标:")
    for metric in transfer_metrics:
        print(f"   • {metric}")
    
    if len(transfer_metrics) == 0:
        print("⚠️  未发现传输指标")
        return
    
    # 计算带宽相关统计
    h2d_metrics = [m for m in transfer_metrics if 'h2d' in m.lower()]
    d2h_metrics = [m for m in transfer_metrics if 'd2h' in m.lower()]
    
    print(f"\n📊 H2D (Host-to-Device) 分析:")
    for metric in h2d_metrics:
        if metric in tk.dataframe.columns:
            data = tk.dataframe[metric].dropna()
            if len(data) > 0:
                total_bytes = data.sum()
                print(f"   {metric}: {total_bytes:,.0f} bytes ({total_bytes/1e9:.2f} GB)")
    
    print(f"\n📊 D2H (Device-to-Host) 分析:")
    for metric in d2h_metrics:
        if metric in tk.dataframe.columns:
            data = tk.dataframe[metric].dropna()
            if len(data) > 0:
                total_bytes = data.sum()
                print(f"   {metric}: {total_bytes:,.0f} bytes ({total_bytes/1e6:.2f} MB)")

def hotspot_correlation_analysis(tk):
    """热点函数相关性分析"""
    print("\n" + "="*60)
    print("🔥 热点函数相关性分析")
    print("="*60)
    
    # 选择关键指标进行相关性分析
    key_metrics = []
    
    # 时间指标
    time_metrics = [col for col in tk.performance_cols if 'time' in col.lower()]
    if time_metrics:
        key_metrics.extend(time_metrics[:2])
    
    # GPU 指标
    gpu_metrics = [col for col in tk.performance_cols 
                   if any(kw in col.lower() for kw in ['gins', 'gker'])]
    if gpu_metrics:
        key_metrics.extend(gpu_metrics[:2])
    
    # 传输指标
    transfer_metrics = [col for col in tk.performance_cols 
                       if 'gxcopy' in col.lower()]
    if transfer_metrics:
        key_metrics.extend(transfer_metrics[:2])
    
    print(f"\n🎯 选择 {len(key_metrics)} 个关键指标进行相关性分析:")
    for metric in key_metrics:
        print(f"   • {metric}")
    
    if len(key_metrics) >= 2:
        try:
            # 使用 Thicket 的相关性分析功能
            print(f"\n📊 计算节点级相关性...")
            
            # 进行成对相关性分析
            correlation_pairs = []
            for i in range(len(key_metrics)):
                for j in range(i+1, len(key_metrics)):
                    col1, col2 = key_metrics[i], key_metrics[j]
                    try:
                        th_stats.correlation_nodewise(tk, column1=col1, column2=col2, correlation="pearson")
                        correlation_pairs.append((col1, col2))
                        print(f"   ✅ {col1} vs {col2}")
                    except Exception as e:
                        print(f"   ⚠️  {col1} vs {col2}: {e}")
            
            # 显示相关性结果
            correlation_cols = [col for col in tk.statsframe.dataframe.columns 
                              if '_vs_' in str(col) and 'pearson' in str(col)]
            
            if correlation_cols:
                print(f"\n🔗 成功计算 {len(correlation_cols)} 个相关性指标:")
                for col in correlation_cols[:5]:
                    print(f"   • {col}")
                    
                # 显示一些相关性结果
                print(f"\n📈 相关性分析结果样例:")
                for col in correlation_cols[:3]:
                    corr_data = tk.statsframe.dataframe[col].dropna()
                    if len(corr_data) > 0:
                        mean_corr = corr_data.mean()
                        print(f"   {col}: 平均相关性 = {mean_corr:.3f}")
                    
        except Exception as e:
            print(f"⚠️  相关性分析失败: {e}")

def performance_scoring_analysis(tk):
    """性能评分分析"""
    print("\n" + "="*60)
    print("📊 性能评分分析")
    print("="*60)
    
    # 选择主要指标进行评分
    time_metric = next((col for col in tk.performance_cols if 'time' in col.lower()), None)
    
    if not time_metric:
        print("⚠️  未找到时间指标，跳过评分分析")
        return
    
    print(f"\n🎯 基于指标 '{time_metric}' 进行性能评分")
    
    try:
        # 计算统计量为评分做准备
        th_stats.mean(tk, columns=[time_metric])
        th_stats.std(tk, columns=[time_metric])
        
        print(f"✅ 评分相关统计量计算完成")
        
        # 显示性能热点排名
        print(f"\n🏆 性能热点排名 (基于 {time_metric}):")
        
        sorted_data = tk.dataframe.sort_values(time_metric, ascending=False)
        top_10 = sorted_data.head(10)
        
        for i, (idx, row) in enumerate(top_10.iterrows()):
            func_name = str(idx[0]) if isinstance(idx, tuple) else str(idx)
            if len(func_name) > 45:
                func_name = func_name[:42] + "..."
            score = row[time_metric]
            print(f"   {i+1:2d}. {func_name:<45} {score:>12,.2f}")
            
    except Exception as e:
        print(f"⚠️  评分分析失败: {e}")

def advanced_thicket_analysis(tk):
    """高级 Thicket 分析功能展示"""
    print("\n" + "="*60)
    print("🧪 高级 Thicket 分析功能")
    print("="*60)
    
    # 1. 查询功能
    print(f"\n🔍 Thicket 查询功能示例:")
    try:
        # 查询包含特定关键词的函数
        gpu_nodes = tk.dataframe[tk.dataframe.index.get_level_values(0).str.contains('cuda|gpu|kernel', case=False, na=False)]
        if len(gpu_nodes) > 0:
            print(f"   🎯 找到 {len(gpu_nodes)} 个 GPU 相关函数")
        else:
            print(f"   📝 未找到明显的 GPU 函数名")
    except Exception as e:
        print(f"   ⚠️  查询失败: {e}")
    
    # 2. 数据过滤
    print(f"\n🔧 数据过滤功能:")
    try:
        # 过滤掉值为0的行
        original_shape = tk.dataframe.shape
        time_metric = next((col for col in tk.performance_cols if 'time' in col.lower()), None)
        
        if time_metric:
            non_zero_data = tk.dataframe[tk.dataframe[time_metric] > 0]
            print(f"   📊 原始数据: {original_shape[0]} 行")
            print(f"   🎯 非零数据: {non_zero_data.shape[0]} 行")
            print(f"   📈 过滤比例: {(1 - non_zero_data.shape[0]/original_shape[0])*100:.1f}%")
    except Exception as e:
        print(f"   ⚠️  过滤失败: {e}")
    
    # 3. 树形可视化预览
    print(f"\n🌳 调用树结构预览:")
    try:
        # 显示调用树的基本信息
        graph_info = f"   🔗 图节点数: {len(list(tk.graph.traverse()))}"
        print(graph_info)
        
        # 显示根节点信息
        root_nodes = [node for node in tk.graph.traverse() if not node.parents]
        print(f"   🌱 根节点数: {len(root_nodes)}")
        
        if root_nodes:
            root = root_nodes[0]
            print(f"   📝 根节点: {root.frame.get('name', 'unknown')}")
            
    except Exception as e:
        print(f"   ⚠️  树分析失败: {e}")

def generate_optimization_recommendations(tk):
    """基于 Thicket 分析生成优化建议"""
    print("\n" + "="*60)
    print("💡 基于 Thicket 分析的优化建议")
    print("="*60)
    
    recommendations = []
    
    # 分析 GPU 利用率
    gpu_metrics = [col for col in tk.performance_cols 
                   if any(keyword in col.lower() for keyword in ['gins', 'gker'])]
    
    if gpu_metrics:
        for metric in gpu_metrics[:3]:
            if metric in tk.dataframe.columns:
                data = tk.dataframe[metric].dropna()
                if len(data) > 0:
                    total_ops = data.sum()
                    if total_ops > 0:
                        print(f"🚀 {metric}: {total_ops:,.0f} operations")
    
    # 分析数据传输
    h2d_metrics = [col for col in tk.performance_cols if 'h2d' in col.lower()]
    if h2d_metrics:
        for metric in h2d_metrics:
            if metric in tk.dataframe.columns:
                data = tk.dataframe[metric].dropna()
                if len(data) > 0:
                    total_transfer = data.sum()
                    if total_transfer > 1e9:  # > 1GB
                        recommendations.append(f"📡 大量 H2D 传输 ({total_transfer/1e9:.2f} GB) - 考虑数据局部性优化")
    
    # 分析函数热点
    time_metrics = [col for col in tk.performance_cols if 'time' in col.lower()]
    if time_metrics:
        time_metric = time_metrics[0]
        if time_metric in tk.dataframe.columns:
            # 找出占时间比例最大的函数
            sorted_funcs = tk.dataframe.sort_values(time_metric, ascending=False)
            top_func_time = sorted_funcs.iloc[0][time_metric]
            total_time = tk.dataframe[time_metric].sum()
            
            if top_func_time / total_time > 0.3:  # 如果单个函数占30%以上时间
                func_name = str(sorted_funcs.index[0])
                recommendations.append(f"🎯 热点函数优化: {func_name[:50]}... 占总时间 {top_func_time/total_time*100:.1f}%")
    
    print(f"\n📋 优化建议总结:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print(f"   ✅ 性能分布相对均衡，继续按朋友建议优化带宽")
    
    print(f"\n🔬 Thicket 高级功能建议:")
    print(f"   • 使用 tk.tree() 生成可视化调用树")
    print(f"   • 使用 thicket.stats.display_heatmap() 生成热力图")
    print(f"   • 使用 tk.query() 进行复杂性能查询")
    print(f"   • 比较多个优化版本的性能差异")

def main():
    """主函数 - Thicket 高级分析流程"""
    print("🔬 DLRM Thicket 高级性能分析工具")
    print("基于 LLNL Thicket 的多维度性能数据分析")
    print("="*60)
    
    # 加载 HPCToolkit 数据
    db_path = "hpctoolkit-python3.11-database"
    if not os.path.exists(db_path):
        print(f"❌ 数据库路径不存在: {db_path}")
        return
    
    tk = load_hpctoolkit_with_thicket(db_path)
    if not tk:
        return
    
    try:
        # 执行各种分析
        explore_thicket_structure(tk)
        gpu_performance_analysis(tk)
        bandwidth_efficiency_analysis(tk)
        hotspot_correlation_analysis(tk)
        performance_scoring_analysis(tk)
        advanced_thicket_analysis(tk)
        generate_optimization_recommendations(tk)
        
        print("\n" + "="*60)
        print("✅ Thicket 高级分析完成!")
        print("🎯 现在可以使用 Thicket 的更多高级功能:")
        print("   - 多版本性能对比")
        print("   - 交互式可视化")
        print("   - 统计显著性测试")
        print("   - 性能预测建模")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
