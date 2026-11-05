#!/usr/bin/env python3
"""
Academic Quality Plots Generator
Generates publication-ready plots for ML-integrated Benders vs Exact methods comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def setup_academic_style():
    """Set up matplotlib for academic publications"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'lines.linewidth': 1.5,
        'lines.markersize': 6
    })

def get_academic_colors():
    """Return colorblind-friendly academic color palette"""
    return {
        'Exact': '#000000',     # Black - baseline
        'ClB': '#E69F00',       # Orange
        'ClB-Ctx': '#56B4E9',   # Sky blue
        'TrB': '#009E73',       # Bluish green
        'TrB-Ctx': '#F0E442',   # Yellow
        'Brs': '#0072B2',       # Blue
        'Dual-Cl': '#D55E00'    # Vermillion
    }

def load_and_process_data():
    """Load and process the experimental data"""
    
    # Load data
    file_path = '/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/RESULTS/Small_experiment7/Merged.xlsx'
    print(f"Loading data from: {file_path}")
    
    df = pd.read_excel(file_path)
    print(f"Original data shape: {df.shape}")
    
    # Apply filtering
    df_filtered = df[
        (df['stopping_reason'] == 'OPTIMAL') & 
        (df['use_stochastic_capacity'] == False) & 
        (df['switching_strategy'] == 'no_switch')
    ].copy()
    
    print(f"Filtered data shape: {df_filtered.shape}")
    
    # Method renaming
    method_mapping = {
        'CLUSTERING_WITH_BANDIT': 'ClB',
        'CLUSTERING_WITH_BANDIT_WITH_CONTEXT': 'ClB-Ctx', 
        'TREE_WITH_BANDIT': 'TrB',
        'TREE_WITH_BANDIT_WITH_CONTEXT': 'TrB-Ctx',
        'BERTSIMAS_RANDOM_SAMPLING': 'Brs',
        'DUAL_CLUSTERING': 'Dual-Cl',
        'exact': 'Exact'
    }
    
    df_filtered['short_label'] = df_filtered['approach'].map(method_mapping)
    
    # Exclude unwanted methods
    excluded_methods = ['Knn', 'Knn-Ctx', 'uncert']
    df_analysis = df_filtered[~df_filtered['short_label'].isin(excluded_methods)].copy()
    
    print(f"Methods after filtering: {sorted(df_analysis['short_label'].unique())}")
    
    return df_analysis

def calculate_metrics(df_analysis):
    """Calculate performance metrics"""
    
    # Separate exact and ML methods
    exact_data = df_analysis[df_analysis['short_label'] == 'Exact'].copy()
    ml_data = df_analysis[df_analysis['short_label'] != 'Exact'].copy()
    
    print(f"Exact data points: {len(exact_data)}")
    print(f"ML data points: {len(ml_data)}")
    
    # Merge for comparison
    merge_cols = ['original_file_name', 'num_subproblems', 'selected_subproblems_ratio', 'original_Prob_Obj_Value']
    
    merged = pd.merge(ml_data, exact_data[merge_cols + ['total_time']], 
                      on=merge_cols, suffixes=('_ml', '_exact'))
    
    print(f"Merged data points: {len(merged)}")
    
    # Calculate metrics
    merged['worst_gap'] = np.maximum(0, ((merged['lower_bound'] - merged['original_Prob_Obj_Value']) / merged['original_Prob_Obj_Value']) * 100)
    merged['opt_gap'] = ((merged['objective_value'] - merged['original_Prob_Obj_Value']) / merged['original_Prob_Obj_Value']) * 100
    merged['time_ratio'] = merged['total_time_ml'] / merged['total_time_exact']
    
    return merged

def prepare_plot_data(merged_metrics, x_var, y_var):
    """Prepare data for plotting by grouping and aggregating"""
    
    if x_var not in merged_metrics.columns or y_var not in merged_metrics.columns:
        print(f"Warning: {x_var} or {y_var} not found in data")
        return pd.DataFrame()
    
    # Group by method and x variable, calculate mean
    grouped = merged_metrics.groupby(['short_label', x_var])[y_var].agg(['mean', 'std', 'count']).reset_index()
    grouped.columns = ['method', x_var, 'mean', 'std', 'count']
    
    return grouped

def create_academic_plot(plot_data, x_var, y_var, ylabel, filename, output_dir):
    """Create a single academic-quality plot"""
    
    if len(plot_data) == 0:
        print(f"  ⚠️ No data for {y_var} vs {x_var}")
        return
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    colors = get_academic_colors()
    
    # Plot each method
    for method in sorted(plot_data['method'].unique()):
        method_data = plot_data[plot_data['method'] == method]
        
        if len(method_data) == 0:
            continue
            
        color = colors.get(method, 'gray')
        linestyle = '--' if method == 'Exact' else '-'
        linewidth = 2.0 if method == 'Exact' else 1.5
        marker = 'o'
        
        ax.plot(method_data[x_var], method_data['mean'],
               color=color, linestyle=linestyle, linewidth=linewidth,
               marker=marker, markersize=5, label=method,
               markerfacecolor='white', markeredgewidth=1.2, markeredgecolor=color)
    
    # Format plot
    ax.set_xlabel(x_var.replace('_', ' ').title())
    ax.set_ylabel(ylabel)
    
    # Professional legend
    legend = ax.legend(frameon=True, fancybox=False, shadow=False,
                      bbox_to_anchor=(1.05, 1), loc='upper left',
                      borderaxespad=0, edgecolor='black', facecolor='white')
    legend.get_frame().set_linewidth(0.5)
    
    # Academic grid style
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Clean up spines
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('black')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save in both formats
    png_path = os.path.join(output_dir, f"academic_{filename}.png")
    eps_path = os.path.join(output_dir, f"academic_{filename}.eps")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(eps_path, format='eps', bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"  ✅ Saved: academic_{filename}.png and .eps")

def main():
    """Main function to generate all academic plots"""
    
    print("=== ACADEMIC PAPER PLOTS GENERATOR ===")
    print("Generating publication-quality figures...")
    
    # Set up academic style
    setup_academic_style()
    
    # Load and process data
    df_analysis = load_and_process_data()
    
    # Calculate metrics
    merged_metrics = calculate_metrics(df_analysis)
    
    # Set up output directory
    output_dir = '/mnt/c/Programming/Research Projects/Benders_Integration_with_ML/RESULTS/Small_experiment7'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot configurations
    plot_configs = [
        ('worst_gap', 'Worst-case Optimality Gap (%)', 'worst_gap'),
        ('opt_gap', 'Optimality Gap (%)', 'opt_gap'),
        ('time_ratio', 'Time Ratio', 'time_ratio')
    ]
    
    x_variables = ['num_subproblems', 'selected_subproblems_ratio']
    
    # Generate plots
    for x_var in x_variables:
        if x_var not in merged_metrics.columns:
            print(f"Warning: {x_var} not found in data. Skipping.")
            continue
        
        print(f"\n--- Creating plots for {x_var} ---")
        
        for y_var, ylabel, short_name in plot_configs:
            print(f"Creating: {y_var} vs {x_var}")
            
            # Prepare plot data
            plot_data = prepare_plot_data(merged_metrics, x_var, y_var)
            
            # Create plot
            filename = f"{short_name}_vs_{x_var}"
            create_academic_plot(plot_data, x_var, y_var, ylabel, filename, output_dir)
    
    print(f"\n🎯 All academic plots completed!")
    print(f"📁 Location: {output_dir}")
    print("📄 Both PNG (300 DPI) and EPS formats available for LaTeX compatibility")

if __name__ == "__main__":
    main()