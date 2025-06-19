# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 17:12:56 2025

@author: Qiong Wu
"""

# Clear all variables in the current Python environment
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Confirm variables are cleared
print("All local variables have been cleared.")

# Clear console
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_console()

import sys
try:
    # When running as a script
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # When running in a notebook or interactive session
    script_dir = os.getcwd()
    
sys.path.append(script_dir)

"============================================================================="
"============================================================================="
"============================================================================="

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Load your results
BASE_DIR = Path("D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling")
OUTPUT_DIR = BASE_DIR / "Output"
BTC_Vol_Result_File_Name = "btc_volatility_analysis_20250619_170620_All_External_Effects.csv"
results_df = pd.read_csv(OUTPUT_DIR / BTC_Vol_Result_File_Name)

# Quick data summary
print("=== BTC VOLATILITY ANALYSIS RESULTS ===")
print(f"Total observations: {len(results_df)}")
print(f"Date range: {results_df['file_date'].min()} to {results_df['file_date'].max()}")
print(f"Unique events: {results_df['event_id'].nunique()}")
print(f"Model types: {results_df['model_type'].value_counts()}")
print(f"Windows: {results_df['window'].value_counts()}")
print(f"Importance levels: {results_df['important'].value_counts().sort_index()}")

print("\n=== MODEL PERFORMANCE ===")
performance = results_df.groupby('model_type').agg({
    'rmse': ['mean', 'std', 'min', 'max'],
    'realized_vol': 'mean',
    'avg_implied_vol': 'mean'
}).round(4)
print(performance)

print("\n=== VOLATILITY BY IMPORTANCE ===")
vol_by_importance = results_df.groupby('important').agg({
    'realized_vol': ['mean', 'std', 'count'],
    'avg_implied_vol': ['mean', 'std'],
    'rmse': 'mean'
}).round(4)
print(vol_by_importance)

# Create enhanced visualizations using the corrected function
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Simple corrected volatility plot for immediate use
def simple_enhanced_volatility_plot(results_df):
    """Create a simple enhanced volatility plot."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('BTC Options Volatility Analysis Results (All External)', fontsize=16, fontweight='bold')
    
    # 1. Enhanced Volatility Comparison
    valid_data = results_df.dropna(subset=['realized_vol', 'avg_implied_vol', 'important'])
    
    if not valid_data.empty:
        # Separate by model type
        bates_data = valid_data[valid_data['model_type'] == 'Bates']
        heston_data = valid_data[valid_data['model_type'] == 'Heston']
        
        # Create importance-based color mapping
        importance_norm = plt.Normalize(vmin=valid_data['important'].min(), 
                                      vmax=valid_data['important'].max())
        
        # Plot with importance coloring
        if not bates_data.empty:
            sizes_bates = (bates_data['important'] / valid_data['important'].max()) * 100 + 20
            scatter1 = ax1.scatter(bates_data['realized_vol'], bates_data['avg_implied_vol'], 
                                c=bates_data['important'], s=sizes_bates, alpha=0.7, 
                                cmap='Blues', edgecolors='darkblue', linewidth=0.5, label='Bates')
        
        if not heston_data.empty:
            sizes_heston = (heston_data['important'] / valid_data['important'].max()) * 100 + 20
            scatter2 = ax1.scatter(heston_data['realized_vol'], heston_data['avg_implied_vol'], 
                                c=heston_data['important'], s=sizes_heston, alpha=0.7, 
                                cmap='Reds', edgecolors='darkred', linewidth=0.5, 
                                label='Heston', marker='^')
        
        # Add diagonal line
        max_vol = max(valid_data['realized_vol'].max(), valid_data['avg_implied_vol'].max())
        min_vol = min(valid_data['realized_vol'].min(), valid_data['avg_implied_vol'].min())
        ax1.plot([min_vol, max_vol], [min_vol, max_vol], 'k--', alpha=0.6, linewidth=2)
        
        ax1.set_xlabel('Realized Volatility (%)', fontweight='bold')
        ax1.set_ylabel('Implied Volatility (%)', fontweight='bold')
        ax1.set_title('Realized vs Implied Volatility\n(Size ∝ Importance)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar.set_label('Event Importance', fontweight='bold')
    
    # 2. RMSE by Model Type
    model_rmse = results_df.groupby('model_type')['rmse'].agg(['mean', 'std', 'count'])
    bars = ax2.bar(model_rmse.index, model_rmse['mean'], 
                   yerr=model_rmse['std'], capsize=10, alpha=0.8,
                   color=['#3498db', '#e74c3c'])
    ax2.set_title('Model Performance (RMSE)', fontweight='bold')
    ax2.set_ylabel('Average RMSE')
    ax2.grid(True, alpha=0.3)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, model_rmse['count'])):
        ax2.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + model_rmse['std'].iloc[i] + 0.001,
                f'n={count}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Volatility by Importance
    if 'important' in results_df.columns:
        sns.boxplot(data=results_df, x='important', y='realized_vol', ax=ax3)
        ax3.set_title('Realized Volatility by Event Importance', fontweight='bold')
        ax3.set_xlabel('Event Importance Level')
        ax3.set_ylabel('Realized Volatility (%)')
        ax3.grid(True, alpha=0.3)
    
    # 4. Jump Contribution (Bates only)
    bates_only = results_df[results_df['model_type'] == 'Bates'].copy()
    if not bates_only.empty:
        # Calculate jump contribution
        bates_only['jump_contribution'] = bates_only.apply(
            lambda row: row.get('lambdaJ', 0) * (row.get('muJ', 0)**2 + row.get('sigmaJ', 0)**2) 
            if all(pd.notna([row.get('lambdaJ'), row.get('muJ'), row.get('sigmaJ')])) else np.nan, 
            axis=1
        )
        
        valid_jump = bates_only.dropna(subset=['jump_contribution', 'important'])
        if not valid_jump.empty:
            scatter = ax4.scatter(valid_jump['important'], valid_jump['jump_contribution'],
                                c=valid_jump['realized_vol'], cmap='plasma', s=60, alpha=0.7)
            ax4.set_xlabel('Event Importance')
            ax4.set_ylabel('Jump Contribution to Variance')
            ax4.set_title('Jump Risk vs Event Importance', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
            cbar.set_label('Realized Vol (%)', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"btc_analysis_{timestamp}_all_external.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nEnhanced visualization saved as: {plot_file}")
    
    return fig

# Run the analysis
fig = simple_enhanced_volatility_plot(results_df)

# Additional insights
print("\n=== KEY INSIGHTS ===")
correlation = results_df['realized_vol'].corr(results_df['avg_implied_vol'])
print(f"Overall correlation between realized and implied vol: {correlation:.3f}")

# Check if higher importance events have different volatility patterns
high_importance = results_df[results_df['important'] >= 4]
low_importance = results_df[results_df['important'] <= 3]

if not high_importance.empty and not low_importance.empty:
    print(f"High importance events (≥4): Mean realized vol = {high_importance['realized_vol'].mean():.2f}%")
    print(f"Low importance events (≤3): Mean realized vol = {low_importance['realized_vol'].mean():.2f}%")
    print(f"High importance events: Mean RMSE = {high_importance['rmse'].mean():.4f}")
    print(f"Low importance events: Mean RMSE = {low_importance['rmse'].mean():.4f}")
