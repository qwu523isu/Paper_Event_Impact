# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:50:38 2025

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

import psutil
import multiprocessing
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import QuantLib as ql
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import scipy
import matplotlib
import seaborn
import tqdm
import joblib

# ============================================
# Part 1: Check Resource
# ============================================
# Check system resources
memory = psutil.virtual_memory()
cpu_count = multiprocessing.cpu_count()
print(f"RAM: {memory.total / (1024**3):.2f} GB")
print(f"CPU Cores: {cpu_count}")
print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================
# Part 2: Verify QuantLib
# ============================================
print(f"QuantLib version: {ql.__version__}")
print(f"Today's date: {ql.Date.todaysDate()}")

# ============================================
# Part 3: Create Directories
# ============================================
BASE_DIR = Path("D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling")
DATA_DIR = BASE_DIR / "Data" / "BTC"
#EVENT_WINDOWS_DIR = DATA_DIR / "event_windows"
EVENT_WINDOWS_DIR = DATA_DIR / "event_windows" /"202101"
OUTPUT_DIR = BASE_DIR / "Output"
SCRIPTS_DIR = BASE_DIR

# Create directories
for directory in [BASE_DIR, DATA_DIR, EVENT_WINDOWS_DIR, OUTPUT_DIR, SCRIPTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Directory: {directory}")

# ============================================
# Part 4: Check Data Files
# ============================================
"""
# We only select the external events (All except BTC) with importance > 3
"""
# timeline_file = DATA_DIR / "BTC_Event_Timeline.csv"
timeline_file = DATA_DIR / "All_Except_BTC_Importance_Bytime.csv"
window_files = list(EVENT_WINDOWS_DIR.glob("BTC_event_windows_only_*.csv"))

# Timeline file
size_mb = timeline_file.stat().st_size / (1024 * 1024)
# Event window files
total_size = sum(f.stat().st_size for f in window_files) / (1024 * 1024)

# ============================================
# Part 5: Leverage Scripts
# ============================================
bates_utils_path = SCRIPTS_DIR / "bates_utils.py"
optimized_bates_path = SCRIPTS_DIR / "optimized_bates_utils.py"

# Import modules
import bates_utils
import optimized_bates_utils

# ============================================
# Part 7: Main Analysis
# ============================================
# Import analysis functions
from bates_utils import (
    setup_logging, load_events, load_btc_options, prepare_options, 
    compute_volatility_metrics, filter_intraday_windows
)
from optimized_bates_utils import (
    calibrate_model_parallel, calibrate_heston_model_parallel
)

# Setup
TIMESTAMP_COL = "local_timestamp"
logger = setup_logging(OUTPUT_DIR, "volatility_analysis.log")

# Load event timeline
events_df = pd.read_csv(timeline_file)
events_df['newsDatetime'] = pd.to_datetime(events_df['newsDatetime'], utc=True)

# Process files
all_results = []
intervals_minutes = [1]

for file_idx, file_path in enumerate(tqdm.tqdm(window_files, desc="Processing files")):
    # Load data
    #btc_data = pd.read_csv(file_path)
    btc_data = load_btc_options(file_path)
    
    
    # Extract date from filename
    file_date = pd.to_datetime(file_path.stem.split('_')[-1], format='%Y%m%d').date()
    
    # Find events for this date
    if not events_df.empty:
        day_events = events_df[events_df['newsDatetime'].dt.date == file_date]
    else:
        # Create synthetic event
        day_events = pd.DataFrame({
            'event_id': [f"{file_date}_synthetic"],
            'newsDatetime': [btc_data['local_timestamp'].iloc[len(btc_data)//2]]
        })
    
    if day_events.empty:
        continue
    
    # Extract date from filename
    file_date = pd.to_datetime(file_path.stem.split('_')[-1], format='%Y%m%d').date()
    
    # Find events for this date
    if not events_df.empty:
        day_events = events_df[events_df['newsDatetime'].dt.date == file_date]
    else:
        # Create synthetic event
        day_events = pd.DataFrame({
            'event_id': [f"{file_date}_synthetic"],
            'newsDatetime': [btc_data['local_timestamp'].iloc[len(btc_data)//2]]
        })
    
    if day_events.empty:
        continue
    
    # Process each event
    for _, event_row in day_events.iterrows():
        event_id = event_row.get('event_id', f"{file_date}")
        event_time = event_row['newsDatetime']
        important = event_row.get('important')
        
        # Filter data around event time
        windows = filter_intraday_windows(btc_data, event_time, intervals_minutes, TIMESTAMP_COL)
        
        for window_key, window_data in windows.items():
            if len(window_data) < 8:
                continue
            
            # Prepare options data
            opts = prepare_options(window_data, now=event_time)
            if opts.empty:
                continue
            
            s0 = window_data["underlying_price"].iloc[-1]
            if not np.isfinite(s0):
                continue
            
            # Compute volatility metrics
            vol = compute_volatility_metrics(window_data, s0, TIMESTAMP_COL)
            
            # Calibrate models
            for model_type, calibrate_func in [("Bates", calibrate_model_parallel), 
                                             ("Heston", calibrate_heston_model_parallel)]:
                pars = calibrate_func(opts, s0=s0)
                if pars.get("error"):
                    continue
                
                # Store result
                result = {
                    "file_date": file_date.strftime('%Y-%m-%d'),
                    "important": important,
                    "event_id": event_id,
                    "event_time": event_time,
                    "window": window_key,
                    "model_type": model_type,
                    "rmse": pars["rmse"],
                    "realized_vol": vol["realized_vol"],
                    "avg_implied_vol": vol["avg_implied_vol"],
                    "num_options": len(opts),
                    "underlying_price": s0
                }
                
                # Add model parameters
                param_keys = ["v0", "kappa", "theta", "sigma", "rho", "lambdaJ", "muJ", "sigmaJ"]
                for key in param_keys:
                    result[key] = pars.get(key, np.nan)
                
                all_results.append(result)

# ============================================
# Part 8: Save Results
# ============================================
if all_results:
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"btc_volatility_analysis_{timestamp}_All_External_Effects.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"Analysis complete")
    print(f"Processed {len(results_df)} windows")
    print(f"Results saved to: {output_file}")
else:
    print("No results generated")
    results_df = None

# ============================================
# Part 9: Visualize Results
# ============================================
if results_df is not None and not results_df.empty:    
    print("Creating visualizations")
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('BTC Volatility Analysis Results', fontsize=16, fontweight='bold')
    
    # RMSE by Model
    model_rmse = results_df.groupby('model_type')['rmse'].agg(['mean', 'std'])
    axes[0, 0].bar(model_rmse.index, model_rmse['mean'], 
                  yerr=model_rmse['std'], capsize=10, alpha=0.7)
    axes[0, 0].set_title('Average RMSE by Model Type')
    axes[0, 0].set_ylabel('RMSE')
    
    # Volatility Comparison
    axes[0, 1].scatter(results_df['realized_vol'], results_df['avg_implied_vol'], 
                      alpha=0.6, c=['blue' if x == 'Bates' else 'red' for x in results_df['model_type']])
    max_vol = max(results_df['realized_vol'].max(), results_df['avg_implied_vol'].max())
    axes[0, 1].plot([0, max_vol], [0, max_vol], 'k--', alpha=0.5)
    axes[0, 1].set_xlabel('Realized Volatility')
    axes[0, 1].set_ylabel('Implied Volatility')
    axes[0, 1].set_title('Realized vs Implied Volatility')
    
    # Window Analysis
    window_stats = results_df.groupby('window')['rmse'].mean()
    axes[1, 0].bar(range(len(window_stats)), window_stats.values, alpha=0.7)
    axes[1, 0].set_title('Average RMSE by Window Type')
    axes[1, 0].set_xlabel('Window Type')
    axes[1, 0].set_ylabel('Average RMSE')
    axes[1, 0].set_xticks(range(len(window_stats)))
    axes[1, 0].set_xticklabels(window_stats.index, rotation=45)
    
    # Time Series
    if 'file_date' in results_df.columns:
        daily_rmse = results_df.groupby(['file_date', 'model_type'])['rmse'].mean().unstack()
        if not daily_rmse.empty:
            daily_rmse.plot(ax=axes[1, 1], marker='o', alpha=0.7)
            axes[1, 1].set_title('Daily Average RMSE')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('RMSE')
            axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = OUTPUT_DIR / f"volatility_analysis_plots_{timestamp}_All_External_Effects.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plots saved to: {plot_file}")

