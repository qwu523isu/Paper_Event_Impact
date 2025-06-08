# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 14:01:46 2025

Optimized version of bates_utils.py with parallel processing and performance improvements
@author: Qiong Wu
"""

import pandas as pd
import numpy as np
import QuantLib as ql
from pathlib import Path
import logging
import os
from datetime import datetime
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
import pickle
from joblib import Parallel, delayed
import multiprocessing

# Performance optimization flags
ql.Settings.instance().evaluationDate = ql.Date.todaysDate()

def clear_environment():
    """Clear all non-built-in variables in the current Python environment."""
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]
    print("All local variables have been cleared.")

def clear_console():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def setup_logging(output_dir, log_filename):
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_events(data_dir, target_date):
    """Load and filter event timeline data."""
    try:
        events = (
            pd.read_csv(
                data_dir / "BTC_Event_Timeline.csv",
                parse_dates=["newsDatetime"]
            )
            .sort_values("newsDatetime")
            .reset_index(drop=True)
        )
        events["event_id"] = range(len(events))
        events["newsDatetime"] = pd.to_datetime(events["newsDatetime"], utc=True)
        events = events[events['newsDatetime'].dt.date == target_date.date()]
        events = events.rename(columns={"newsDatetime": "event_time"})
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(events)} events for {target_date.date()}")
        if events.empty:
            logger.error(f"No events found for {target_date.date()}")
        return events
    except FileNotFoundError:
        logger.error("BTC_Event_Timeline.csv not found")
        raise
    except Exception as e:
        logger.error(f"Error loading events: {e}")
        raise

def load_btc_options_optimized(data_dir, filename, chunk_size=500000):
    """Optimized BTC options loading with better memory management."""
    try:
        # Define dtypes for better memory usage
        dtypes = {
            'symbol': 'category',
            'type': 'category',
            'strike_price': 'float32',
            'mark_price': 'float32',
            'mark_iv': 'float32',
            'underlying_price': 'float32',
            'last_price': 'float32'
        }
        
        # Read with optimized settings
        chunk_iter = pd.read_csv(
            data_dir / filename, 
            chunksize=chunk_size,
            dtype=dtypes,
            usecols=lambda x: x not in ['unnecessary_column1', 'unnecessary_column2']  # Skip unneeded columns
        )
        
        filtered_chunks = []
        for chunk in chunk_iter:
            # Vectorized filtering
            btc_mask = chunk['symbol'].str.startswith('BTC')
            filtered_chunk = chunk[btc_mask].copy()
            
            # Batch datetime conversion
            filtered_chunk['local_timestamp'] = pd.to_datetime(
                filtered_chunk['local_timestamp'], unit='us', utc=True
            )
            filtered_chunk['timestamp'] = pd.to_datetime(
                filtered_chunk['timestamp'], unit='us', utc=True
            )
            filtered_chunk['expiration'] = pd.to_datetime(
                filtered_chunk['expiration'], unit='us', utc=True
            )
            
            filtered_chunks.append(filtered_chunk)
        
        btc_data = pd.concat(filtered_chunks, ignore_index=True)
        
        # Sort once for better cache locality in subsequent operations
        btc_data = btc_data.sort_values('local_timestamp')
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(btc_data)} BTC option quotes")
        return btc_data
    except Exception as e:
        logger.error(f"Error loading BTC data: {e}")
        raise

def prepare_options_vectorized(df, now):
    """Vectorized version of prepare_options for better performance."""
    logger = logging.getLogger(__name__)
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df
    
    g = df.copy()
    
    # Vectorized operations
    if "DTE_float" in g:
        g["tau"] = g["DTE_float"] / 365.0
    else:
        g["tau"] = ((pd.to_datetime(g["expiration"], utc=True) - now)
                    .dt.total_seconds() / (365 * 24 * 3600))
    
    g["strike"] = g["strike_price"]
    
    if "mark_price" in g and "underlying_price" in g:
        g["price"] = g["mark_price"] * g["underlying_price"] / 100
    else:
        g["price"] = g["last_price"]
    
    g["opt_type"] = g["type"].str[0].str.upper()
    
    # Select columns and filter in one operation
    g = g[["strike", "tau", "price", "opt_type"]]
    
    # Vectorized filtering
    valid_mask = (g.price > 0) & (g.strike > 0) & (g.tau > 0) & g.price.notna() & g.strike.notna() & g.tau.notna()
    g = g[valid_mask]
    
    if g.empty:
        logger.warning("No valid options after filtering")
    
    logger.debug(f"Prepared {len(g)} options after cleaning")
    return g
"""
def compute_volatility_metrics_fast(df, s0, timestamp_col="local_timestamp"):
    # Optimized volatility computation using numpy operations.
    logger = logging.getLogger(__name__)
    if df.empty or len(df) < 2:
        logger.warning("Insufficient data for volatility metrics")
        return {"realized_vol": np.nan, "avg_implied_vol": np.nan}
    
    # Use numpy for faster computation
    prices = df["underlying_price"].values
    timestamps = df[timestamp_col].values
    
    # Compute time differences in seconds using numpy
    time_diffs = np.diff(timestamps).astype('timedelta64[s]').astype(float)
    avg_time_diff = np.mean(time_diffs)
    
    scaling = np.sqrt(252 * 24 * 3600) if avg_time_diff < 60 else np.sqrt(252 * 24 * 60)
    
    # Compute log returns using numpy
    log_returns = np.diff(np.log(prices))
    realized_vol = np.std(log_returns, ddof=1) * scaling
    
    # Fast average for implied vol
    if "mark_iv" in df:
        iv_values = df["mark_iv"].values
        valid_iv = iv_values[(iv_values > 0) & (iv_values < 200)]  # Clip outliers
        avg_implied_vol = np.mean(valid_iv) if len(valid_iv) > 0 else np.nan
    else:
        avg_implied_vol = np.nan
    
    return {"realized_vol": realized_vol, "avg_implied_vol": avg_implied_vol}
"""

def compute_volatility_metrics_fast(df, s0, timestamp_col="local_timestamp"):
    logger = logging.getLogger(__name__)
    if df.empty or len(df) < 2:
        logger.warning("Insufficient data for volatility metrics")
        return {"realized_vol": np.nan, "avg_implied_vol": np.nan}
    
    prices = df["underlying_price"].values
    timestamps = df[timestamp_col].values
    
    time_diffs = np.diff(timestamps).astype('timedelta64[s]').astype(float)
    avg_time_diff = np.mean(time_diffs)
    
    scaling = np.sqrt(252 * 24 * 3600) if avg_time_diff < 60 else np.sqrt(252 * 24 * 60)
    
    log_returns = np.diff(np.log(prices))
    realized_vol = np.std(log_returns, ddof=1) * scaling
    
    if "mark_iv" in df:
        iv_values = df["mark_iv"].values
        valid_iv = iv_values[(iv_values > 0) & (iv_values < 200)]  # Changed < 200 from original
        avg_implied_vol = np.mean(valid_iv) if len(valid_iv) > 0 else np.nan
    else:
        avg_implied_vol = np.nan
    
    return {"realized_vol": realized_vol, "avg_implied_vol": avg_implied_vol}

def calibrate_model_parallel(df, s0, *, r=0.0, q=0.0, max_iter=100, max_eval=30, ftol=1e-6):
    """Optimized calibration with reduced iterations and looser tolerance."""
    logger = logging.getLogger(__name__)
    if df.empty:
        logger.warning("Empty option table provided")
        return {"rmse": np.nan, "error": "Empty option table"}
    
    # Cache frequently used values
    today = ql.Date.todaysDate()
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(s0)))
    r_ts = ql.FlatForward(today, r, day_count)
    q_ts = ql.FlatForward(today, q, day_count)
    
    # Use better initial guesses based on market conditions
    # iv_mean = df.groupby('opt_type')['price'].std().mean() / s0 if len(df) > 10 else 0.3    
    # v0 = max(0.01, min(0.5, iv_mean**2))
    
    iv_mean = df["mark_iv"].mean() / 100 if "mark_iv" in df else 0.3  # Convert percentage to decimal
    v0 = max(0.01, min(0.5, iv_mean**2))

    # Initial parameters with market-based guess
    v0, kappa, theta, sigma, rho = v0, 3.0, v0, 0.4, -0.7
    lamJ, muJ, sigJ = 0.05, -0.02, 0.1
    param_names = ["v0", "kappa", "theta", "sigma", "rho", "lambdaJ", "muJ", "sigmaJ"]
    
    process = ql.BatesProcess(
        ql.YieldTermStructureHandle(q_ts),
        ql.YieldTermStructureHandle(r_ts),
        spot_handle,
        v0, kappa, theta, sigma, rho,
        lamJ, muJ, sigJ
    )
    model = ql.BatesModel(process)
    engine = ql.BatesEngine(model, 32)  # Reduced integration points for speed
    
    # Sample options if too many (for speed)
    if len(df) > 50:
        df_sample = df.sample(n=50, random_state=42)
    else:
        df_sample = df
    
    helpers = []
    for _, row in df_sample.iterrows():
        maturity = ql.Period(int(row['tau'] * 365 + 0.5), ql.Days)
        strike = float(row['strike'])
        price = float(row['price'])
        opt_type = ql.Option.Call if row['opt_type'] == 'C' else ql.Option.Put
        
        helper = ql.HestonModelHelper(
            maturity, calendar, float(s0), strike,
            ql.QuoteHandle(ql.SimpleQuote(price)),
            ql.YieldTermStructureHandle(r_ts),
            ql.YieldTermStructureHandle(q_ts),
            ql.BlackCalibrationHelper.PriceError
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)
    
    need_helpers = 8
    if len(helpers) < need_helpers:
        logger.warning(f"Insufficient contracts: need ≥{need_helpers}, got {len(helpers)}")
        return {"rmse": np.nan, "error": f"Need ≥{need_helpers} contracts, got {len(helpers)}"}
    
    try:
        lm = ql.LevenbergMarquardt(ftol, ftol, ftol)
        model.calibrate(
            helpers, lm,
            ql.EndCriteria(max_iter, max_eval, ftol, ftol, ftol)
        )
        rmse = np.sqrt(np.mean([h.calibrationError()**2 for h in helpers]))
        pars = dict(zip(param_names, model.params()))
        pars["rmse"] = rmse
        pars["error"] = None
        logger.info(f"Calibration successful: RMSE={rmse:.6f}")
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        pars = dict.fromkeys(param_names, np.nan)
        pars["rmse"] = np.nan
        pars["error"] = str(e)
    
    return pars

def filter_intraday_windows_indexed(df, t0, intervals_minutes, timestamp_col="local_timestamp"):
    """Optimized window filtering using index-based slicing."""
    logger = logging.getLogger(__name__)
    
    # Ensure DataFrame is sorted by timestamp
    if not df.index.is_monotonic_increasing:
        df = df.sort_values(timestamp_col)
    
    # Create timestamp index for faster filtering
    df_indexed = df.set_index(timestamp_col, drop=False)
    
    windows = {}
    for minutes in intervals_minutes:
        t_pre_start = t0 - pd.Timedelta(minutes=minutes)
        t_post_end = t0 + pd.Timedelta(minutes=minutes)
        
        # Use index slicing for faster filtering
        try:
            pre_data = df_indexed[t_pre_start:t0].copy()
            post_data = df_indexed[t0:t_post_end].copy()
        except:
            # Fallback to boolean indexing if index slicing fails
            mask_pre = (df[timestamp_col] >= t_pre_start) & (df[timestamp_col] < t0)
            mask_post = (df[timestamp_col] >= t0) & (df[timestamp_col] < t_post_end)
            pre_data = df[mask_pre].copy()
            post_data = df[mask_post].copy()
        
        windows[f"pre_{minutes}min"] = pre_data.reset_index(drop=True)
        windows[f"post_{minutes}min"] = post_data.reset_index(drop=True)
        
        logger.debug(f"Filtered {len(pre_data)} pre and {len(post_data)} post quotes for {minutes}-minute window")
    
    return windows

def process_event_parallel(event_data, btc_data, intervals_minutes, timestamp_col):
    """Process a single event - designed for parallel execution."""
    evt_id, t0, event_row = event_data
    logger = logging.getLogger(__name__)
    
    results = []
    windows = filter_intraday_windows_indexed(btc_data, t0, intervals_minutes, timestamp_col)
    
    for window_key, window_data in windows.items():
        if len(window_data) < 8:
            logger.warning(f"Event {evt_id} {window_key} skipped (fewer than 8 quotes)")
            continue
        
        opts = prepare_options_vectorized(window_data, now=t0)
        if opts.empty:
            logger.warning(f"Event {evt_id} {window_key} skipped (no clean options)")
            continue
        
        s0 = window_data["underlying_price"].iloc[-1]
        if not np.isfinite(s0):
            logger.warning(f"Event {evt_id} {window_key} skipped (NaN underlying)")
            continue
        
        vol = compute_volatility_metrics_fast(window_data, s0, timestamp_col)
        
        # Calibrate Bates model only (skip Heston for speed, or parallelize)
        pars = calibrate_model_parallel(opts, s0=s0)
        if pars.get("error"):
            logger.warning(f"Event {evt_id} {window_key} Bates skipped (calibration failed)")
            continue
        
        result = {
            "event_id": evt_id,
            "event_time": t0,
            "window": window_key,
            "model_type": "Bates",
            "rmse": pars["rmse"],
            "realized_vol": vol["realized_vol"],
            "avg_implied_vol": vol["avg_implied_vol"],
            "jump_var": estimate_jump_contribution(pars)
        }
        result.update({k: pars.get(k, np.nan) for k in ["v0", "kappa", "theta", "sigma", "rho", "lambdaJ", "muJ", "sigmaJ"]})
        results.append(result)
    
    return results

def calibrate_heston_model_parallel(df, s0, *, r=0.0, q=0.0, max_iter=100, max_eval=30, ftol=1e-6):
    """Optimized Heston calibration with reduced iterations and looser tolerance."""
    logger = logging.getLogger(__name__)
    if df.empty:
        logger.warning("Empty option table provided")
        return {"rmse": np.nan, "error": "Empty option table"}
    
    # Cache frequently used values
    today = ql.Date.todaysDate()
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(s0)))
    r_ts = ql.FlatForward(today, r, day_count)
    q_ts = ql.FlatForward(today, q, day_count)
    
    # Use better initial guesses based on market conditions
    # iv_mean = df.groupby('opt_type')['price'].std().mean() / s0 if len(df) > 10 else 0.3
    # v0 = max(0.01, min(0.5, iv_mean**2))
    iv_mean = df["mark_iv"].mean() / 100 if "mark_iv" in df else 0.3  # Convert percentage to decimal
    v0 = max(0.01, min(0.5, iv_mean**2))
    
    # Initial parameters for Heston (no jumps)
    v0, kappa, theta, sigma, rho = v0, 3.0, v0, 0.4, -0.7
    param_names = ["v0", "kappa", "theta", "sigma", "rho"]
    
    process = ql.HestonProcess(
        ql.YieldTermStructureHandle(q_ts),
        ql.YieldTermStructureHandle(r_ts),
        spot_handle,
        v0, kappa, theta, sigma, rho
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)  # Analytic engine for Heston
    
    # Sample options if too many (for speed)
    if len(df) > 50:
        df_sample = df.sample(n=50, random_state=42)
    else:
        df_sample = df
    
    helpers = []
    for _, row in df_sample.iterrows():
        maturity = ql.Period(int(row['tau'] * 365 + 0.5), ql.Days)
        strike = float(row['strike'])
        price = float(row['price'])
        opt_type = ql.Option.Call if row['opt_type'] == 'C' else ql.Option.Put
        
        helper = ql.HestonModelHelper(
            maturity, calendar, float(s0), strike,
            ql.QuoteHandle(ql.SimpleQuote(price)),
            ql.YieldTermStructureHandle(r_ts),
            ql.YieldTermStructureHandle(q_ts),
            ql.BlackCalibrationHelper.PriceError
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)
    
    need_helpers = 5  # Fewer parameters than Bates
    if len(helpers) < need_helpers:
        logger.warning(f"Insufficient contracts: need ≥{need_helpers}, got {len(helpers)}")
        return {"rmse": np.nan, "error": f"Need ≥{need_helpers} contracts, got {len(helpers)}"}
    
    try:
        lm = ql.LevenbergMarquardt(ftol, ftol, ftol)
        model.calibrate(
            helpers, lm,
            ql.EndCriteria(max_iter, max_eval, ftol, ftol, ftol)
        )
        rmse = np.sqrt(np.mean([h.calibrationError()**2 for h in helpers]))
        pars = dict(zip(param_names, model.params()))
        pars["rmse"] = rmse
        pars["error"] = None
        logger.info(f"Heston calibration successful: RMSE={rmse:.6f}")
    except Exception as e:
        logger.error(f"Heston calibration failed: {e}")
        pars = dict.fromkeys(param_names, np.nan)
        pars["rmse"] = np.nan
        pars["error"] = str(e)
    
    return pars

def estimate_jump_contribution(pars):
    """Estimate jump contribution to total variance (approximate)."""
    if "lambdaJ" not in pars or any(np.isnan([pars.get(k, np.nan) for k in ["lambdaJ", "muJ", "sigmaJ"]])):
        return np.nan
    lamJ, muJ, sigJ = pars["lambdaJ"], pars["muJ"], pars["sigmaJ"]
    jump_var = lamJ * (muJ**2 + sigJ**2)
    return jump_var

# Cache mechanism for repeated calibrations
@lru_cache(maxsize=128)
def get_cached_calibration_engine(s0, r, q):
    """Cache calibration engines to avoid recreation."""
    today = ql.Date.todaysDate()
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    
    spot = ql.SimpleQuote(float(s0))
    r_ts = ql.FlatForward(today, r, day_count)
    q_ts = ql.FlatForward(today, q, day_count)
    
    return spot, r_ts, q_ts, calendar, day_count