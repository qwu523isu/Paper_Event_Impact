# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:49:27 2025

@author: Qiong Wu
Shared utilities for Bates model analysis.
Created on 2025-06-04
"""

import pandas as pd
import numpy as np
import QuantLib as ql
from pathlib import Path
import logging
import os
from datetime import datetime
from scipy.optimize import minimize

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

#def load_btc_options(data_dir, filename, chunk_size=100000):
def load_btc_options(filename, chunk_size=100000):    
    """Load and filter BTC options data in chunks."""
    logger = logging.getLogger(__name__)  # Define logger at the beginning
    
    try:
        #chunk_iter = pd.read_csv(data_dir / filename, chunksize=chunk_size)
        chunk_iter = pd.read_csv(filename, chunksize=chunk_size)
        filtered_chunks = []
        
        for chunk in chunk_iter:
            btc_filter = chunk['symbol'].str.startswith('BTC')
            filtered_chunk = chunk[btc_filter].copy()
            
            # Handle local_timestamp - try microseconds first, then datetime string
            try:
                filtered_chunk['local_timestamp'] = pd.to_datetime(
                    filtered_chunk['local_timestamp'], unit='us', utc=True
                )
            except (ValueError, TypeError):
                # Use format='mixed' for mixed datetime formats
                filtered_chunk['local_timestamp'] = pd.to_datetime(
                    filtered_chunk['local_timestamp'], format='mixed', utc=True
                )
            
            # Handle timestamp
            try:
                filtered_chunk['timestamp'] = pd.to_datetime(
                    filtered_chunk['timestamp'], unit='us', utc=True
                )
            except (ValueError, TypeError):
                # Use format='mixed' for mixed datetime formats
                filtered_chunk['timestamp'] = pd.to_datetime(
                    filtered_chunk['timestamp'], format='mixed', utc=True
                )
            
            # Handle expiration
            try:
                filtered_chunk['expiration'] = pd.to_datetime(
                    filtered_chunk['expiration'], unit='us', utc=True
                )
            except (ValueError, TypeError):
                # Use format='mixed' for mixed datetime formats
                filtered_chunk['expiration'] = pd.to_datetime(
                    filtered_chunk['expiration'], format='mixed', utc=True
                )
            
            logger.debug(f"Chunk has {len(filtered_chunk)} rows, {filtered_chunk['strike_price'].nunique()} unique strikes")
            filtered_chunks.append(filtered_chunk)
        
        btc_data = pd.concat(filtered_chunks, ignore_index=True)
        logger.info(f"Loaded {len(btc_data)} BTC option quotes")
        return btc_data
        
    except FileNotFoundError:
        logger.error(f"{filename} not found")
        raise
    except Exception as e:
        logger.error(f"Error loading BTC data: {e}")
        raise

def prepare_options(df, now):
    """Build the skinny DataFrame calibrate_bates() expects."""
    logger = logging.getLogger(__name__)
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return df
    
    g = df.copy()
    # ---- tau (years to expiry) -------------------------------------------
    if "DTE_float" in g:
        g["tau"] = g["DTE_float"] / 365.0
    else:
        g["tau"] = ((pd.to_datetime(g["expiration"], utc=True) - now)
                    .dt.total_seconds() / (365 * 24 * 3600))
    # ---- strike ----------------------------------------------------------
    g["strike"] = g["strike_price"]
    # ---- price -----------------------------------------------------------
    if "mark_price" in g and "underlying_price" in g:
        g["price"] = g["mark_price"] * g["underlying_price"]
    else:
        g["price"] = g["last_price"]
    # ---- opt_type --------------------------------------------------------
    g["opt_type"] = g["type"].str[0].str.upper()
    g = g[["strike", "tau", "price", "opt_type"]]
    g = g.replace([np.inf, -np.inf], np.nan).dropna()
    invalid_rows = g[~((g.price > 0) & (g.strike > 0) & (g.tau > 0))]
    if not invalid_rows.empty:
        logger.warning(f"Dropped {len(invalid_rows)} rows due to invalid price/strike/tau")
    g = g[(g.price > 0) & (g.strike > 0) & (g.tau > 0)]
    if g.empty:
        logger.warning("No valid options after price/strike/tau filter")
    logger.debug(f"Prepared {len(g)} options after cleaning")
    return g
"""
def compute_volatility_metrics(df, s0, timestamp_col="local_timestamp"):
    # Compute realized volatility and average implied volatility for the window.
    logger = logging.getLogger(__name__)
    if df.empty or len(df) < 2:
        logger.warning("Insufficient data for volatility metrics")
        return {"realized_vol": np.nan, "avg_implied_vol": np.nan}
    
    time_diff = df[timestamp_col].diff().dt.total_seconds().mean()
    logger.debug(f"Average time between quotes: {time_diff:.2f} seconds")
    scaling = np.sqrt(252 * 24 * 3600) if time_diff < 60 else np.sqrt(252 * 24 * 60)
    log_returns = np.log(df["underlying_price"] / df["underlying_price"].shift(1))
    realized_vol = np.std(log_returns, ddof=1) * scaling
    avg_implied_vol = df["mark_iv"].clip(0, 2).mean() if "mark_iv" in df else np.nan
    
    return {"realized_vol": realized_vol, "avg_implied_vol": avg_implied_vol}
"""

def compute_volatility_metrics(df, s0, timestamp_col="local_timestamp"):
    logger = logging.getLogger(__name__)
    if df.empty or len(df) < 2:
        logger.warning("Insufficient data for volatility metrics")
        return {"realized_vol": np.nan, "avg_implied_vol": np.nan}
    
    time_diff = df[timestamp_col].diff().dt.total_seconds().mean()
    logger.debug(f"Average time between quotes: {time_diff:.2f} seconds")
    scaling = np.sqrt(252 * 24 * 3600) if time_diff < 60 else np.sqrt(252 * 24 * 60)
    log_returns = np.log(df["underlying_price"] / df["underlying_price"].shift(1))
    realized_vol = np.std(log_returns, ddof=1) * scaling
    avg_implied_vol = df["mark_iv"].clip(0, 200).mean() if "mark_iv" in df else np.nan  # Changed clip(0, 2) to clip(0, 200)
    
    return {"realized_vol": realized_vol, "avg_implied_vol": avg_implied_vol}

def estimate_jump_contribution(pars):
    """Estimate jump contribution to total variance (approximate)."""
    if "lambdaJ" not in pars or any(np.isnan([pars.get(k, np.nan) for k in ["lambdaJ", "muJ", "sigmaJ"]])):
        return np.nan
    lamJ, muJ, sigJ = pars["lambdaJ"], pars["muJ"], pars["sigmaJ"]
    jump_var = lamJ * (muJ**2 + sigJ**2)
    return jump_var
"""
def calibrate_model(df, s0, *, r=0.0, q=0.0, max_iter=200, max_eval=50, ftol=1e-8):
    # Calibrate a simplified Bates model to option slice using a faster analytical engine.
    logger = logging.getLogger(__name__)
    if df.empty:
        logger.warning("Empty option table provided")
        return {"rmse": np.nan, "error": "Empty option table"}
    
    today = ql.Date.todaysDate()
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = today

    spot = ql.SimpleQuote(float(s0))
    r_ts = ql.FlatForward(today, r, day_count)
    q_ts = ql.FlatForward(today, q, day_count)

    v0, kappa, theta, sigma, rho = 0.04, 2.0, 0.04, 0.5, -0.5
    lamJ, muJ, sigJ = 0.1, -0.05, 0.15
    param_names = ["v0", "kappa", "theta", "sigma", "rho", "lambdaJ", "muJ", "sigmaJ"]

    process = ql.BatesProcess(
        ql.YieldTermStructureHandle(q_ts),
        ql.YieldTermStructureHandle(r_ts),
        ql.QuoteHandle(spot),
        v0, kappa, theta, sigma, rho,
        lamJ, muJ, sigJ
    )
    model = ql.BatesModel(process)
    engine = ql.BatesEngine(model, 64)

    helpers = []
    for _, r in df.iterrows():
        maturity = ql.Period(int(r['tau'] * 365 + 0.5), ql.Days)
        strike = r['strike']
        price = r['price']
        opt_type = ql.Option.Call if r['opt_type'] == 'C' else ql.Option.Put
        helper = ql.HestonModelHelper(
            maturity, calendar, s0, strike,
            ql.QuoteHandle(ql.SimpleQuote(float(price))),
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
        lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
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
"""

# This is the corrected calibrate_model function for your bates_utils.py

def calibrate_model(df, s0, *, r=0.0, q=0.0, max_iter=200, max_eval=50, ftol=1e-8):
    """Calibrate a simplified Bates model to option slice using a faster analytical engine."""
    logger = logging.getLogger(__name__)
    if df.empty:
        logger.warning("Empty option table provided")
        return {"rmse": np.nan, "error": "Empty option table"}
    
    today = ql.Date.todaysDate()
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = today

    spot = ql.SimpleQuote(float(s0))
    r_ts = ql.FlatForward(today, r, day_count)
    q_ts = ql.FlatForward(today, q, day_count)

    v0, kappa, theta, sigma, rho = 0.04, 2.0, 0.04, 0.5, -0.5
    lamJ, muJ, sigJ = 0.1, -0.05, 0.15
    param_names = ["v0", "kappa", "theta", "sigma", "rho", "lambdaJ", "muJ", "sigmaJ"]

    process = ql.BatesProcess(
        ql.YieldTermStructureHandle(q_ts),
        ql.YieldTermStructureHandle(r_ts),
        ql.QuoteHandle(spot),
        v0, kappa, theta, sigma, rho,
        lamJ, muJ, sigJ
    )
    model = ql.BatesModel(process)
    engine = ql.BatesEngine(model, 64)

    helpers = []
    for _, row in df.iterrows():
        maturity = ql.Period(int(row['tau'] * 365 + 0.5), ql.Days)
        strike = float(row['strike'])  # Ensure it's a float
        price = float(row['price'])    # Ensure it's a float
        opt_type = ql.Option.Call if row['opt_type'] == 'C' else ql.Option.Put
        
        # FIXED: Pass s0 as a float, not the spot handle
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
        lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
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

def filter_intraday_windows(df, t0, intervals_minutes, timestamp_col="local_timestamp"):
    """Filter data for multiple intraday time windows before and after t0."""
    logger = logging.getLogger(__name__)
    windows = {}
    for minutes in intervals_minutes:
        t_pre_start = t0 - pd.Timedelta(minutes=minutes)
        t_post_end = t0 + pd.Timedelta(minutes=minutes)
        mask_pre = (df[timestamp_col] >= t_pre_start) & (df[timestamp_col] < t0)
        mask_post = (df[timestamp_col] >= t0) & (df[timestamp_col] < t_post_end)
        pre_data = df[mask_pre].copy()
        post_data = df[mask_post].copy()
        windows[f"pre_{minutes}min"] = pre_data
        windows[f"post_{minutes}min"] = post_data
        logger.debug(f"Filtered {len(pre_data)} pre and {len(post_data)} post quotes for {minutes}-minute window")
    return windows

"""
def calibrate_heston_model(df, s0, *, r=0.0, q=0.0, max_iter=200, max_eval=50, ftol=1e-8):
    # Calibrate a Heston SV model to option slice.
    logger = logging.getLogger(__name__)
    if df.empty:
        logger.warning("Empty option table provided")
        return {"rmse": np.nan, "error": "Empty option table"}
    
    today = ql.Date.todaysDate()
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = today

    spot = ql.SimpleQuote(float(s0))
    r_ts = ql.FlatForward(today, r, day_count)
    q_ts = ql.FlatForward(today, q, day_count)

    v0, kappa, theta, sigma, rho = 0.04, 2.0, 0.04, 0.5, -0.5
    param_names = ["v0", "kappa", "theta", "sigma", "rho"]

    process = ql.HestonProcess(
        ql.YieldTermStructureHandle(q_ts),
        ql.YieldTermStructureHandle(r_ts),
        ql.QuoteHandle(spot),
        v0, kappa, theta, sigma, rho
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    helpers = []
    for _, r in df.iterrows():
        maturity = ql.Period(int(r['tau'] * 365 + 0.5), ql.Days)
        strike = r['strike']
        price = r['price']
        opt_type = ql.Option.Call if r['opt_type'] == 'C' else ql.Option.Put
        helper = ql.HestonModelHelper(
            maturity, calendar, s0, strike,
            ql.QuoteHandle(ql.SimpleQuote(float(price))),
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
        lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
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
"""

# This is the corrected calibrate_heston_model function for your bates_utils.py

def calibrate_heston_model(df, s0, *, r=0.0, q=0.0, max_iter=200, max_eval=50, ftol=1e-8):
    """Calibrate a Heston SV model to option slice."""
    logger = logging.getLogger(__name__)
    if df.empty:
        logger.warning("Empty option table provided")
        return {"rmse": np.nan, "error": "Empty option table"}
    
    today = ql.Date.todaysDate()
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = today

    spot = ql.SimpleQuote(float(s0))
    r_ts = ql.FlatForward(today, r, day_count)
    q_ts = ql.FlatForward(today, q, day_count)

    v0, kappa, theta, sigma, rho = 0.04, 2.0, 0.04, 0.5, -0.5
    param_names = ["v0", "kappa", "theta", "sigma", "rho"]

    process = ql.HestonProcess(
        ql.YieldTermStructureHandle(q_ts),
        ql.YieldTermStructureHandle(r_ts),
        ql.QuoteHandle(spot),
        v0, kappa, theta, sigma, rho
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    helpers = []
    for _, row in df.iterrows():
        maturity = ql.Period(int(row['tau'] * 365 + 0.5), ql.Days)
        strike = float(row['strike'])  # Ensure it's a float
        price = float(row['price'])    # Ensure it's a float
        opt_type = ql.Option.Call if row['opt_type'] == 'C' else ql.Option.Put
        
        # FIXED: Pass s0 as a float, not the spot handle
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
        lm = ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
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

def estimate_event_risk_regression(df, t0, events, timestamp_col="local_timestamp"):
    """Estimate event risk using ATM Black-Scholes IV regression."""
    logger = logging.getLogger(__name__)
    if df.empty or "mark_iv" not in df:
        logger.warning("No valid IV data for regression")
        return {"event_vol": np.nan, "error": "No IV data"}
    
    
    df["tau"] = ((pd.to_datetime(df["expiration"], utc=True) - t0)
                 .dt.total_seconds() / (365 * 24 * 3600))
    logger.debug("Computed tau from expiration and t0")
    
    # Filter out invalid tau
    df = df[df["tau"] > 0].copy()
    if df.empty:
        logger.warning("No valid options after tau filter")
        return {"event_vol": np.nan, "error": "No valid tau"}
    
    # Filter ATM options (approximate by selecting closest to underlying)
    if "strike_price" not in df or "underlying_price" not in df:
        logger.error("Cannot compute moneyness: 'strike_price' or 'underlying_price' missing")
        return {"event_vol": np.nan, "error": "Missing strike_price or underlying_price"}
    
    df["moneyness"] = np.abs(df["strike_price"] / df["underlying_price"] - 1)
    try:
        atm_df = df.loc[df.groupby("tau")["moneyness"].idxmin()]
    except KeyError as e:
        logger.error(f"Grouping by tau failed: {e}")
        return {"event_vol": np.nan, "error": f"Grouping error: {e}"}
    
    # Compute time to maturity (years) for consistency
    atm_df["tau_years"] = atm_df["tau"]  # Already computed
    
    # Count events until each maturity
    event_counts = []
    for tau in atm_df["tau_years"]:
        future_events = events[(events["event_time"] >= t0) & 
                              (events["event_time"] <= t0 + pd.Timedelta(days=tau*365))]
        event_counts.append(len(future_events))
    atm_df["event_count"] = event_counts
    
    # Regression model: IV^2 = sigma^2 + (event_count / tau) * sigma_event^2
    def objective(params, iv2, tau, n_events):
        sigma2, sigma_event2 = params
        predicted_iv2 = sigma2 + (n_events / tau) * sigma_event2
        return np.sum((iv2 - predicted_iv2)**2)
    
    iv2 = (atm_df["mark_iv"] / 100)**2  # Convert IV to decimal
    tau = atm_df["tau_years"]
    n_events = atm_df["event_count"]
    
    if len(iv2) < 3:
        logger.warning("Insufficient maturities for regression")
        return {"event_vol": np.nan, "error": "Insufficient maturities"}
    
    # Minimize with non-negative constraints
    result = minimize(
        objective,
        [0.01, 0.01],  # Initial guess for sigma^2, sigma_event^2
        args=(iv2, tau, n_events),
        bounds=[(0, None), (0, None)]
    )
    
    if result.success:
        sigma_event2 = result.x[1]
        event_vol = np.sqrt(sigma_event2) * 100  # Convert to percentage
        logger.info(f"Regression-based event volatility: {event_vol:.2f}%")
        return {"event_vol": event_vol, "error": None}
    else:
        logger.error(f"Regression failed: {result.message}")
        return {"event_vol": np.nan, "error": result.message}




