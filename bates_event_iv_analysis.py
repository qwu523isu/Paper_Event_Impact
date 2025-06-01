
"""
Bates SVJ Event‑Jump Analysis
=============================
Run this end‑to‑end in Spyder (or any IDE) after adjusting the two file paths
at the top of the script.

* Loads Deribit option snapshot and CryptoPanic headline timeline
* Tags each option quote with event windows (+1 h after each headline)
* Computes Black‑Scholes implied volatility (USD premium)
* Outputs diagnostics tables
* Skeletons for baseline Bates calibration + event‑jump augmentation

Requirements
------------
pip install pandas numpy scipy matplotlib tqdm
"""

# ---------------------------------------------------------------------
# 0. Imports & paths
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import timedelta, datetime
import matplotlib.pyplot as plt

# ---- Adjust these ----
EVENTS_CSV  = Path('D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling/Data/BTC/2021/BTC_Event_Timeline.csv')                 # CryptoPanic timeline
OPTIONS_CSV = Path('D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling/Data/processed_daily_data_2021-01-06.csv')    # Deribit snapshot
TAGGED_OUT  = Path('D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling/Data/processed_daily_data_2021-01-06_tagged.csv')
IV_OUT      = Path('D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling/Data/processed_daily_data_2021-01-06_tagged_with_iv.csv')

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
events  =  pd.read_csv("D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling/Data/BTC_Event_Timeline.csv", parse_dates=['newsDatetime'])
data_path = "D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling/Data/BTC/2021/processed_daily_data_2021-01-06.csv"
options = pd.read_csv(data_path)
    
snapshot_date = pd.to_datetime('2021-01-06').date()
events_day = events[events['newsDatetime'].dt.date == snapshot_date].copy()

# Build +1 h windows
events_day['win_start'] = events_day['newsDatetime']
events_day['win_end']   = events_day['newsDatetime'] + pd.Timedelta(hours=1)
events_day['event_id']  = np.arange(len(events_day))

# ---------------------------------------------------------------------
# 2. Tag option quotes with event_id
# ---------------------------------------------------------------------
opt = options.copy()
opt['local_timestamp'] = pd.to_datetime(opt['local_timestamp'])
opt['event_id'] = -1   # default = non‑event

for _, w in events_day.iterrows():
    mask = (opt['local_timestamp'] >= w.win_start) & (opt['local_timestamp'] < w.win_end)
    opt.loc[mask, 'event_id'] = w.event_id

opt.to_csv(TAGGED_OUT, index=False)
print(f'Saved tagged snapshot → {TAGGED_OUT}')

# ---------------------------------------------------------------------
# 3. Black‑Scholes implied volatility (USD premium)
# ---------------------------------------------------------------------
def bs_price(S, K, T, r, sigma, typ='call', q=0.0):
    if T <= 0 or sigma <= 0:
        return max(0.0, S*exp(-q*T) - K*exp(-r*T)) if typ=='call' else                    max(0.0, K*exp(-r*T) - S*exp(-q*T))
    d1 = (log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if typ=='call':
        return S*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    else:
        return K*exp(-r*T)*norm.cdf(-d2) - S*exp(-q*T)*norm.cdf(-d1)

def implied_vol(target_price, S, K, T, r=0.0, typ='call', q=0.0, tol=1e-6, max_iter=100):
    if target_price <= 0:  # no solution
        return np.nan
    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        price = bs_price(S, K, T, r, mid, typ, q)
        if abs(price - target_price) < tol:
            return mid
        if price > target_price:
            hi = mid
        else:
            lo = mid
    return np.nan

def calc_iv(row, r=0.0):
    S = row['underlying_price']
    K = row['strike_price']
    T = row['DTE_float'] / 365.0
    # Deribit mark_price is quoted in BTC; convert to USD premium
    price = row['mark_price'] * S
    typ = 'call' if row['type'].lower() == 'call' else 'put'
    return implied_vol(price, S, K, T, r, typ)

opt['iv_calc'] = opt.apply(calc_iv, axis=1)
opt.to_csv(IV_OUT, index=False)
print(f'Saved IV snapshot → {IV_OUT}')

# ---------------------------------------------------------------------
# 4. Quick diagnostics
# ---------------------------------------------------------------------
def summary(df, label):
    atm = df.loc[df['delta'].abs().between(0.4,0.6), 'iv_calc']
    return {
        'window': label,
        'quotes': len(df),
        'unique_tickers': df['symbol'].nunique(),
        'median_IV(%)': np.nanmedian(df['iv_calc'])*100,
        'ATM_IV_mean(%)': np.nanmean(atm)*100 if not atm.empty else np.nan
    }

rows = [summary(opt[opt['event_id']==-1], 'non‑event')]
for eid in sorted(events_day['event_id']):
    rows.append(summary(opt[opt['event_id']==eid], f'event {eid}'))

diag = pd.DataFrame(rows)
print('\n=== Diagnostics (6 Jan 2021) ===')
print(diag.to_string(index=False))

# ---------------------------------------------------------------------
# 5. Skeleton — Bates SVJ calibration
# ---------------------------------------------------------------------
# ---- Model characteristic functions ---------------------------------
def heston_cf(u, t, kappa, theta, sigma, rho, v0, r=0.0):
    d = np.sqrt((rho*sigma*1j*u - kappa)**2 + sigma**2*(u*1j + u**2))
    g = (kappa - rho*sigma*1j*u - d) / (kappa - rho*sigma*1j*u + d)
    exp_dt = np.exp(-d*t)
    C = r*1j*u*t + (kappa*theta)/(sigma**2)*((kappa - rho*sigma*1j*u - d)*t - 2.0*np.log((1-g*exp_dt)/(1-g)))
    D = ((kappa - rho*sigma*1j*u - d)/sigma**2) * ((1 - exp_dt)/(1 - g*exp_dt))
    return np.exp(C + D*v0)

def bates_cf(u, t, params, dt_evt=0.0):
    # params: dict holding Bates + jump params
    phi_heston = heston_cf(u, t, params['kappa'], params['theta'], params['sigma'],
                           params['rho'], params['v0'], params.get('r', 0.0))
    muJ, sigJ = params['mu_J'], params['sigma_J']
    psiJ = np.exp(1j*u*muJ - 0.5*sigJ**2*u**2) - 1.0
    lj = params['lambda0']*t + params.get('lambda_evt', 0.0)*dt_evt
    return phi_heston * np.exp(lj * psiJ)

# ---- FFT/Carr‑Madan pricing stub (to be filled) ----------------------
def call_price_fft(S, K, T, params, dt_evt=0.0, alpha=1.5, N=4096, B=1000):
    """
    Computes call price via Carr‑Madan FFT.
    For brevity, left as TODO — you can fill in or swap for QuantLib‑Python.
    """
    raise NotImplementedError('FFT pricing not yet implemented.')

# ---- Calibration stubs ----------------------------------------------
def objective_baseline(pvec, market_df):
    """ Sum‑square error between market IV and Bates model IV (no event) """
    # unpack parameters...
    # compute model IV grid ...
    # return Σ_w error²
    pass  # TODO

def objective_event(theta_evt, baseline_params, market_evt_df, dt_evt_arr):
    """ Calibrate λ_evt or μ_evt to match event‑window surface """
    # combine baseline_params + event params in a dict
    # compute model IV with bates_cf(..., dt_evt=dt_evt_arr[i])
    pass  # TODO
