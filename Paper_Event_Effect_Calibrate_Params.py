# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 16:08:01 2025

Bates SVJ: pre- vs post-event calibration
Created on 2025-06-01 by Qiong Wu
"""

from bates_utils import (
    clear_environment, clear_console, setup_logging,
    load_events, load_btc_options, prepare_options,
    compute_volatility_metrics, estimate_jump_contribution, calibrate_model
)
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# ---------------------------------------------------------------- configuration
DATA_DIR = Path(r"D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling/Data/BTC")
OUTPUT_DIR = Path(r"D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling/output")
TIMESTAMP_COL = "local_timestamp"
WINDOW_MINUTES = 1

# ---------------------------------------------------------------- initialize
clear_environment()
clear_console()
logger = setup_logging(OUTPUT_DIR, "volatility_analysis_1min.log")

# ---------------------------------------------------------------- main loop
def main():
    target_date = datetime(2021, 1, 3)
    events = load_events(DATA_DIR, target_date)
    if events.empty:
        logger.error("No events found, exiting.")
        return

    btc_data = load_btc_options(
        DATA_DIR,
        "deribit_options_chain_2021-01-03_OPTIONS.csv",
        chunk_size=100000
    )

    results = []
    for _, row in events.iterrows():
        evt_id, t0 = row.event_id, row.event_time
        pre = btc_data[(btc_data[TIMESTAMP_COL] >= t0 - pd.Timedelta(minutes=WINDOW_MINUTES)) &
                       (btc_data[TIMESTAMP_COL] < t0)]
        post = btc_data[(btc_data[TIMESTAMP_COL] >= t0) &
                        (btc_data[TIMESTAMP_COL] < t0 + pd.Timedelta(minutes=WINDOW_MINUTES))]

        if len(pre) < 8 or len(post) < 8:
            logger.warning(f"Event {evt_id} skipped (fewer than 8 quotes: pre={len(pre)}, post={len(post)})")
            continue

        opts_pre = prepare_options(pre, now=t0)
        opts_post = prepare_options(post, now=t0)
        if opts_pre.empty or opts_post.empty:
            logger.warning(f"Event {evt_id} skipped (no clean options)")
            continue

        s0_pre = pre["underlying_price"].iloc[-1]
        s0_post = post["underlying_price"].iloc[-1]
        if not np.isfinite(s0_pre) or not np.isfinite(s0_post):
            logger.warning(f"Event {evt_id} skipped (NaN underlying)")
            continue

        vol_pre = compute_volatility_metrics(pre, s0_pre, TIMESTAMP_COL)
        vol_post = compute_volatility_metrics(post, s0_post, TIMESTAMP_COL)

        pars_pre = calibrate_model(opts_pre, s0=s0_pre)
        pars_post = calibrate_model(opts_post, s0=s0_post)

        if pars_pre.get("error") or pars_post.get("error"):
            logger.warning(f"Event {evt_id} skipped (calibration failed)")
            continue

        diff = {f"Δ{k}": pars_post[k] - pars_pre[k]
                for k in pars_pre if k not in ["rmse", "error"]}
        diff.update({
            "event_id": evt_id,
            "event_time": t0,
            "rmse_pre": pars_pre["rmse"],
            "rmse_post": pars_post["rmse"],
            "Δrealized_vol": vol_post["realized_vol"] - vol_pre["realized_vol"],
            "Δavg_implied_vol": vol_post["avg_implied_vol"] - vol_pre["avg_implied_vol"],
            "jump_var_pre": estimate_jump_contribution(pars_pre),
            "jump_var_post": estimate_jump_contribution(pars_post)
        })

        for param in ["lambdaJ", "muJ", "sigmaJ"]:
            pre_val = pars_pre[param]
            delta = diff[f"Δ{param}"]
            rel_change = delta / pre_val if abs(pre_val) > 1e-6 else np.nan
            diff[f"{param}_rel_change"] = rel_change
            diff[f"{param}_significant"] = int(abs(rel_change) > 0.5) if not np.isnan(rel_change) else 0

        results.append(diff)
        logger.info(f"Finished event {evt_id}")

    out_df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "bates_volatility_jump_analysis_1min.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"Diagnostics written to {out_path}")
    logger.info(f"Processed {len(results)} of {len(events)} events successfully")

# ---------------------------------------------------------------- run script
if __name__ == "__main__":
    main()