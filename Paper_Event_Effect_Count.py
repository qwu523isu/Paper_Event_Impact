# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:55:32 2025

@author: Qiong Wu

Count BTC option quotes before and after events
Created on 2025-06-04
"""

from bates_utils import (
    clear_environment, clear_console, load_events, load_btc_options
)
from pathlib import Path
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------- configuration
DATA_DIR = Path(r"D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling/Data/BTC")
OUTPUT_DIR = Path(r"D:/2024-2025/Research/Paper6/Paper_6_News_Effect/1_Modeling/Output")
TIMESTAMP_COL = "local_timestamp"
WINDOW_MINUTES = 5

# ---------------------------------------------------------------- initialize
clear_environment()
clear_console()

# ---------------------------------------------------------------- main loop
def main():
    target_date = datetime(2021, 1, 3)
    events = load_events(DATA_DIR, target_date)
    if events.empty:
        print("No events found, exiting.")
        return

    btc_data = load_btc_options(
        DATA_DIR,
        "deribit_options_chain_2021-01-03_OPTIONS.csv",
        chunk_size=100000
    )

    clips_before, clips_after = [], []
    for _, row in events.iterrows():
        evt_id, t0 = row.event_id, row.event_time
        t_before = t0 - pd.Timedelta(minutes=WINDOW_MINUTES)
        t_after = t0 + pd.Timedelta(minutes=WINDOW_MINUTES)

        mask_before = (btc_data[TIMESTAMP_COL] >= t_before) & (btc_data[TIMESTAMP_COL] < t0)
        mask_after = (btc_data[TIMESTAMP_COL] >= t0) & (btc_data[TIMESTAMP_COL] < t_after)

        cut_b = btc_data.loc[mask_before].copy()
        cut_a = btc_data.loc[mask_after].copy()

        if not cut_b.empty:
            cut_b["event_id"] = evt_id
            cut_b["event_time"] = t0
            cut_b["window"] = "before"
            clips_before.append(cut_b)

        if not cut_a.empty:
            cut_a["event_id"] = evt_id
            cut_a["event_time"] = t0
            cut_a["window"] = "after"
            clips_after.append(cut_a)

    combined = pd.concat(clips_before + clips_after, ignore_index=True)
    summary_df = (
        combined.groupby(["event_id", "window"])
                .size()
                .unstack(fill_value=0)
                .rename(columns={"before": "n_pre", "after": "n_after"})
                .reset_index()
                .merge(events[["event_id", "event_time"]], on="event_id")
    )
    summary_df = summary_df[["event_id", "event_time", "n_pre", "n_after"]]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "event_quote_counts.csv"
    summary_df.to_csv(csv_path, index=False)
    print(summary_df.head())

# ---------------------------------------------------------------- run script
if __name__ == "__main__":
    main()