# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 10:31:17 2025

@author: Qiong Wu
"""
import pandas as pd
from datetime import datetime, timedelta

def create_event_window_file():
    """Only keep data within specified time window of each event"""
    
    # Paths
    event_file = r"D:\2024-2025\Research\Paper6\Paper_6_News_Effect\1_Modeling\Data\BTC\BTC_Event_Timeline.csv"
    input_file = r"D:\2024-2025\Research\Paper6\Paper_6_News_Effect\1_Modeling\Data\BTC\deribit_options_chain_2021-01-03_OPTIONS.csv"
    output_file = r"D:\2024-2025\Research\Paper6\Paper_6_News_Effect\1_Modeling\Data\BTC\BTC_event_windows_only.csv"
    
    # First load events for 2021-01-03
    print("Loading events...")
    events = pd.read_csv(event_file)
    events['newsDatetime'] = pd.to_datetime(events['newsDatetime'], utc=True)
    events_jan3 = events[events['newsDatetime'].dt.date == datetime(2021, 1, 3).date()]
    
    print(f"Found {len(events_jan3)} events on 2021-01-03")
    print("\nEvent times:")
    for idx, event in events_jan3.iterrows():
        print(f"  Event {idx}: {event['newsDatetime']}")
    
    # Create time windows (60 minutes before and after each event)
    window_minutes = 60  # Adjust this if you want different window size
    time_ranges = []
    
    for _, event in events_jan3.iterrows():
        start = event['newsDatetime'] - timedelta(minutes=window_minutes)
        end = event['newsDatetime'] + timedelta(minutes=window_minutes)
        time_ranges.append((start, end))
        print(f"\nWindow: {start} to {end}")
    
    # Merge overlapping windows to optimize filtering
    time_ranges.sort(key=lambda x: x[0])
    merged_ranges = []
    
    for start, end in time_ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            # Overlapping windows, merge them
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))
    
    print(f"\nMerged into {len(merged_ranges)} time windows")
    for i, (start, end) in enumerate(merged_ranges):
        duration = (end - start).total_seconds() / 60
        print(f"  Window {i+1}: {start} to {end} ({duration:.0f} minutes)")
    
    # Read and filter options data
    print("\nProcessing options data...")
    filtered_chunks = []
    total_rows = 0
    kept_rows = 0
    chunk_size = 100000
    
    # Count total rows first (optional, for progress tracking)
    print("Counting total rows...")
    with pd.read_csv(input_file, chunksize=chunk_size) as reader:
        for chunk in reader:
            total_rows += len(chunk)
    print(f"Total rows in file: {total_rows:,}")
    
    # Now filter
    processed_rows = 0
    with pd.read_csv(input_file, chunksize=chunk_size) as reader:
        for chunk_num, chunk in enumerate(reader):
            processed_rows += len(chunk)
            
            # Filter BTC only
            btc_mask = chunk['symbol'].str.startswith('BTC')
            chunk = chunk[btc_mask]
            
            if chunk.empty:
                continue
            
            # Convert timestamps
            chunk['local_timestamp'] = pd.to_datetime(chunk['local_timestamp'], unit='us', utc=True)
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], unit='us', utc=True)
            
            # Keep only rows within event windows
            mask = pd.Series(False, index=chunk.index)
            
            # Check each time range
            for start, end in merged_ranges:
                # Use local_timestamp for filtering (matches your analysis)
                mask |= (chunk['local_timestamp'] >= start) & (chunk['local_timestamp'] <= end)
            
            filtered = chunk[mask]
            
            if not filtered.empty:
                filtered_chunks.append(filtered)
                kept_rows += len(filtered)
            
            # Progress update
            if (chunk_num + 1) % 10 == 0:
                progress = (processed_rows / total_rows) * 100
                print(f"Progress: {progress:.1f}% - Processed {processed_rows:,} rows, kept {kept_rows:,} rows")
    
    # Combine all filtered chunks
    print("\nCombining filtered data...")
    if filtered_chunks:
        result = pd.concat(filtered_chunks, ignore_index=True)
        
        # Sort by timestamp
        result = result.sort_values('local_timestamp')
        
        # Save result
        print(f"Saving to {output_file}...")
        result.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"SUMMARY:")
        print(f"{'='*50}")
        print(f"Original rows: {total_rows:,}")
        print(f"Filtered rows: {len(result):,}")
        print(f"Reduction: {(1 - len(result)/total_rows)*100:.1f}%")
        print(f"Output file: {output_file}")
        
        # Estimate file size
        file_size_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"Estimated file size: ~{file_size_mb:.0f} MB")
        
        # Show data distribution across time windows
        print(f"\nData distribution across event windows:")
        for i, (start, end) in enumerate(merged_ranges):
            window_data = result[
                (result['local_timestamp'] >= start) & 
                (result['local_timestamp'] <= end)
            ]
            print(f"  Window {i+1}: {len(window_data):,} rows")
    else:
        print("No data found in the specified time windows!")
    
    return output_file

# If you want even more filtering options:
def create_event_window_file_custom(window_minutes=60, events_only=True):
    """
    Custom version with adjustable parameters
    
    Parameters:
    - window_minutes: minutes before/after each event (default 60)
    - events_only: if True, only keep data around events; if False, keep all January 3rd data
    """
    # Same code as above but with parameters...
    pass

if __name__ == "__main__":
    # Run the filtering
    output_file = create_event_window_file()
    
    print("\nDone! You can now upload this smaller file to Colab.")