# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 21:55:13 2025

@author: Qiong Wu
"""

import pandas as pd
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EventWindowProcessor:
    """Process options data to extract event windows with enhanced functionality"""
    
    def __init__(self, 
                 event_file: str,
                 input_files: List[str] = None,
                 input_file: str = None,  # For backward compatibility
                 output_dir: str = None,
                 window_minutes: int = 60,
                 chunk_size: int = 100000):
        """
        Initialize the processor
        
        Parameters:
        - event_file: Path to CSV with event timeline
        - input_files: List of paths to options data CSV files
        - input_file: Single input file (for backward compatibility)
        - output_dir: Directory for output files (default: same as first input)
        - window_minutes: Minutes before/after each event
        - chunk_size: Rows per chunk for processing
        """
        self.event_file = Path(event_file)
        
        # Handle both single file and multiple files
        if input_files:
            self.input_files = [Path(f) for f in input_files]
        elif input_file:
            self.input_files = [Path(input_file)]
        else:
            raise ValueError("Either input_files or input_file must be provided")
        
        self.output_dir = Path(output_dir) if output_dir else self.input_files[0].parent
        self.window_minutes = window_minutes
        self.chunk_size = chunk_size
        
        # Validate files exist
        if not self.event_file.exists():
            raise FileNotFoundError(f"Event file not found: {self.event_file}")
        
        for input_file in self.input_files:
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_date_from_filename(self, filepath: Path) -> str:
        """Extract date from filename like 'deribit_options_chain_2021-02-13_OPTIONS.csv'"""
        filename = filepath.stem
        # Look for date pattern YYYY-MM-DD in filename
        import re
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        match = re.search(date_pattern, filename)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not extract date from filename: {filename}")
    
    def load_events(self, target_dates: List[str] = None) -> pd.DataFrame:
        """
        Load and filter events
        
        Parameters:
        - target_dates: List of dates in 'YYYY-MM-DD' format (default: ['2021-01-03'])
        """
        if target_dates is None:
            target_dates = ['2021-01-03']
        
        logger.info(f"Loading events from {self.event_file}")
        
        try:
            events = pd.read_csv(self.event_file)
            
            # Validate required columns
            required_cols = ['newsDatetime']
            missing_cols = [col for col in required_cols if col not in events.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in event file: {missing_cols}")
            
            # Convert datetime
            events['newsDatetime'] = pd.to_datetime(events['newsDatetime'], utc=True)
            
            # Filter by target dates
            date_objects = [datetime.strptime(date, '%Y-%m-%d').date() for date in target_dates]
            events_filtered = events[events['newsDatetime'].dt.date.isin(date_objects)]
            
            logger.info(f"Found {len(events_filtered)} events on {target_dates}")
            
            if events_filtered.empty:
                logger.warning(f"No events found for dates: {target_dates}")
                return events_filtered
            
            # Log event details
            for idx, event in events_filtered.iterrows():
                logger.info(f"  Event {idx}: {event['newsDatetime']}")
                
            return events_filtered
            
        except Exception as e:
            logger.error(f"Error loading events: {e}")
            raise
    
    def create_time_windows(self, events: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """Create and merge overlapping time windows"""
        
        if events.empty:
            return []
        
        logger.info(f"Creating time windows (±{self.window_minutes} minutes)")
        
        # Create windows
        time_ranges = []
        for _, event in events.iterrows():
            start = event['newsDatetime'] - timedelta(minutes=self.window_minutes)
            end = event['newsDatetime'] + timedelta(minutes=self.window_minutes)
            time_ranges.append((start, end))
            logger.info(f"Window: {start} to {end}")
        
        # Merge overlapping windows
        time_ranges.sort(key=lambda x: x[0])
        merged_ranges = []
        
        for start, end in time_ranges:
            if merged_ranges and start <= merged_ranges[-1][1]:
                # Overlapping windows, merge them
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
            else:
                merged_ranges.append((start, end))
        
        logger.info(f"Merged into {len(merged_ranges)} time windows:")
        for i, (start, end) in enumerate(merged_ranges):
            duration = (end - start).total_seconds() / 60
            logger.info(f"  Window {i+1}: {start} to {end} ({duration:.0f} minutes)")
        
        return merged_ranges
    
    def count_total_rows(self, input_file: Path) -> int:
        """Count total rows in a specific input file"""
        logger.info(f"Counting total rows in {input_file.name}...")
        total_rows = 0
        
        try:
            with pd.read_csv(input_file, chunksize=self.chunk_size) as reader:
                for chunk in reader:
                    total_rows += len(chunk)
            
            logger.info(f"Total rows in {input_file.name}: {total_rows:,}")
            return total_rows
        
        except Exception as e:
            logger.error(f"Error counting rows in {input_file.name}: {e}")
            return 0
    
    def filter_options_data(self, input_file: Path, time_ranges: List[Tuple[datetime, datetime]], 
                          symbol_filter: str = 'BTC') -> pd.DataFrame:
        """Filter options data within time windows for a specific file"""
        
        if not time_ranges:
            logger.warning("No time ranges provided")
            return pd.DataFrame()
        
        logger.info(f"Processing options data from {input_file.name}...")
        
        # Count total rows for progress tracking
        total_rows = self.count_total_rows(input_file)
        if total_rows == 0:
            return pd.DataFrame()
        
        filtered_chunks = []
        processed_rows = 0
        kept_rows = 0
        
        try:
            with pd.read_csv(input_file, chunksize=self.chunk_size) as reader:
                for chunk_num, chunk in enumerate(reader):
                    processed_rows += len(chunk)
                    
                    # Filter by symbol (e.g., BTC)
                    if symbol_filter:
                        symbol_mask = chunk['symbol'].str.startswith(symbol_filter)
                        chunk = chunk[symbol_mask]
                    
                    if chunk.empty:
                        continue
                    
                    # Convert timestamps with error handling
                    try:
                        chunk['local_timestamp'] = pd.to_datetime(chunk['local_timestamp'], 
                                                                unit='us', utc=True)
                        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], 
                                                          unit='us', utc=True)
                    except Exception as e:
                        logger.warning(f"Error converting timestamps in chunk {chunk_num}: {e}")
                        continue
                    
                    # Filter by time windows
                    mask = pd.Series(False, index=chunk.index)
                    
                    for start, end in time_ranges:
                        mask |= ((chunk['local_timestamp'] >= start) & 
                                (chunk['local_timestamp'] <= end))
                    
                    filtered = chunk[mask]
                    
                    if not filtered.empty:
                        filtered_chunks.append(filtered)
                        kept_rows += len(filtered)
                    
                    # Progress update
                    if (chunk_num + 1) % 10 == 0 or processed_rows >= total_rows:
                        progress = (processed_rows / total_rows) * 100
                        logger.info(f"Progress: {progress:.1f}% - "
                                  f"Processed {processed_rows:,} rows, kept {kept_rows:,} rows")
        
        except Exception as e:
            logger.error(f"Error processing data from {input_file.name}: {e}")
            raise
        
        # Combine filtered chunks
        if filtered_chunks:
            logger.info(f"Combining filtered data from {input_file.name}...")
            result = pd.concat(filtered_chunks, ignore_index=True)
            result = result.sort_values('local_timestamp')
            
            logger.info(f"Final dataset from {input_file.name}: {len(result):,} rows")
            return result
        else:
            logger.warning(f"No data found in the specified time windows for {input_file.name}!")
            return pd.DataFrame()
    

    def save_results(self, data: pd.DataFrame, date_str: str, suffix: str = "event_windows") -> str:
        """Save filtered data with date-specific filename"""
        
        if data.empty:
            logger.warning("No data to save")
            return ""
        
        # Convert date format from YYYY-MM-DD to YYYYMMDD
        formatted_date = date_str.replace('-', '')
        
        # Generate output filename: BTC_event_windows_20210213.csv
        output_file = self.output_dir / f"BTC_{suffix}_{formatted_date}.csv"
        
        logger.info(f"Saving to {output_file}...")
        
        try:
            data.to_csv(output_file, index=False)
            
            # Calculate file size
            file_size_mb = output_file.stat().st_size / 1024 / 1024
            memory_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
            
            logger.info(f"File saved successfully!")
            logger.info(f"File size: {file_size_mb:.1f} MB")
            logger.info(f"Memory usage: {memory_size_mb:.1f} MB")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    def process_event_windows(self, 
                            symbol_filter: str = 'BTC',
                            save_suffix: str = "event_windows") -> List[str]:
        """
        Main processing function for multiple files
        
        Parameters:
        - symbol_filter: Symbol prefix to filter (e.g., 'BTC')
        - save_suffix: Suffix for output filename
        
        Returns:
        - List of paths to output files
        """
        
        logger.info("="*60)
        logger.info("STARTING MULTI-FILE EVENT WINDOW PROCESSING")
        logger.info("="*60)
        
        output_files = []
        
        try:
            for input_file in self.input_files:
                logger.info(f"\n{'-'*40}")
                logger.info(f"Processing file: {input_file.name}")
                logger.info(f"{'-'*40}")
                
                # Extract date from filename
                try:
                    file_date = self.extract_date_from_filename(input_file)
                    logger.info(f"Extracted date: {file_date}")
                except ValueError as e:
                    logger.error(f"Skipping file {input_file.name}: {e}")
                    continue
                
                # Load events for this specific date
                events = self.load_events([file_date])
                if events.empty:
                    logger.warning(f"No events found for {file_date}, skipping...")
                    continue
                
                # Create time windows
                time_ranges = self.create_time_windows(events)
                if not time_ranges:
                    logger.warning(f"No time windows created for {file_date}, skipping...")
                    continue
                
                # Filter data for this file
                filtered_data = self.filter_options_data(input_file, time_ranges, symbol_filter)
                if filtered_data.empty:
                    logger.warning(f"No filtered data for {file_date}, skipping...")
                    continue
                
                # Save results with date-specific filename
                output_file = self.save_results(filtered_data, file_date, save_suffix)
                if output_file:
                    output_files.append(output_file)
                
                # Print summary for this file
                total_original = self.count_total_rows(input_file)
                if total_original > 0:
                    reduction = (1 - len(filtered_data)/total_original) * 100
                    logger.info(f"\nFile {input_file.name} Summary:")
                    logger.info(f"  Original rows: {total_original:,}")
                    logger.info(f"  Filtered rows: {len(filtered_data):,}")
                    logger.info(f"  Data reduction: {reduction:.1f}%")
            
            # Print final summary
            logger.info("\n" + "="*60)
            logger.info("MULTI-FILE PROCESSING COMPLETE")
            logger.info("="*60)
            logger.info(f"Processed files: {len(self.input_files)}")
            logger.info(f"Output files created: {len(output_files)}")
            
            if output_files:
                logger.info("\nOutput files:")
                for output_file in output_files:
                    logger.info(f"  {output_file}")
            
            logger.info("="*60)
            
            return output_files
            
        except Exception as e:
            logger.error(f"Multi-file processing failed: {e}")
            raise
    
    def process_single_file_by_date(self, 
                                  target_date: str,
                                  symbol_filter: str = 'BTC',
                                  save_suffix: str = "event_windows") -> str:
        """
        Process a single file for a specific date (backward compatibility)
        
        Parameters:
        - target_date: Date string in 'YYYY-MM-DD' format
        - symbol_filter: Symbol prefix to filter (e.g., 'BTC')
        - save_suffix: Suffix for output filename
        
        Returns:
        - Path to output file
        """
        
        # Find the input file that matches the target date
        target_file = None
        for input_file in self.input_files:
            try:
                file_date = self.extract_date_from_filename(input_file)
                if file_date == target_date:
                    target_file = input_file
                    break
            except ValueError:
                continue
        
        if not target_file:
            raise ValueError(f"No input file found for date {target_date}")
        
        logger.info("="*60)
        logger.info(f"PROCESSING SINGLE FILE FOR DATE: {target_date}")
        logger.info("="*60)
        
        try:
            # Load events
            events = self.load_events([target_date])
            if events.empty:
                return ""
            
            # Create time windows
            time_ranges = self.create_time_windows(events)
            if not time_ranges:
                return ""
            
            # Filter data
            filtered_data = self.filter_options_data(target_file, time_ranges, symbol_filter)
            if filtered_data.empty:
                return ""
            
            # Save results
            output_file = self.save_results(filtered_data, target_date, save_suffix)
            
            # Print final summary
            total_original = self.count_total_rows(target_file)
            if total_original > 0:
                reduction = (1 - len(filtered_data)/total_original) * 100
                logger.info("\n" + "="*60)
                logger.info("PROCESSING COMPLETE")
                logger.info("="*60)
                logger.info(f"Original rows: {total_original:,}")
                logger.info(f"Filtered rows: {len(filtered_data):,}")
                logger.info(f"Data reduction: {reduction:.1f}%")
                logger.info(f"Output file: {output_file}")
                logger.info("="*60)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

def main():
    """Automatically find and process all OPTIONS files in directory"""
    
    import glob
    
    # Directory containing your files
    data_dir = r"D:\2024-2025\Research\Paper6\Paper_6_News_Effect\1_Modeling\Data\BTC\event_windows"
    
    # Find all OPTIONS files
    pattern = os.path.join(data_dir, "*_OPTIONS.csv")
    input_files = glob.glob(pattern)
    
    print(f"Found {len(input_files)} OPTIONS files:")
    for file in input_files:
        print(f"  {os.path.basename(file)}")
    
    if not input_files:
        print("No OPTIONS files found!")
        return
    
    # Configuration
    config = {
        'event_file': os.path.join(r"D:\2024-2025\Research\Paper6\Paper_6_News_Effect\1_Modeling\Data\BTC", "BTC_Event_Timeline.csv"),
        'input_files': input_files,
        'output_dir': data_dir,
        'window_minutes': 60,
        'chunk_size': 100000
    }
    
    # Create processor
    processor = EventWindowProcessor(**config)
    
    # Process all files
    output_files = processor.process_event_windows(
        symbol_filter='BTC',
        save_suffix='event_windows'
    )
    
    print(f"\nBatch processing complete! Created {len(output_files)} output files:")
    for output_file in output_files:
        print(f"  {os.path.basename(output_file)}")


if __name__ == "__main__":
    main()