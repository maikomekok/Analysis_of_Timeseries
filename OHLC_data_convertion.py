import os
import pandas as pd
import numpy as np
import tarfile
import shutil
from datetime import datetime
import re


def calculate_mid_price(df):
    """Calculate mid-price for a dataframe - handles swapped bid/ask columns"""
    all_mid_prices = []

    for _, row in df.iterrows():
        row_mid_prices = []

        for i in range(1, 21):
            actual_bid_col = f'ask_prc{i}'  # Actual bid price is in ask column
            actual_ask_col = f'bid_prc{i}'  # Actual ask price is in bid column

            if (actual_bid_col in row and actual_ask_col in row and
                    pd.notna(row[actual_bid_col]) and pd.notna(row[actual_ask_col])):
                mid_price = (row[actual_bid_col] + row[actual_ask_col]) / 2
                row_mid_prices.append(mid_price)

        if row_mid_prices:
            all_mid_prices.append(np.mean(row_mid_prices))

    if all_mid_prices:
        return np.mean(all_mid_prices)
    return np.nan


def extract_datetime_from_csv(df):
    """Extract datetime from the CSV date column"""
    try:
        if 'date' in df.columns and len(df) > 0:
            # Get the first timestamp from the CSV
            return pd.to_datetime(df['date'].iloc[0])
    except Exception as e:
        print(f"Could not parse datetime from CSV data: {e}")
    return None


def extract_date_from_filename(filename):
    """Extract date from tar.gz filename"""
    basename = os.path.basename(filename)

    # Try YYYYMMDD format at start of filename
    if len(basename) >= 8 and basename[:8].isdigit():
        try:
            return datetime.strptime(basename[:8], '%Y%m%d').strftime('%Y-%m-%d')
        except:
            pass

    # Try to find YYYYMMDD pattern anywhere in filename
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', basename)
    if date_match:
        year, month, day = date_match.groups()
        try:
            return datetime.strptime(f"{year}{month}{day}", '%Y%m%d').strftime('%Y-%m-%d')
        except:
            pass

    # Fallback to today
    return datetime.now().strftime('%Y-%m-%d')


def extract_tarfile(tar_file_path, extract_dir):
    try:
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)

        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)

        print(f"Successfully extracted {tar_file_path} to {extract_dir}")
        return True
    except Exception as e:
        print(f"Error extracting {tar_file_path}: {e}")
        return False


def process_directory(input_dir, output_file_template, interval_seconds=60, filter_suffix='16'):
    """
    Process all CSV files in directory into minute/second OHLC data files with date in filename

    Args:
        input_dir (str): Directory containing CSV files
        output_file_template (str): Template for output filename (date will be appended)
        interval_seconds (int): Interval in seconds (60 for 1-minute, 1 for 1-second, etc.)
        filter_suffix (str): Only process files ending with this suffix

    Returns:
        dict: Dictionary of dates processed and their corresponding output files
    """
    results = []
    dates_found = set()
    output_files = {}

    # Find CSV files with the specified suffix filter
    csv_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                # Check if filename (without .csv) ends with the filter suffix
                name_without_ext = os.path.splitext(file)[0]
                if filter_suffix is None or name_without_ext.endswith(filter_suffix):
                    csv_files.append(os.path.join(root, file))

    print(f"Found {len(csv_files)} CSV files ending with '{filter_suffix}' to process")

    if not csv_files:
        print(f"No CSV files ending with '{filter_suffix}' found!")
        return {}

    file_count = 0
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()  # Clean column names

            if len(df) == 0:
                continue

            # Extract datetime from CSV data
            file_timestamp = extract_datetime_from_csv(df)
            if file_timestamp is None:
                continue

            dates_found.add(file_timestamp.date())

            # Process each row in the CSV (each order book snapshot)
            for _, row in df.iterrows():
                try:
                    # Parse the timestamp for this row
                    row_timestamp = pd.to_datetime(row['date'])

                    # Calculate mid price using corrected bid/ask (accounting for swap)
                    actual_bid = row['ask_prc1']  # Bid price is in ask column
                    actual_ask = row['bid_prc1']  # Ask price is in bid column

                    if pd.notna(actual_bid) and pd.notna(actual_ask):
                        mid_price = (actual_bid + actual_ask) / 2

                        # Calculate volumes (also swapped)
                        bid_volume = sum([row.get(f'ask_vol{i}', 0) for i in range(1, 21)
                                          if pd.notna(row.get(f'ask_vol{i}', 0))])  # Actual bid volumes
                        ask_volume = sum([row.get(f'bid_vol{i}', 0) for i in range(1, 21)
                                          if pd.notna(row.get(f'bid_vol{i}', 0))])  # Actual ask volumes

                        results.append({
                            'timestamp': row_timestamp,
                            'mid_price': mid_price,
                            'best_bid': actual_bid,
                            'best_ask': actual_ask,
                            'spread': actual_ask - actual_bid,
                            'bid_volume': bid_volume,
                            'ask_volume': ask_volume,
                            'total_volume': bid_volume + ask_volume,
                            'date': row_timestamp.date()
                        })

                except Exception as e:
                    print(f"Error processing row in {file_path}: {e}")
                    continue

            file_count += 1
            if file_count % 10 == 0:
                print(f"Processed {file_count} files")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not results:
        print("No valid results found!")
        return {}

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('timestamp')

    # Floor to specified interval (60s = 1 minute, 1s = 1 second, etc.)
    result_df['interval'] = result_df['timestamp'].dt.floor(f'{interval_seconds}s')

    # Process each date separately
    for date in dates_found:
        date_str = date.strftime('%Y-%m-%d')
        date_df = result_df[result_df['date'] == date]

        if len(date_df) == 0:
            continue

        # Group by interval and create OHLC data
        grouped = date_df.groupby('interval').agg({
            'mid_price': ['first', 'max', 'min', 'last', 'count'],  # OHLC + count
            'spread': 'mean',
            'bid_volume': 'sum',
            'ask_volume': 'sum',
            'total_volume': 'sum',
            'best_bid': 'last',
            'best_ask': 'last'
        }).reset_index()

        # Flatten column names
        grouped.columns = ['timestamp', 'open', 'high', 'low', 'close', 'num_ticks',
                           'avg_spread', 'bid_volume', 'ask_volume', 'total_volume',
                           'best_bid', 'best_ask']

        # Format timestamp for output
        if interval_seconds >= 60:
            # For minute data and above, show time as HH:MM:SS
            grouped['time_only'] = grouped['timestamp'].dt.strftime('%H:%M:%S')
        else:
            # For second data, include milliseconds if available
            grouped['time_only'] = grouped['timestamp'].dt.strftime('%H:%M:%S.%f').str[:-3]

        # Create final output dataframe
        output_df = pd.DataFrame({
            'timestamp': grouped['time_only'],
            'open': grouped['open'],
            'high': grouped['high'],
            'low': grouped['low'],
            'close': grouped['close'],
            'volume': grouped['total_volume'],
            'bid_volume': grouped['bid_volume'],
            'ask_volume': grouped['ask_volume'],
            'num_ticks': grouped['num_ticks'],
            'avg_spread': grouped['avg_spread'],
            'best_bid': grouped['best_bid'],
            'best_ask': grouped['best_ask']
        })

        # Determine interval name for filename
        if interval_seconds == 60:
            interval_name = "1minute"
        elif interval_seconds == 1:
            interval_name = "1second"
        else:
            interval_name = f"{interval_seconds}second"

        # Create output filename with date
        output_file = output_file_template.replace('.csv', f'_{interval_name}_{date_str}.csv')
        output_files[date_str] = output_file

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save to CSV
        output_df.to_csv(output_file, index=False)
        print(f"Results for {date_str} saved to: {output_file} with {len(output_df)} {interval_name} intervals")

    print(f"Successfully processed {file_count} files for {len(dates_found)} dates")
    return output_files


def process_bitcoin_data(input_dir, output_dir='processed_data', temp_dir='temp_extract',
                         interval_seconds=60, cleanup=True):
    """
    Complete Bitcoin data processing function that:
    1. Finds tar.gz files in input directory
    2. Extracts them to a temporary directory  
    3. Processes CSV files (ending with '16') into OHLC data with date in filename
    4. Cleans up temporary files (if cleanup=True)
    5. Returns paths to all generated output files

    Args:
        input_dir (str): Directory containing tar.gz files to process
        output_dir (str): Directory where output CSV files will be saved
        temp_dir (str): Temporary directory for extraction
        interval_seconds (int): Interval in seconds (60 for 1-minute, 1 for 1-second, etc.)
        cleanup (bool): Whether to delete temporary files after processing

    Returns:
        dict: Dictionary of all generated output files by date
    """
    all_results = {}

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all tar.gz files
    tar_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.tar.gz'):
            tar_files.append(os.path.join(input_dir, file))

    if not tar_files:
        print(f"No tar.gz files found in {input_dir}")
        return {}

    print(f"Found {len(tar_files)} tar.gz files to process")

    # Process each tar.gz file
    for tar_file in tar_files:
        filename = os.path.basename(tar_file)
        print(f"\nProcessing: {filename}")

        # Extract date from filename
        date_str = extract_date_from_filename(filename)
        print(f"Processing data for date: {date_str}")

        # Create extraction directory
        extract_dir = os.path.join(temp_dir, os.path.splitext(os.path.splitext(filename)[0])[0])

        try:
            print(f"Extracting {tar_file}...")
            if not extract_tarfile(tar_file, extract_dir):
                print(f"Failed to extract {tar_file}, skipping...")
                continue

            print(f"Processing CSV files from {tar_file}...")

            # Create output template
            if interval_seconds == 60:
                output_template = os.path.join(output_dir, "btc_1minute_data.csv")
            elif interval_seconds == 1:
                output_template = os.path.join(output_dir, "btc_1second_data.csv")
            else:
                output_template = os.path.join(output_dir, f"btc_{interval_seconds}second_data.csv")

            # Process the extracted files
            file_results = process_directory(extract_dir, output_template, interval_seconds, filter_suffix='16')

            # Add results to overall results
            all_results.update(file_results)

        except Exception as e:
            print(f"Error processing {tar_file}: {e}")
        finally:
            # Clean up temporary files
            if cleanup and os.path.exists(extract_dir):
                print(f"Cleaning up temporary files in {extract_dir}...")
                shutil.rmtree(extract_dir)

    # Final summary
    if all_results:
        print(f"\n✅ Successfully processed {len(tar_files)} tar.gz files")
        print(f"📊 Generated {len(all_results)} output files:")
        for date, output_file in all_results.items():
            file_size = os.path.getsize(output_file) / 1024 if os.path.exists(output_file) else 0
            print(f"  - {date}: {output_file} ({file_size:.1f} KB)")
    else:
        print("❌ No output files were generated")

    return all_results


# Example usage:
if __name__ == "__main__":
    # Process data with 1-minute intervals (60 seconds)
    results = process_bitcoin_data(
        input_dir="path/to/your/tar/files",
        output_dir="btc_1minute_data",
        interval_seconds=60  # 60 seconds = 1 minute
    )

    # Or process with 1-second intervals
    # results = process_bitcoin_data(
    #     input_dir="path/to/your/tar/files", 
    #     output_dir="btc_1second_data",
    #     interval_seconds=1  # 1 second intervals
    # )