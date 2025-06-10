import os
import json
import argparse
from datetime import datetime, timedelta
import traceback
from analyze import (load_and_prepare_data, analyze_multiple_windows, detect_multiple_timeframe_trends,
                     create_detailed_pattern_graph, create_longterm_pattern_figure, display_results,
                     analyze_price_data, find_significant_price_patterns)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_convertion import process_bitcoin_data
from data_fetcher import download_btc_raw_data


def create_improved_overview_plot(dates, prices, all_patterns, num_patterns, date_str, time_str=None):
    """
    Create an improved overview plot with better pattern visualization
    """
    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot the full price series
    ax.plot(dates, prices, color='black', alpha=0.6, linewidth=1.5, label='Price')

    # Define colors for different patterns
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'teal', 'navy', 'olive', 'maroon']

    # Define different marker shapes for each point type
    markers = {
        'A': {'marker': 'o', 'size': 140, 'zorder': 5},  # Circle for A
        'B': {'marker': 's', 'size': 140, 'zorder': 5},  # Square for B
        'C': {'marker': '^', 'size': 140, 'zorder': 5},  # Triangle for C
        'D': {'marker': 'd', 'size': 140, 'zorder': 5}  # Diamond for D
    }

    # Offset multipliers for label positioning
    offset_y = [-0.003, 0.003, -0.005, 0.005, -0.007]

    # Highlight each pattern window
    for i, (result, analysis, window_info) in enumerate(all_patterns[:num_patterns]):
        if i >= len(colors):
            break

        # Get the pattern's time window
        start_idx = window_info['start_idx']
        end_idx = window_info['end_idx']
        color = colors[i % len(colors)]

        # Highlight the window area
        ax.axvspan(dates[start_idx], dates[end_idx - 1], alpha=0.15, color=color)

        # Add window labels
        window_middle = start_idx + (end_idx - start_idx) // 2
        top_y = max(prices) + (max(prices) - min(prices)) * 0.03
        ax.text(dates[window_middle], top_y,
                f"Pattern {i + 1}", color=color, fontweight='bold',
                ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

        # Process the pattern(s) in this window
        patterns_to_plot = result if isinstance(result, list) else [result]

        for pattern in patterns_to_plot:
            # Get pattern direction and status
            direction = pattern.get('direction', 'unknown')
            status = pattern.get('status', 'unknown')

            # Get pattern points
            points = {
                'A': pattern['A'],
                'B': pattern['B'],
                'C': pattern['C'],
                'D': pattern['D']
            }

            # Plot each point
            for label, (idx, price) in points.items():
                # Calculate global index
                global_idx = start_idx + idx

                # Ensure index is in range
                if global_idx >= len(dates):
                    global_idx = min(global_idx, len(dates) - 1)

                # Use different marker for failed patterns
                if status == 'failed' and label == 'C':
                    marker = 'x'
                    size = 160
                else:
                    marker = markers[label]['marker']
                    size = markers[label]['size']

                # Plot the point
                ax.scatter(dates[global_idx], price,
                           color=color,
                           marker=marker,
                           s=size,
                           zorder=markers[label]['zorder'],
                           edgecolors='black')

                y_offset = offset_y[i % len(offset_y)] * (max(prices) - min(prices))

                # Create label text
                if label == 'A':
                    label_text = f"A{i + 1}"
                elif label == 'B':
                    label_text = f"B{i + 1}"
                elif label == 'C':
                    label_text = f"C{i + 1}" + (" (F)" if status == "failed" else "")
                elif label == 'D':
                    label_text = f"D{i + 1}" + (" (C)" if status == "completed" else "")

                # Add the text label
                ax.text(dates[global_idx], price + y_offset, label_text,
                        fontsize=14, fontweight='bold', color=color,
                        ha='center', va='center', zorder=10,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

            # Connect points with lines
            line_indices = [start_idx + points['A'][0], start_idx + points['B'][0],
                            start_idx + points['C'][0], start_idx + points['D'][0]]
            line_indices = [min(idx, len(dates) - 1) for idx in line_indices]
            line_dates = [dates[idx] for idx in line_indices]
            line_prices = [points['A'][1], points['B'][1], points['C'][1], points['D'][1]]

            linestyle = '--' if status == 'failed' else '-'
            ax.plot(line_dates, line_prices, linestyle=linestyle, color=color, alpha=0.7, linewidth=1.5)

    legend_elements = []
    for label, marker_info in markers.items():
        legend_elements.append(plt.Line2D([0], [0], marker=marker_info['marker'], color='w',
                                          markerfacecolor='black', markersize=10, label=f'Point {label}'))

    legend_elements.append(plt.Line2D([0], [0], linestyle='-', color='black', label='Completed/In Progress'))
    legend_elements.append(plt.Line2D([0], [0], linestyle='--', color='black', label='Failed'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

    # Set title
    title = f"Top {num_patterns} Patterns Overview - {date_str}"
    if time_str:
        title += f" (around {time_str})"
    ax.set_title(title, fontsize=16, fontweight='bold')

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(f"{title}\nGenerated: {current_time}", fontsize=16, fontweight='bold')

    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    # Add grid
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def add_pattern_info_panel(fig, all_patterns, num_patterns):
    """
    Add information panel to the overview plot
    """
    info_ax = fig.add_axes([0.1, 0.02, 0.8, 0.15])
    info_ax.axis('off')

    info_text = "PATTERN SUMMARY:\n\n"

    for i, (result, analysis, window_info) in enumerate(all_patterns[:num_patterns]):
        if i >= num_patterns:
            break

        patterns_to_summarize = result if isinstance(result, list) else [result]

        for pattern in patterns_to_summarize:
            direction = pattern.get('direction', 'unknown')
            status = pattern.get('status', 'unknown')
            move_pct = pattern.get('initial_move_pct', 0)
            retrace_pct = pattern.get('retracement_pct', 0)

            pattern_desc = f"Pattern {i + 1}: {direction.upper()}-trend {status.upper()}"
            pattern_desc += f" (Move: {move_pct:.1f}%, Retracement: {retrace_pct:.1f}%)"

            # Extract point prices for easy reference
            a_price = pattern['A'][1]
            b_price = pattern['B'][1]
            c_price = pattern['C'][1]
            d_price = pattern['D'][1]

            point_info = f"    A: ${a_price:.2f}, B: ${b_price:.2f}, C: ${c_price:.2f}, D: ${d_price:.2f}"

            # Add to info text
            info_text += f"{pattern_desc}\n{point_info}\n"

    # Display the info text
    info_ax.text(0, 1, info_text, fontsize=11, va='top', family='monospace')

    return fig


def validate_date(date_str):
    """Validate date format"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def validate_time(time_str):
    """Validate time format"""
    try:
        datetime.strptime(time_str, '%H:%M:%S')
        return True
    except ValueError:
        return False


def load_default_params():
    """Load default parameters for analysis"""
    return {
        "min_change": 0.01,
        "window_sizes": [100, 300, 600, 1000],
        "overlap": 10,
        "top_patterns": 5,
        "time_window": 120,
        "pattern_detection": {
            "retracement_target": 0.5,
            "retracement_tolerance": 0.02,
            "completion_extension": 0.236,
            "failure_level": 0.764,
            "min_move_multiplier": 2.0
        }
    }


def load_params_from_json(json_file):
    """Load parameters from JSON file"""
    try:
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                params = json.load(f)
            print(f"Loaded parameters from {json_file}")
            return params
        else:
            print(f"JSON file {json_file} not found, using default parameters")
            default_params = load_default_params()
            try:
                with open(json_file, 'w') as f:
                    json.dump(default_params, f, indent=4)
                print(f"Created default parameters file at: {os.path.abspath(json_file)}")
            except Exception as e:
                print(f"Could not create default parameters file: {e}")
            return default_params
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return load_default_params()


def add_timestamp_to_plot(fig, ax):
    """Add current timestamp to plot title"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_title = ax.get_title()

    if current_title:
        new_title = f"{current_title}\nGenerated: {current_time}"
    else:
        new_title = f"Pattern Analysis\nGenerated: {current_time}"

    ax.set_title(new_title)
    return fig, ax


def create_timestamped_results_dir(base_dir, date_str, time_str=None):
    """Create timestamped results directory"""
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    if time_str:
        folder_name = f"{date_str}_{time_str.replace(':', '')}_run_{current_time}"
    else:
        folder_name = f"{date_str}_run_{current_time}"

    results_dir = os.path.join(base_dir, folder_name)

    print(f"\n=== DEBUG: CREATING RESULTS DIRECTORY ===")
    print(f"Base directory: {base_dir}")
    print(f"Folder name: {folder_name}")
    print(f"Full path: {results_dir}")

    os.makedirs(base_dir, exist_ok=True)

    try:
        os.makedirs(results_dir, exist_ok=True)
        print(f"Directory created successfully: {os.path.exists(results_dir)}")
    except Exception as e:
        print(f"ERROR creating directory: {e}")
        results_dir = os.path.join(os.path.expanduser("~"), "Desktop", "btc_analysis_results", folder_name)
        print(f"Trying fallback path: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)

    print(f"\n=== RESULTS LOCATION ===")
    print(f"All analysis results will be saved to:")
    print(f"{os.path.abspath(results_dir)}")
    print(f"===========================\n")

    return results_dir


def save_run_log(results_dir, date_str, target_time, params, patterns_count, analysis_date=None):
    """Save log of current run"""
    if analysis_date is None:
        analysis_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    log_file = os.path.join(results_dir, f"btc_data_{date_str}_run_{analysis_date}.txt")

    with open(log_file, 'w') as f:
        f.write(f"Bitcoin Fibonacci Analysis Run Log\n")
        f.write(f"================================\n\n")
        f.write(f"Data Date: {date_str}\n")
        if target_time:
            f.write(f"Target Time: {target_time}\n")
        f.write(f"Analysis Date: {analysis_date}\n\n")

        f.write(f"Parameters:\n")
        f.write(f"  - Minimum Change Threshold: {params['min_change']}\n")
        f.write(f"  - Window Sizes: {params['window_sizes']}\n")
        f.write(f"  - Overlap Percentage: {params['overlap']}\n")
        f.write(f"  - Top Patterns Count: {params['top_patterns']}\n")
        if target_time:
            f.write(f"  - Time Window (minutes): {params['time_window']}\n")

        # Add pattern detection parameters
        pattern_config = params.get('pattern_detection', {})
        f.write(f"\nPattern Detection:\n")
        f.write(f"  - Retracement Target: {pattern_config.get('retracement_target', 0.5) * 100:.1f}%\n")
        f.write(f"  - Retracement Tolerance: ±{pattern_config.get('retracement_tolerance', 0.02) * 100:.1f}%\n")
        f.write(f"  - Completion Extension: {pattern_config.get('completion_extension', 0.236) * 100:.1f}%\n")
        f.write(f"  - Failure Level: {pattern_config.get('failure_level', 0.764) * 100:.1f}%\n")
        f.write(f"  - Min Move Multiplier: {pattern_config.get('min_move_multiplier', 2.0)}\n")

        f.write(f"\nResults:\n")
        f.write(f"  - Total Patterns Detected: {patterns_count}\n")

        f.write(f"\nOutput Files:\n")
        f.write(f"  - patterns_overview_{analysis_date}.png\n")
        f.write(f"  - longterm_patterns_{analysis_date}.png\n")
        for i in range(min(params['top_patterns'], patterns_count)):
            f.write(f"  - pattern_{i + 1}_{analysis_date}.png\n")
            f.write(f"  - pattern_{i + 1}_detailed_{analysis_date}.png\n")

    print(f"Run log saved to: {os.path.abspath(log_file)}")
    return log_file


def build_pattern_config(params_dict):
    """Build pattern configuration from parameters dictionary"""
    pattern_detection = params_dict.get('pattern_detection', {})

    config = {
        "retracement_target": pattern_detection.get("retracement_target", 0.5),
        "retracement_tolerance": pattern_detection.get("retracement_tolerance", 0.02),
        "completion_extension": pattern_detection.get("completion_extension", 0.236),
        "failure_level": pattern_detection.get("failure_level", 0.764),
        "min_move_multiplier": pattern_detection.get("min_move_multiplier", 2.0)
    }

    return config


def test_pattern_detection(data_file, date_str, time_str=None):
    """Test the pattern detection on your data"""
    print("=== TESTING PATTERN DETECTION ===")

    # Load data
    if time_str:
        prices, dates, df = load_and_prepare_data(data_file)
        # Apply time filtering if needed
    else:
        prices, dates, df = load_and_prepare_data(data_file)

    if len(prices) < 10:
        print("Not enough data points for testing")
        return None

    print(f"Testing with {len(prices)} data points")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Price range: ${min(prices):.2f} to ${max(prices):.2f}")

    # Configuration for pattern detection
    pattern_config = {
        "retracement_target": 0.5,
        "retracement_tolerance": 0.02,
        "completion_extension": 0.236,
        "failure_level": 0.764,
        "min_move_multiplier": 2.0
    }

    # Test the pattern detection
    patterns = find_significant_price_patterns(prices, dates, min_change_pct=0.005, config=pattern_config)

    if patterns:
        print(f"\n=== PATTERN DETECTION RESULTS ===")
        print(f"Found {len(patterns)} patterns:")

        for i, pattern in enumerate(patterns):
            print(f"\nPattern {i + 1}:")
            print(f"  Status: {pattern['status']}")
            print(f"  A (absolute low): ${pattern['A'][1]:.2f} at index {pattern['A'][0]}")
            print(f"  B (high): ${pattern['B'][1]:.2f} at index {pattern['B'][0]}")
            print(f"  C (50% retrace): ${pattern['C'][1]:.2f} at index {pattern['C'][0]}")
            print(f"  D: ${pattern['D'][1]:.2f} at index {pattern['D'][0]}")
            print(f"  Initial move: {pattern['initial_move_pct']:.2f}%")
            print(f"  Retracement: {pattern['retracement_pct']:.2f}%")
            print(f"  Failure level: ${pattern['failure_level']:.2f}")
            print(f"  Completion level: ${pattern['completion_level']:.2f}")

        # Create visualization
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot price series
        ax.plot(dates, prices, 'k-', alpha=0.7, linewidth=1, label='Price')

        # Plot patterns
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, pattern in enumerate(patterns):
            color = colors[i % len(colors)]

            # Plot ABCD points
            points = ['A', 'B', 'C', 'D']
            markers = ['o', 's', '^', 'd']

            for j, point in enumerate(points):
                idx, price = pattern[point]
                marker = 'x' if pattern['status'] == 'failed' and point == 'C' else markers[j]
                size = 150 if pattern['status'] == 'failed' and point == 'C' else 120

                ax.scatter(dates[idx], price, color=color, marker=marker,
                           s=size, zorder=5, edgecolor='black')
                ax.text(dates[idx], price, f'{point}{i + 1}', fontsize=12,
                        fontweight='bold', ha='center', va='bottom')

            # Draw lines connecting points
            x_coords = [dates[pattern[p][0]] for p in points]
            y_coords = [pattern[p][1] for p in points]

            linestyle = '--' if pattern['status'] == 'failed' else '-'
            ax.plot(x_coords, y_coords, color=color, linestyle=linestyle,
                    linewidth=2, alpha=0.7, label=f'Pattern {i + 1} ({pattern["status"]})')

            # Draw key levels
            ax.axhline(y=pattern['failure_level'], color='red', linestyle=':',
                       alpha=0.7, label=f'Failure Level {i + 1}')
            ax.axhline(y=pattern['completion_level'], color='green', linestyle=':',
                       alpha=0.7, label=f'Completion Level {i + 1}')

        ax.set_title(f'Pattern Detection Results - {date_str}' +
                     (f' around {time_str}' if time_str else ''))
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return patterns
    else:
        print("No patterns found")
        return None


def parse_args():
    """Parse command line arguments with validation"""
    parser = argparse.ArgumentParser(description='Bitcoin Fibonacci Trend Analysis')
    parser.add_argument('date', type=str, help='Date of data to analyze (YYYY-MM-DD format)')

    # Use dest parameter to ensure consistent attribute names
    parser.add_argument('--time', dest='time', type=str, help='Specific time to analyze (HH:MM:SS format, optional)')
    parser.add_argument('--params', dest='params', type=str, default='parameters.json',
                        help='Path to JSON file with analysis parameters (default: parameters.json)')
    parser.add_argument('--time-window', dest='time_window', type=int,
                        help='Time window in minutes around specified time')
    parser.add_argument('--min-change', dest='min_change', type=float, help='Minimum price change threshold')
    parser.add_argument('--window-sizes', dest='window_sizes', type=str,
                        help='Window sizes for analysis (comma-separated)')
    parser.add_argument('--overlap', dest='overlap', type=int, help='Window overlap percentage')
    parser.add_argument('--top-patterns', dest='top_patterns', type=int, help='Number of top patterns to display')
    parser.add_argument('--save', dest='save', action='store_true',
                        help='Save plots as PNG files instead of displaying them')
    parser.add_argument('--test', dest='test', action='store_true',
                        help='Test the pattern detection method')

    # Input/output paths
    parser.add_argument('--input-dir', dest='input_dir', type=str, default='C:/Users/admin/Desktop/btc_data',
                        help='Input directory for raw data')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='C:/Users/admin/Desktop/btc_minute_data',
                        help='Output directory for processed data')
    parser.add_argument('--temp-dir', dest='temp_dir', type=str, default='C:/Users/admin/Desktop/btc_data',
                        help='Temporary directory for extraction')
    parser.add_argument('--results-dir', dest='results_dir', type=str,
                        help='Custom results directory location')

    args = parser.parse_args()

    # Validate date format
    if not validate_date(args.date):
        parser.error(f"Invalid date format: {args.date}. Please use YYYY-MM-DD format.")

    # Validate time format if provided
    if args.time and not validate_time(args.time):
        parser.error(f"Invalid time format: {args.time}. Please use HH:MM:SS format.")

    # Create parameter dictionary
    params = load_params_from_json(args.params)

    # Override with command-line args if provided
    if args.min_change is not None:
        params['min_change'] = args.min_change

    if args.window_sizes is not None:
        try:
            window_sizes = [int(x) for x in args.window_sizes.split(',')]
            if any(size <= 0 for size in window_sizes):
                parser.error("Window sizes must be positive integers.")
            params['window_sizes'] = window_sizes
        except ValueError:
            parser.error(f"Invalid window sizes: {args.window_sizes}. Please use comma-separated integers.")

    if args.overlap is not None:
        if args.overlap < 0 or args.overlap > 100:
            parser.error(f"Overlap percentage must be between 0 and 100.")
        params['overlap'] = args.overlap

    if args.top_patterns is not None:
        params['top_patterns'] = args.top_patterns

    if args.time_window is not None:
        params['time_window'] = args.time_window

    # Add params to the args object
    args.params_dict = params

    return args

def check_desktop_minute_data(date_str):
    """Check if processed minute data exists on desktop"""
    desktop_paths = [
        os.path.expanduser('/mnt/c/Users/admin/Desktop/btc_minute_data'),
        os.path.expanduser('C:\\Users\\admin\\Desktop\\btc_minute_data'),
        os.path.expanduser('C:/Users/admin/Desktop/btc_minute_data'),
    ]

    for desktop_path in desktop_paths:
        if os.path.exists(desktop_path):
            expected_file = os.path.join(desktop_path, f"btc_minute_data_{date_str}.csv")
            if os.path.exists(expected_file) and os.path.getsize(expected_file) > 0:
                print(f"Found existing processed data on desktop: {os.path.abspath(expected_file)}")
                return expected_file

    return None


def main():
    # Parse command line arguments
    args = parse_args()
    date_str = args.date
    time_str = args.time
    save_plots = True
    params = args.params_dict

    # Build pattern configuration
    pattern_config = build_pattern_config(params)

    print("=== PATTERN DETECTION CONFIGURATION ===")
    print(f"Retracement target: {pattern_config['retracement_target'] * 100:.1f}%")
    print(f"Retracement tolerance: ±{pattern_config['retracement_tolerance'] * 100:.1f}%")
    print(f"Completion extension: {pattern_config['completion_extension'] * 100:.1f}%")
    print(f"Failure level: {pattern_config['failure_level'] * 100:.1f}%")
    print(f"Min move multiplier: {pattern_config['min_move_multiplier']}")
    print("=======================================\n")

    # Create analysis timestamp
    analysis_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


    # Define paths
    input_dir = args.input_dir
    output_dir = args.output_dir
    temp_dir = args.temp_dir

    if args.results_dir:
        base_dir = args.results_dir
    else:
        base_dir = os.path.join(output_dir, "results")

    results_dir = create_timestamped_results_dir(base_dir, date_str, time_str)

    # Ensure directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    data_file = os.path.join(output_dir, f"btc_minute_data_{date_str}.csv")


    # Check for existing data
    desktop_data_file = check_desktop_minute_data(date_str)
    if desktop_data_file:
        print(f"Using existing processed data found on desktop: {os.path.abspath(desktop_data_file)}")
        data_file = desktop_data_file
    elif not os.path.exists(data_file):
        print(f"Processed data file not found: {data_file}")
        print(f"Checking if raw data exists for date: {date_str}")

        raw_file = os.path.join(input_dir, f"btc_raw_{date_str}.tar.gz")

        if not os.path.exists(raw_file):
            print(f"Raw data file not found: {raw_file}")
            print(f"Attempting to download data for date: {date_str}")

            download_result = download_btc_raw_data(date_str, input_dir)

            if not download_result['success']:
                print("Download failed. Cannot proceed with analysis.")
                return

        print(f"Processing raw data for date: {date_str}")
        result_files = process_bitcoin_data(
            input_dir=input_dir,
            output_dir=output_dir,
            temp_dir=temp_dir,
            interval_minutes=1,
            cleanup=True
        )

        if not result_files or date_str not in result_files:
            print(f"Processing failed or no data found for date: {date_str}")
            return
    else:
        print(f"Found existing processed data file: {os.path.abspath(data_file)}")

    # Check if in test mode
    if hasattr(args, 'test') and args.test:
        print("Running in test mode...")
        test_pattern_detection(data_file, date_str, time_str)
        return

    # Extract analysis parameters
    min_change_threshold = params['min_change']
    window_sizes = params['window_sizes']
    overlap_percent = params['overlap']
    top_n = params['top_patterns']
    time_window_minutes = params['time_window']

    # Load and prepare the data
    try:
        # Define time filtering function
        def filter_data_by_time(df, time_str, time_window_minutes=30):
            """Filter dataframe to include only data around specified time"""
            target_time = datetime.strptime(time_str, '%H:%M:%S').time()

            if isinstance(df['timestamp'].iloc[0], str):
                df['time_obj'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time())
            else:
                df['time_obj'] = df['timestamp'].dt.time

            def time_diff_minutes(t1, t2):
                t1_seconds = t1.hour * 3600 + t1.minute * 60 + t1.second
                t2_seconds = t2.hour * 3600 + t2.minute * 60 + t2.second

                diff_seconds = min(abs(t1_seconds - t2_seconds),
                                   24 * 3600 - abs(t1_seconds - t2_seconds))
                return diff_seconds / 60

            df['time_diff'] = df['time_obj'].apply(lambda x: time_diff_minutes(x, target_time))
            filtered_df = df[df['time_diff'] <= time_window_minutes]

            filtered_df = filtered_df.drop(['time_obj', 'time_diff'], axis=1)

            return filtered_df

        # Load data with or without time filtering
        if time_str:
            print(f"Analyzing data for date {date_str} at time {time_str}")
            prices, dates, df = load_and_prepare_data(data_file)

            filtered_df = filter_data_by_time(df, time_str, time_window_minutes)

            if len(filtered_df) < 10:
                print(f"Warning: Not enough data points around time {time_str}.")
                print(f"Using all data for the day instead.")
            else:
                prices = filtered_df["price"].tolist()
                dates = filtered_df["timestamp"].tolist()
                df = filtered_df
                print(f"Filtered to {len(prices)} data points around {time_str}")
        else:
            print(f"Analyzing all data for date {date_str}")
            prices, dates, df = load_and_prepare_data(data_file)

        if len(prices) < 10:
            print("ERROR: Not enough valid data points for analysis")
            return

        print(f"Loaded {len(prices)} data points for analysis")

        # Test single pattern detection first
        print("\n=== TESTING SINGLE PATTERN DETECTION ===")
        single_patterns = find_significant_price_patterns(
            prices, dates, min_change_threshold, config=pattern_config
        )

        if single_patterns:
            print(f"Single pattern detection found {len(single_patterns)} patterns")

            # Create visualization for single pattern detection
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.plot(dates, prices, 'k-', alpha=0.7, linewidth=1, label='Price')

            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, pattern in enumerate(single_patterns):
                color = colors[i % len(colors)]

                # Plot ABCD points
                points = ['A', 'B', 'C', 'D']
                markers = ['o', 's', '^', 'd']
                sizes = [150, 150, 150, 150]

                for j, point in enumerate(points):
                    idx, price = pattern[point]

                    # Use X marker for failed C points
                    marker = 'x' if pattern['status'] == 'failed' and point == 'C' else markers[j]
                    size = 200 if pattern['status'] == 'failed' and point == 'C' else sizes[j]

                    ax.scatter(dates[idx], price, color=color, marker=marker,
                               s=size, zorder=5, edgecolor='black', linewidth=2)

                    # Add labels
                    label_text = f'{point}{i + 1}'
                    if point == 'C' and pattern['status'] == 'failed':
                        label_text += ' (FAILED)'

                    ax.text(dates[idx], price, label_text, fontsize=12,
                            fontweight='bold', ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                # Connect points with lines
                x_coords = [dates[pattern[p][0]] for p in points]
                y_coords = [pattern[p][1] for p in points]

                linestyle = '--' if pattern['status'] == 'failed' else '-'
                linewidth = 3 if pattern['status'] == 'failed' else 2
                ax.plot(x_coords, y_coords, color=color, linestyle=linestyle,
                        linewidth=linewidth, alpha=0.8,
                        label=f'Pattern {i + 1} ({pattern["status"].upper()})')

                # Draw key levels
                ax.axhline(y=pattern['failure_level'], color='darkred', linestyle=':',
                           linewidth=2, alpha=0.7, label=f'76.4% Failure Level')
                ax.axhline(y=pattern['completion_level'], color='darkgreen', linestyle=':',
                           linewidth=2, alpha=0.7, label=f'-23.6% Completion Level')

                # Add level annotations
                ax.text(dates[0], pattern['failure_level'],
                        f'Failure: ${pattern["failure_level"]:.2f}',
                        fontsize=10, color='darkred', fontweight='bold')
                ax.text(dates[0], pattern['completion_level'],
                        f'Completion: ${pattern["completion_level"]:.2f}',
                        fontsize=10, color='darkgreen', fontweight='bold')

            title = f'Pattern Detection - {date_str}'
            if time_str:
                title += f' around {time_str}'
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)

            plt.tight_layout()

            # Show the figure
            plt.figure(fig.number)
            plt.draw()
            plt.pause(0.5)

            # Save if requested
            if save_plots:
                single_pattern_file = os.path.join(results_dir, f"single_pattern_detection_{analysis_timestamp}.png")
                plt.savefig(single_pattern_file, dpi=300, bbox_inches='tight')
                print(f"Saved single pattern detection to: {os.path.abspath(single_pattern_file)}")

        # Perform multi-window analysis
        print("\n=== PERFORMING MULTI-WINDOW ANALYSIS ===")
        all_patterns = analyze_multiple_windows(
            prices,
            dates,
            window_sizes=window_sizes,
            overlap_percent=overlap_percent,
            min_change_threshold=min_change_threshold,
            allow_multiple_patterns=True,
            detect_long_failures=True,
            pattern_config=pattern_config
        )

        # Display results for the top N patterns
        num_patterns = min(top_n, len(all_patterns))

        if num_patterns == 0:
            print("No patterns detected in any window.")
            save_run_log(results_dir, date_str, time_str, params, 0, analysis_timestamp)
            return

        print(f"Detected {len(all_patterns)} patterns across all windows, showing top {num_patterns}")

        # Save log file
        log_file = save_run_log(results_dir, date_str, time_str, params, len(all_patterns), analysis_timestamp)

        # Turn on interactive mode
        plt.ion()

        # Create detailed graphs for each pattern
        for i, (result, analysis, window_info) in enumerate(all_patterns[:num_patterns]):
            print(f"\n--- Pattern {i + 1} ---")
            print(f"Window: {window_info['window_size']} data points")
            print(f"Period: {window_info['start_date']} to {window_info['end_date']}")
            print(f"Analysis: {analysis}")

            # Get trends for this window
            window_prices = window_info['window_prices']
            window_dates = window_info['window_dates']
            trends = detect_multiple_timeframe_trends(window_prices)

            # Create the detailed graph
            fig = create_detailed_pattern_graph(i + 1, result, analysis, window_info, trends)

            # Show and save
            plt.figure(fig.number)
            plt.draw()
            plt.pause(0.5)

            if save_plots:
                detailed_file = os.path.join(results_dir, f"pattern_{i + 1}_detailed_{analysis_timestamp}.png")
                plt.savefig(detailed_file, dpi=300, bbox_inches='tight')
                print(f"Saved detailed pattern {i + 1} to: {os.path.abspath(detailed_file)}")
                plt.close(fig)

            # Create individual pattern plots
            fig, ax = display_results(window_prices, window_dates, result, analysis, trends)

            # Set title with window information
            title = f"Pattern {i + 1} - {window_info['start_date'].strftime('%Y-%m-%d')} to {window_info['end_date'].strftime('%Y-%m-%d')}"
            if time_str:
                title += f" (around {time_str})"
            ax.set_title(title)

            # Add timestamp to plot title
            add_timestamp_to_plot(fig, ax)

            # Show the figure
            plt.figure(fig.number)
            plt.draw()
            plt.pause(0.5)

            # Save if requested
            if save_plots:
                pattern_file = os.path.join(results_dir, f"pattern_{i + 1}_{analysis_timestamp}.png")
                plt.savefig(pattern_file, dpi=300, bbox_inches='tight')
                print(f"Saved pattern {i + 1} to: {os.path.abspath(pattern_file)}")
                plt.close(fig)

        # Create overview plots
        if all_patterns:
            fig, ax = create_improved_overview_plot(dates, prices, all_patterns, num_patterns, date_str, time_str)
            fig = add_pattern_info_panel(fig, all_patterns, num_patterns)

            # Create long-term patterns figure
            longterm_fig = create_longterm_pattern_figure(prices, dates, all_patterns)
            if longterm_fig:
                if save_plots:
                    plt.figure(longterm_fig.number)
                    longterm_file = os.path.join(results_dir, f"longterm_patterns_{analysis_timestamp}.png")
                    plt.savefig(longterm_file, dpi=300, bbox_inches='tight')
                    print(f"Saved long-term patterns to: {os.path.abspath(longterm_file)}")
                    plt.close(longterm_fig)
                else:
                    plt.figure(longterm_fig.number)
                    plt.draw()
                    plt.pause(0.5)

            # Save overview
            if save_plots:
                overview_file = os.path.join(results_dir, f"patterns_overview_{analysis_timestamp}.png")
                plt.savefig(overview_file, dpi=300, bbox_inches='tight')
                print(f"Saved patterns overview to: {os.path.abspath(overview_file)}")
                plt.close(fig)

                print(f"\n=== ANALYSIS COMPLETE ===")
                print(f"All files have been saved to:")
                print(f"{os.path.abspath(results_dir)}")
                print(f"Run log: {os.path.basename(log_file)}")
                print(f"Pattern Detection Method: Absolute Low with 50% Retracement")
                print(f"Total Patterns Found: {len(all_patterns)}")
                print(f"Significant Patterns Analyzed: {num_patterns}")
                print(f"========================")
            else:
                plt.figure(fig.number)
                plt.draw()

            plt.ioff()
            print("\nDisplaying all pattern charts. Close the windows to exit.")
            plt.show()

    except Exception as e:
        print(f"ERROR: An exception occurred during analysis: {e}")
        traceback.print_exc()
        error_log = os.path.join(results_dir, f"error_log_{analysis_timestamp}.txt")
        with open(error_log, 'w') as f:
            f.write(f"Error during analysis: {e}\n\n")
            f.write(f"Pattern Detection Configuration:\n")
            f.write(f"  - Retracement Target: {pattern_config['retracement_target'] * 100:.1f}%\n")
            f.write(f"  - Retracement Tolerance: ±{pattern_config['retracement_tolerance'] * 100:.1f}%\n")
            f.write(f"  - Completion Extension: {pattern_config['completion_extension'] * 100:.1f}%\n")
            f.write(f"  - Failure Level: {pattern_config['failure_level'] * 100:.1f}%\n")
            f.write(f"  - Min Move Multiplier: {pattern_config['min_move_multiplier']}\n\n")
            f.write(traceback.format_exc())
        print(f"Error log saved to: {os.path.abspath(error_log)}")


if __name__ == "__main__":
    try:
        main()
        print("\nAnalysis completed successfully!")
        print("✅ Pattern Detection: A = Absolute Low, C = 50% Retracement, D = Above A")
        print("✅ Pattern Status: Failed/Completed/In-Progress based on 76.4% and -23.6% levels")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        traceback.print_exc()
        print("\nFor help with command line options, run: python main.py --help")

    print("\n" + "=" * 70)
    print("USAGE EXAMPLES:")
    print("=" * 70)
    print("  python main.py 2023-01-15                       # Analyze all data for January 15, 2023")
    print("  python main.py 2023-01-15 --time 14:30:00       # Analyze data around 2:30 PM")
    print("  python main.py 2023-01-15 --params custom.json  # Use custom parameters from JSON file")
    print("  python main.py 2023-01-15 --min-change 0.01     # Override min change threshold")
    print("  python main.py 2023-01-15 --save                # Save plots instead of displaying them")
    print("  python main.py 2023-01-15 --test                # Test pattern detection method")
    print("  python main.py 2023-01-15 --results-dir C:/path/to/custom/dir  # Custom results location")
    print("=" * 70)
    print("PATTERN DETECTION REQUIREMENTS:")
    print("  ✓ A: Absolute lowest point in dataset")
    print("  ✓ B: Highest point after A with valid 50% retracement")
    print("  ✓ C: 50% retracement from A-B move (±2% tolerance)")
    print("  ✓ D: Above A level, determined by completion/failure criteria")
    print("  ✓ Failed: When price breaks 76.4% level and can't reach completion")
    print("  ✓ Completed: When price reaches -23.6% extension above B")
    print("=" * 70)