import os
import json
import argparse
from datetime import datetime, timedelta
import traceback
from analyze import load_and_prepare_data, analyze_multiple_windows, detect_multiple_timeframe_trends, \
    create_detailed_pattern_graph, create_longterm_pattern_figure, display_results, analyze_price_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_convertion import process_bitcoin_data
from data_fetcher import download_btc_raw_data


def create_improved_overview_plot(dates, prices, all_patterns, num_patterns, date_str, time_str=None):
    """
    Create an improved overview plot with better pattern visualization and less overlap
    """
    fig, ax = plt.subplots(figsize=(18, 12))  # Use a larger figure size

    # Plot the full price series
    ax.plot(dates, prices, color='black', alpha=0.6, linewidth=1.5, label='Price')

    # Define colors for different patterns
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'teal', 'navy', 'olive', 'maroon']

    # Define different marker shapes for each point type with larger sizes
    markers = {
        'A': {'marker': 'o', 'size': 140, 'zorder': 5},  # Circle for A
        'B': {'marker': 's', 'size': 140, 'zorder': 5},  # Square for B
        'C': {'marker': '^', 'size': 140, 'zorder': 5},  # Triangle for C
        'D': {'marker': 'd', 'size': 140, 'zorder': 5}  # Diamond for D
    }

    # Offset multipliers for label positioning to avoid overlap
    offset_x = 0  # Horizontal offset
    offset_y = [-0.003, 0.003, -0.005, 0.005, -0.007]  # Different vertical offsets for each pattern

    # Highlight each pattern window with less opacity
    for i, (result, analysis, window_info) in enumerate(all_patterns[:num_patterns]):
        if i >= len(colors):
            break

        # Get the pattern's time window
        start_idx = window_info['start_idx']
        end_idx = window_info['end_idx']
        color = colors[i % len(colors)]

        # Highlight the window area with reduced opacity
        ax.axvspan(dates[start_idx], dates[end_idx - 1], alpha=0.15, color=color)

        # Add window labels at the top of the chart instead of in the middle
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

            # Plot each point with appropriate marker and offset the label
            for label, (idx, price) in points.items():
                # Calculate global index
                global_idx = start_idx + idx

                # Ensure index is in range
                if global_idx >= len(dates):
                    print(f"Warning: Point {label} has index out of range.")
                    global_idx = min(global_idx, len(dates) - 1)

                # Plot the point
                ax.scatter(dates[global_idx], price,
                           color=color,
                           marker=markers[label]['marker'],
                           s=markers[label]['size'],
                           zorder=markers[label]['zorder'],
                           edgecolors='black')

                # Add offset to label position to reduce overlap
                y_offset = offset_y[i % len(offset_y)] * (max(prices) - min(prices))

                # Create a more informative label
                if label == 'A':
                    label_text = f"A{i + 1}"
                elif label == 'B':
                    label_text = f"B{i + 1}"
                elif label == 'C':
                    label_text = f"C{i + 1}" + (" (F)" if status == "failed" else "")
                elif label == 'D':
                    label_text = f"D{i + 1}" + (" (C)" if status == "completed" else "")

                # Add the text label with offset
                ax.text(dates[global_idx], price + y_offset, label_text,
                        fontsize=14, fontweight='bold', color=color,
                        ha='center', va='center', zorder=10,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

            # Connect the points with lines to better show pattern shape
            line_indices = [start_idx + points['A'][0], start_idx + points['B'][0],
                            start_idx + points['C'][0], start_idx + points['D'][0]]
            # Clamp indices to valid range
            line_indices = [min(idx, len(dates) - 1) for idx in line_indices]
            line_dates = [dates[idx] for idx in line_indices]
            line_prices = [points['A'][1], points['B'][1], points['C'][1], points['D'][1]]

            # Use dashed line if failed, solid otherwise
            linestyle = '--' if status == 'failed' else '-'
            ax.plot(line_dates, line_prices, linestyle=linestyle, color=color, alpha=0.7, linewidth=1.5)

    # Add pattern legend
    legend_elements = []
    for label, marker_info in markers.items():
        legend_elements.append(plt.Line2D([0], [0], marker=marker_info['marker'], color='w',
                                          markerfacecolor='black', markersize=10, label=f'Point {label}'))

    # Add pattern status legend
    legend_elements.append(plt.Line2D([0], [0], linestyle='-', color='black', label='Completed/In Progress'))
    legend_elements.append(plt.Line2D([0], [0], linestyle='--', color='black', label='Failed'))

    # Position legend at top left to avoid overlapping with patterns
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)

    # Set title
    title = f"Top {num_patterns} Patterns Overview - {date_str}"
    if time_str:
        title += f" (around {time_str})"
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Add current timestamp to title
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(f"{title}\nGenerated: {current_time}", fontsize=16, fontweight='bold')

    # Format axes
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
    Add a non-overlapping information panel to the overview plot
    """
    # Create a new axis for the info panel (20% of the bottom of the figure)
    info_ax = fig.add_axes([0.1, 0.02, 0.8, 0.15])
    info_ax.axis('off')  # Hide axes

    # Create information text
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

            # Format pattern description
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
    """Validate that the date is in the correct format"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def validate_time(time_str):
    """Validate that the time is in the correct format"""
    try:
        datetime.strptime(time_str, '%H:%M:%S')
        return True
    except ValueError:
        return False


def load_default_params():
    """Load default parameters for analysis"""
    return {
        "min_change": 0.005,
        "window_sizes": [200, 400, 500],
        "overlap": 50,
        "top_patterns": 5,
        "time_window": 30
    }


def load_params_from_json(json_file):
    """Load parameters from a JSON file"""
    try:
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                params = json.load(f)
            print(f"Loaded parameters from {json_file}")
            return params
        else:
            print(f"JSON file {json_file} not found, using default parameters")
            # If the file doesn't exist, create it with default parameters
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


# New function to add timestamps to plot titles
def add_timestamp_to_plot(fig, ax):
    """Add current timestamp to the plot title"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_title = ax.get_title()

    # Add timestamp to title
    if current_title:
        new_title = f"{current_title}\nGenerated: {current_time}"
    else:
        new_title = f"Pattern Analysis\nGenerated: {current_time}"

    ax.set_title(new_title)
    return fig, ax


# New function to create timestamped directories
def create_timestamped_results_dir(base_dir, date_str, time_str=None):
    """Create a timestamped results directory"""
    # Current timestamp for uniqueness
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create directory path with timestamp
    if time_str:
        # If time is provided, include in folder name
        folder_name = f"{date_str}_{time_str.replace(':', '')}_run_{current_time}"
    else:
        # Otherwise just use date and timestamp
        folder_name = f"{date_str}_run_{current_time}"

    results_dir = os.path.join(base_dir, folder_name)

    # Debug output
    print(f"\n=== DEBUG: CREATING RESULTS DIRECTORY ===")
    print(f"Base directory: {base_dir}")
    print(f"Folder name: {folder_name}")
    print(f"Full path: {results_dir}")

    # Make sure base_dir exists first
    os.makedirs(base_dir, exist_ok=True)

    # Now create the results directory
    try:
        os.makedirs(results_dir, exist_ok=True)
        print(f"Directory created successfully: {os.path.exists(results_dir)}")
    except Exception as e:
        print(f"ERROR creating directory: {e}")
        # Fallback to a simpler path if there's an issue
        results_dir = os.path.join(os.path.expanduser("~"), "Desktop", "btc_analysis_results", folder_name)
        print(f"Trying fallback path: {results_dir}")
        os.makedirs(results_dir, exist_ok=True)

    print(f"\n=== RESULTS LOCATION ===")
    print(f"All analysis results will be saved to:")
    print(f"{os.path.abspath(results_dir)}")
    print(f"===========================\n")

    return results_dir


def save_run_log(results_dir, date_str, target_time, params, patterns_count, analysis_date=None):
    """Save a log of the current run"""
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


def parse_args():
    """Parse command line arguments with validation"""
    parser = argparse.ArgumentParser(description='Bitcoin Fibonacci Trend Analysis')
    parser.add_argument('date', type=str, help='Date of data to analyze (YYYY-MM-DD format)')
    parser.add_argument('--time', type=str, help='Specific time to analyze (HH:MM:SS format, optional)')
    parser.add_argument('--params', type=str, default='params.json',
                        help='Path to JSON file with analysis parameters (default: params.json)')
    parser.add_argument('--time-window', type=int, help='Time window in minutes around specified time')
    parser.add_argument('--min-change', type=float, help='Minimum price change threshold')
    parser.add_argument('--window-sizes', type=str, help='Window sizes for analysis (comma-separated)')
    parser.add_argument('--overlap', type=int, help='Window overlap percentage')
    parser.add_argument('--top-patterns', type=int, help='Number of top patterns to display')
    parser.add_argument('--save', action='store_true',
                        help='Save plots as PNG files instead of displaying them')

    # Input/output paths
    parser.add_argument('--input-dir', type=str, default='C:/Users/admin/Desktop/btc_data',
                        help='Input directory for raw data')
    parser.add_argument('--output-dir', type=str, default='C:/Users/admin/Desktop/btc_minute_data',
                        help='Output directory for processed data')
    parser.add_argument('--temp-dir', type=str, default='C:/Users/admin/Desktop/btc_data',
                        help='Temporary directory for extraction')
    parser.add_argument('--results-dir', type=str,
                        help='Custom results directory location')

    args = parser.parse_args()

    # Validate date format
    if not validate_date(args.date):
        parser.error(f"Invalid date format: {args.date}. Please use YYYY-MM-DD format.")

    # Validate time format if provided
    if args.time and not validate_time(args.time):
        parser.error(f"Invalid time format: {args.time}. Please use HH:MM:SS format.")

    # Create a parameter dictionary by combining JSON and command-line args
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
    """
    Check if processed minute data exists on the desktop

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        file_path if found, None otherwise
    """
    # Define possible desktop paths for different OS environments
    desktop_paths = [
        # Windows WSL path
        os.path.expanduser('/mnt/c/Users/admin/Desktop/btc_minute_data'),
        # Regular Windows path
        os.path.expanduser('C:\\Users\\admin\\Desktop\\btc_minute_data'),
        # Alternative Windows path
        os.path.expanduser('C:/Users/admin/Desktop/btc_minute_data'),
    ]

    # Check if any of the desktop paths exist
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
    save_plots = True  # Whether to save plots as PNG files
    params = args.params_dict

    # Create an analysis timestamp
    analysis_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Define paths
    input_dir = args.input_dir
    output_dir = args.output_dir
    temp_dir = args.temp_dir

    # Create a results directory with timestamp for output figures and logs
    if args.results_dir:
        # Use the custom results directory as the base if provided
        base_dir = args.results_dir
    else:
        # Otherwise use the default base path
        base_dir = os.path.join(output_dir, "results")

    # Create a timestamped subdirectory for this run
    results_dir = create_timestamped_results_dir(base_dir, date_str, time_str)

    # Ensure other directories exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Define expected processed data file path
    data_file = os.path.join(output_dir, f"btc_minute_data_{date_str}.csv")

    # First check if processed data exists on desktop
    desktop_data_file = check_desktop_minute_data(date_str)
    if desktop_data_file:
        print(f"Using existing processed data found on desktop: {os.path.abspath(desktop_data_file)}")
        data_file = desktop_data_file
    # If not found on desktop, check in the output directory
    elif not os.path.exists(data_file):
        print(f"Processed data file not found: {data_file}")
        print(f"Checking if raw data exists for date: {date_str}")

        # Define raw data file path
        raw_file = os.path.join(input_dir, f"btc_raw_{date_str}.tar.gz")

        # Check if raw data exists
        if not os.path.exists(raw_file):
            print(f"Raw data file not found: {raw_file}")
            print(f"Attempting to download data for date: {date_str}")

            # Download raw data
            download_result = download_btc_raw_data(date_str, input_dir)

            if not download_result['success']:
                print("Download failed. Cannot proceed with analysis.")
                return

        # Process the raw data
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

    # At this point, we should have the processed data file
    if not os.path.exists(data_file):
        print(f"Error: Expected processed data file not found: {data_file}")
        return

    # Extract analysis parameters
    min_change_threshold = params['min_change']
    window_sizes = params['window_sizes']
    overlap_percent = params['overlap']
    top_n = params['top_patterns']
    time_window_minutes = params['time_window']

    # Load and prepare the data
    try:
        # Default to using all data points
        date_range = None
        index_range = None

        print(f"Loading data from: {os.path.abspath(data_file)}")

        # Define the time filtering function
        def filter_data_by_time(df, time_str, time_window_minutes=30):
            """
            Filter the dataframe to include only data around the specified time.
            """
            from datetime import datetime, timedelta

            # Parse the provided time
            target_time = datetime.strptime(time_str, '%H:%M:%S').time()

            # Convert strings to datetime.time objects if needed
            if isinstance(df['timestamp'].iloc[0], str):
                df['time_obj'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%H:%M:%S').time())
            else:
                # Assume it's already a datetime
                df['time_obj'] = df['timestamp'].dt.time

            # Calculate time difference in minutes
            def time_diff_minutes(t1, t2):
                # Convert time objects to total seconds
                t1_seconds = t1.hour * 3600 + t1.minute * 60 + t1.second
                t2_seconds = t2.hour * 3600 + t2.minute * 60 + t2.second

                # Calculate difference and handle day boundary
                diff_seconds = min(abs(t1_seconds - t2_seconds),
                                   24 * 3600 - abs(t1_seconds - t2_seconds))
                return diff_seconds / 60

            # Apply the filter based on time difference
            df['time_diff'] = df['time_obj'].apply(lambda x: time_diff_minutes(x, target_time))
            filtered_df = df[df['time_diff'] <= time_window_minutes]

            # Clean up temporary columns
            filtered_df = filtered_df.drop(['time_obj', 'time_diff'], axis=1)

            return filtered_df

        # Load the data with or without time filtering
        if time_str:
            print(f"Analyzing data for date {date_str} at time {time_str}")
            # Load the data first
            prices, dates, df = load_and_prepare_data(data_file, date_range, index_range)

            # Apply time filtering
            filtered_df = filter_data_by_time(df, time_str, time_window_minutes)

            if len(filtered_df) < 10:
                print(f"Warning: Not enough data points around time {time_str}.")
                print(f"Using all data for the day instead.")
            else:
                # Extract prices and dates from filtered dataframe
                prices = filtered_df["price"].tolist()
                dates = filtered_df["timestamp"].tolist()
                df = filtered_df
                print(f"Filtered to {len(prices)} data points around {time_str}")
        else:
            print(f"Analyzing all data for date {date_str}")
            prices, dates, df = load_and_prepare_data(data_file, date_range, index_range)

        # Make sure we have enough data points
        if len(prices) < 10:
            print("ERROR: Not enough valid data points for analysis")
            return

        print(f"Loaded {len(prices)} data points for analysis")

        # Perform multi-window analysis
        print("Performing multi-window Fibonacci analysis...")
        all_patterns = analyze_multiple_windows(
            prices,
            dates,
            window_sizes=window_sizes,
            overlap_percent=overlap_percent,
            min_change_threshold=min_change_threshold,
            allow_multiple_patterns=True,
            detect_long_failures=True
        )
        print(all_patterns)

        # Display results for the top N patterns
        num_patterns = min(top_n, len(all_patterns))

        if num_patterns == 0:
            print("No patterns detected in any window.")
            save_run_log(results_dir, date_str, time_str, params, 0, analysis_timestamp)
            return

        print(f"Detected {len(all_patterns)} patterns, showing top {num_patterns}")

        # Always save a log file with the run details
        log_file = save_run_log(results_dir, date_str, time_str, params, len(all_patterns), analysis_timestamp)

        # Turn on interactive mode for real-time display
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

            # Add timestamp to detailed graph title (first axis is the main chart)
            chart_ax = fig.get_axes()[0]
            add_timestamp_to_plot(fig, chart_ax)

            # Always show the figure in real-time
            plt.figure(fig.number)
            plt.draw()
            plt.pause(0.5)  # Pause briefly to allow display to update

            # Save if requested
            if save_plots:
                # Save the detailed graph with timestamp in filename
                detailed_file = os.path.join(results_dir, f"pattern_{i + 1}_detailed_{analysis_timestamp}.png")
                plt.savefig(detailed_file)
                print(f"Saved detailed pattern {i + 1} to: {os.path.abspath(detailed_file)}")
                plt.close(fig)  # Close after saving

            # Create individual pattern plots
            fig, ax = display_results(window_prices, window_dates, result, analysis, trends)

            # Set a title that includes the window information
            title = f"Pattern {i + 1} - {window_info['start_date'].strftime('%Y-%m-%d')} to {window_info['end_date'].strftime('%Y-%m-%d')}"
            if time_str:
                title += f" (around {time_str})"
            ax.set_title(title)

            # Add timestamp to plot title
            add_timestamp_to_plot(fig, ax)

            # Always show the figure in real-time
            plt.figure(fig.number)
            plt.draw()
            plt.pause(0.5)  # Pause briefly to allow display to update

            # Show all patterns in a summary figure
            if all_patterns:
                # Create improved overview plot
                fig, ax = create_improved_overview_plot(dates, prices, all_patterns, num_patterns, date_str, time_str)

                # Add information panel with pattern details
                fig = add_pattern_info_panel(fig, all_patterns, num_patterns)

                # Create and handle long-term patterns figure
                longterm_fig = create_longterm_pattern_figure(prices, dates, all_patterns)
                if longterm_fig:
                    if save_plots:
                        plt.figure(longterm_fig.number)
                        longterm_file = os.path.join(results_dir, f"longterm_patterns_{analysis_timestamp}.png")
                        plt.savefig(longterm_file)
                        print(f"Saved long-term patterns to: {os.path.abspath(longterm_file)}")
                        plt.close(longterm_fig)
                    else:
                        plt.figure(longterm_fig.number)
                        plt.draw()
                        plt.pause(0.5)  # Pause briefly to allow display to update

                # Save if requested
                if save_plots:
                    overview_file = os.path.join(results_dir, f"patterns_overview_{analysis_timestamp}.png")
                    plt.savefig(overview_file)
                    print(f"Saved patterns overview to: {os.path.abspath(overview_file)}")
                    plt.close(fig)

                    print(f"\n=== ANALYSIS COMPLETE ===")
                    print(f"All files have been saved to:")
                    print(f"{os.path.abspath(results_dir)}")
                    print(f"Run log: {os.path.basename(log_file)}")
                    print(f"========================")
                else:
                    # Display in real-time
                    plt.figure(fig.number)
                    plt.draw()

                # Turn off interactive mode and show all figures
                plt.ioff()
                print("\nDisplaying all pattern charts. Close the windows to exit.")
                plt.show()  # This will block until all figures are closed
    except Exception as e:
        print(f"ERROR: An exception occurred during analysis: {e}")
        traceback.print_exc()
        # Still try to save a log file with the error
        error_log = os.path.join(results_dir, f"error_log_{analysis_timestamp}.txt")
        with open(error_log, 'w') as f:
            f.write(f"Error during analysis: {e}\n\n")
            f.write(traceback.format_exc())
        print(f"Error log saved to: {os.path.abspath(error_log)}")


if __name__ == "__main__":
    try:
        main()
        print("\nAnalysis completed successfully!")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        traceback.print_exc()
        print("\nFor help with command line options, run: python main.py --help")

    print("\nUsage examples:")
    print("  python main.py 2023-01-15                       # Analyze all data for January 15, 2023")
    print("  python main.py 2023-01-15 --time 14:30:00       # Analyze data around 2:30 PM")
    print("  python main.py 2023-01-15 --params custom.json  # Use custom parameters from JSON file")
    print("  python main.py 2023-01-15 --min-change 0.01     # Override min change threshold")
    print("  python main.py 2023-01-15 --save                # Save plots instead of displaying them")
    print("  python main.py 2023-01-15 --results-dir C:/path/to/custom/dir  # Custom results location")