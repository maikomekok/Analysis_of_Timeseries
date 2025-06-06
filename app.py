from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import json
from datetime import datetime
import sys

# Add your existing modules
sys.path.append('.')  # Adjust path as needed
from analyze import load_and_prepare_data, analyze_multiple_windows, detect_multiple_timeframe_trends
from data_convertion import process_bitcoin_data
from data_fetcher import download_btc_raw_data

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

DATA_DIR = "C:/Users/admin/Desktop/btc_minute_data"
RESULTS_DIR = "C:/Users/admin/Desktop/btc_minute_data/results"
RAW_DATA_DIR = "C:/Users/admin/Desktop/btc_data"
TEMP_DIR = "C:/Users/admin/Desktop/btc_data"


def load_parameters(params_file='parameters.json'):
    """Load analysis parameters from your JSON file"""
    try:
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                params = json.load(f)
            print(f"Loaded parameters from {params_file}")
            return params
        else:
            print(f"Parameters file {params_file} not found, using defaults")
            # Default parameters matching your typical setup
            return {
                "min_change": 0.003,
                "window_sizes": [200, 400, 600, 1000],
                "overlap": 20,
                "top_patterns": 5,
                "time_window": 30,
                "pattern_detection": {
                    "retracement_target": 0.5,
                    "failure_level": 0.8,
                    "completion_extension": 0.236,
                    "retracement_tolerance": 0.2,
                    "use_strict_swing_points": True,
                    "min_move_multiplier": 2.0
                }
            }
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return {
            "min_change": 0.003,
            "window_sizes": [200, 400, 600, 1000],
            "overlap": 50,
            "top_patterns": 5,
            "time_window": 30
        }


@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        with open('bitcoin_analyzer.html', 'r') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return """
        <h1>Bitcoin Analyzer</h1>
        <p>Please make sure bitcoin_analyzer.html is in the same directory as app.py</p>
        <p>Files needed:</p>
        <ul>
            <li>app.py (this file)</li>
            <li>bitcoin_analyzer.html</li>
            <li>parameters.json (your config)</li>
            <li>analyze.py (your analysis code)</li>
        </ul>
        """


@app.route('/api/analyze', methods=['POST'])
def analyze_bitcoin_data():
    """Main API endpoint for Bitcoin analysis using your parameters"""
    try:
        # Load your parameters
        params = load_parameters()

        # Get request data
        data = request.json
        date_str = data.get('date')
        time_str = data.get('time')
        time_window = data.get('timeWindow', params.get('time_window', 30))

        # Use your min_change parameter, but allow frontend override
        min_change = data.get('minChange', params.get('min_change', 0.003))

        if not date_str:
            return jsonify({'error': 'Date is required'}), 400

        print(
            f"Analyzing data for {date_str} with parameters: min_change={min_change}, window_sizes={params.get('window_sizes')}")

        # Check if processed data exists (try multiple possible locations)
        possible_files = [
            os.path.join(DATA_DIR, f"btc_minute_data_{date_str}.csv"),
            f"btc_minute_data_{date_str}.csv",  # Current directory
            os.path.join(".", f"btc_minute_data_{date_str}.csv")
        ]

        data_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                data_file = file_path
                print(f"Found data file: {data_file}")
                break

        if not data_file:
            print(f"No processed data found for {date_str}, attempting to download and process...")
            # Try to download and process data
            raw_result = download_and_process_data(date_str, params)
            if not raw_result['success']:
                return jsonify(
                    {'error': f'Could not obtain data for {date_str}: {raw_result.get("error", "Unknown error")}'}), 404
            data_file = raw_result.get('file')

        # Load and analyze the data using your actual code
        try:
            print(f"Loading data from: {data_file}")
            prices, dates, df = load_and_prepare_data(data_file)
            print(f"Loaded {len(prices)} data points")

            # Filter by time if specified
            if time_str:
                print(f"Filtering data around time {time_str} with window {time_window} minutes")
                prices, dates, df = filter_data_by_time(prices, dates, df, time_str, time_window)
                print(f"Filtered to {len(prices)} data points")

            if len(prices) < 10:
                return jsonify({'error': 'Not enough data points for analysis'}), 400

            # Use YOUR window sizes and parameters from parameters.json
            window_sizes = params.get('window_sizes', [200, 400, 600, 1000])
            overlap_percent = params.get('overlap', 50)
            pattern_config = params.get('pattern_detection', {})

            print(f"Running analysis with window_sizes={window_sizes}, overlap={overlap_percent}%")
            print(f"Pattern config: {pattern_config}")

            # Perform analysis using your existing code with your parameters
            all_patterns = analyze_multiple_windows(
                prices,
                dates,
                window_sizes=window_sizes,
                overlap_percent=overlap_percent,
                min_change_threshold=min_change,
                allow_multiple_patterns=True,
                detect_long_failures=True
            )

            print(f"Found {len(all_patterns)} patterns")

            # Get trends using your code
            trends = detect_multiple_timeframe_trends(prices)
            print(f"Detected trends: {trends}")

            # Format response
            top_patterns_count = params.get('top_patterns', 5)
            response_data = {
                'success': True,
                'date': date_str,
                'time': time_str,
                'timeWindow': time_window,
                'dataPoints': len(prices),
                'priceRange': {
                    'min': float(min(prices)),
                    'max': float(max(prices)),
                    'range_pct': float((max(prices) - min(prices)) / min(prices) * 100)
                },
                'trends': trends,
                'patterns': format_patterns_for_frontend(all_patterns[:top_patterns_count]),
                'prices': [float(p) for p in prices],
                'timestamps': [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in dates],
                'parameters_used': {
                    'min_change': min_change,
                    'window_sizes': window_sizes,
                    'overlap': overlap_percent,
                    'pattern_config': pattern_config
                }
            }

            print(f"Returning {len(response_data['patterns'])} patterns to frontend")
            return jsonify(response_data)

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    except Exception as e:
        print(f"Request processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Request processing failed: {str(e)}'}), 500


def download_and_process_data(date_str, params):
    """Download and process raw data if needed using your existing code"""
    try:
        print(f"Attempting to download raw data for {date_str}")

        # Use your actual paths
        download_result = download_btc_raw_data(date_str, RAW_DATA_DIR)

        if not download_result['success']:
            print(f"Download failed: {download_result}")
            return {'success': False, 'error': 'Download failed'}

        print("Download successful, processing data...")

        # Process raw data using your existing code
        result_files = process_bitcoin_data(
            input_dir=RAW_DATA_DIR,
            output_dir=DATA_DIR,
            temp_dir=TEMP_DIR,
            interval_minutes=1,
            cleanup=True
        )

        print(f"Processing result: {result_files}")

        if date_str in result_files:
            return {'success': True, 'file': result_files[date_str]}
        else:
            return {'success': False, 'error': 'Processing failed - no output file generated'}

    except Exception as e:
        print(f"Download/process error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def filter_data_by_time(prices, dates, df, time_str, time_window_minutes):
    """Filter data by specific time window"""
    from datetime import datetime, time

    try:
        # Parse target time (handle both HH:MM and HH:MM:SS formats)
        if len(time_str.split(':')) == 2:
            target_time = datetime.strptime(time_str, '%H:%M').time()
        else:
            target_time = datetime.strptime(time_str, '%H:%M:%S').time()

        # Convert timestamps to time objects if needed
        if isinstance(dates[0], str):
            # Handle different string formats
            time_objects = []
            for d in dates:
                try:
                    if ':' in d and len(d.split(':')) >= 2:
                        time_objects.append(datetime.strptime(d, '%H:%M:%S').time())
                    else:
                        # Fallback parsing
                        time_objects.append(datetime.strptime(d, '%H:%M').time())
                except:
                    # If parsing fails, skip this entry
                    continue
        else:
            time_objects = [d.time() if hasattr(d, 'time') else d for d in dates]

        # Find indices within time window
        target_minutes = target_time.hour * 60 + target_time.minute
        valid_indices = []

        for i, time_obj in enumerate(time_objects):
            if hasattr(time_obj, 'hour'):
                time_minutes = time_obj.hour * 60 + time_obj.minute
                diff_minutes = min(abs(time_minutes - target_minutes),
                                   1440 - abs(time_minutes - target_minutes))  # Handle midnight wrap

                if diff_minutes <= time_window_minutes:
                    valid_indices.append(i)

        if valid_indices:
            filtered_prices = [prices[i] for i in valid_indices]
            filtered_dates = [dates[i] for i in valid_indices]
            return filtered_prices, filtered_dates, df.iloc[valid_indices] if len(df) > max(valid_indices) else df
        else:
            print("No data points found in specified time window, returning all data")
            return prices, dates, df

    except Exception as e:
        print(f"Time filtering error: {str(e)}, returning all data")
        return prices, dates, df


def format_patterns_for_frontend(all_patterns):
    """Format pattern data for frontend consumption"""
    formatted_patterns = []

    for pattern_data in all_patterns:
        result, analysis, window_info = pattern_data

        # Handle both single pattern and list of patterns
        patterns_to_process = [result] if not isinstance(result, list) else result

        for pattern in patterns_to_process:
            # Extract window start index for proper coordinate conversion
            window_start_idx = window_info.get('start_idx', 0)

            formatted_pattern = {
                'direction': pattern.get('direction', 'unknown'),
                'status': pattern.get('status', 'unknown'),
                'A': pattern.get('A', [0, 0]),
                'B': pattern.get('B', [0, 0]),
                'C': pattern.get('C', [0, 0]),
                'D': pattern.get('D', [0, 0]),
                'initial_move_pct': pattern.get('initial_move_pct', 0),
                'retracement_pct': pattern.get('retracement_pct', 0),
                'fifty_pct_level': pattern.get('fifty_pct_level'),
                'failure_level': pattern.get('failure_level'),
                'completion_level': pattern.get('completion_level'),
                'target_level': pattern.get('target_level'),
                'long_term': pattern.get('long_term', False),
                'pattern_type': pattern.get('pattern_type', 'Fibonacci'),
                'window_info': {
                    'window_size': window_info.get('window_size', 0),
                    'start_date': window_info.get('start_date').isoformat() if window_info.get(
                        'start_date') and hasattr(window_info.get('start_date'), 'isoformat') else str(
                        window_info.get('start_date', '')),
                    'end_date': window_info.get('end_date').isoformat() if window_info.get('end_date') and hasattr(
                        window_info.get('end_date'), 'isoformat') else str(window_info.get('end_date', '')),
                    'start_idx': window_start_idx
                }
            }
            formatted_patterns.append(formatted_pattern)

    return formatted_patterns


@app.route('/api/available-dates', methods=['GET'])
def get_available_dates():
    """Get list of available dates for analysis"""
    try:
        available_dates = []

        # Check processed data directory
        if os.path.exists(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                if filename.startswith('btc_minute_data_') and filename.endswith('.csv'):
                    date_str = filename.replace('btc_minute_data_', '').replace('.csv', '')
                    try:
                        # Validate date format
                        datetime.strptime(date_str, '%Y-%m-%d')
                        available_dates.append(date_str)
                    except ValueError:
                        continue

        # Also check current directory
        for filename in os.listdir('.'):
            if filename.startswith('btc_minute_data_') and filename.endswith('.csv'):
                date_str = filename.replace('btc_minute_data_', '').replace('.csv', '')
                try:
                    # Validate date format
                    datetime.strptime(date_str, '%Y-%m-%d')
                    if date_str not in available_dates:
                        available_dates.append(date_str)
                except ValueError:
                    continue

        available_dates.sort(reverse=True)  # Most recent first
        print(f"Available dates: {available_dates}")
        return jsonify({'dates': available_dates})

    except Exception as e:
        print(f"Error getting available dates: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    """Get current analysis parameters"""
    params = load_parameters()
    return jsonify(params)


if __name__ == '__main__':
    # Make sure required directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Load and display parameters
    params = load_parameters()

    print(f"Starting Bitcoin Analyzer Web Server...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Parameters loaded: {params}")
    print(f"Window sizes: {params.get('window_sizes', 'Not found')}")
    print(f"Min change threshold: {params.get('min_change', 'Not found')}")
    print(f"Open your browser to: http://localhost:5000")

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)