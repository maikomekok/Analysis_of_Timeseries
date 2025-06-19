from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
import sys

sys.path.append('.')
from analyze import (load_and_prepare_data, analyze_multiple_windows,
                     detect_multiple_timeframe_trends, find_significant_price_patterns)

app = Flask(__name__)
CORS(app)

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
            return load_default_params()
    except Exception as e:
        print(f"Error loading parameters: {e}")
        return load_default_params()


def load_default_params():
    """Load default parameters"""
    return {
        "min_change": 0.005,
        "window_sizes": [100, 300, 600, 1000],
        "overlap": 10,
        "top_patterns": 10,
        "time_window": 120,
        "pattern_detection": {
            "retracement_target": 0.5,
            "retracement_tolerance": 0.02,
            "completion_extension": 0.236,
            "failure_level": 0.764,
            "min_move_multiplier": 2.0,
            "search_beyond_window_for_failure": True
        }
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
        """


@app.route('/api/analyze', methods=['POST'])
def analyze_bitcoin_data():
    """Main API endpoint for Bitcoin analysis"""
    try:
        # Load parameters from parameters.json
        params = load_parameters()

        # Get request data
        data = request.json
        date_str = data.get('date')
        time_str = data.get('time')
        time_window = data.get('timeWindow', params.get('time_window', 120))

        # Allow frontend to override min_change
        min_change = data.get('minChange', params.get('min_change', 0.005))

        if not date_str:
            return jsonify({'error': 'Date is required'}), 400

        print(f"=== BITCOIN ANALYSIS REQUEST ===")
        print(f"Date: {date_str}")
        print(f"Time: {time_str}")
        print(f"Min Change: {min_change}")
        print(f"Window sizes: {params.get('window_sizes')}")
        print("=" * 40)

        # Check if processed data exists
        possible_files = [
            os.path.join(DATA_DIR, f"btc_minute_data_{date_str}.csv"),
            f"btc_minute_data_{date_str}.csv",
            os.path.join(".", f"btc_minute_data_{date_str}.csv")
        ]

        data_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                data_file = file_path
                print(f"Found data file: {data_file}")
                break

        if not data_file:
            print(f"No processed data found for {date_str}")
            return jsonify({'error': f'No data file found for {date_str}'}), 404

        try:
            print(f"Loading data from: {data_file}")
            prices, dates, df = load_and_prepare_data(data_file)
            print(f"Loaded {len(prices)} data points")

            # Filter by time if specified
            if time_str:
                print(f"Filtering data around time {time_str}")
                prices, dates, df = filter_data_by_time(prices, dates, df, time_str, time_window)
                print(f"Filtered to {len(prices)} data points")

            if len(prices) < 10:
                return jsonify({'error': 'Not enough data points for analysis'}), 400

            # Extract parameters exactly like main.py does
            window_sizes = params.get('window_sizes', [100, 300, 600, 1000])
            overlap_percent = params.get('overlap', 10)
            pattern_config = params.get('pattern_detection', {})
            top_patterns_count = params.get('top_patterns', 10)

            print(f"Running analysis exactly like main.py:")
            print(f"  Window sizes: {window_sizes}")
            print(f"  Overlap: {overlap_percent}%")
            print(f"  Min change: {min_change}")
            print(f"  Pattern config: {pattern_config}")

            all_patterns = analyze_multiple_windows(
                prices,
                dates,
                window_sizes=window_sizes,
                overlap_percent=overlap_percent,
                min_change_threshold=min_change,
                allow_multiple_patterns=True,
                detect_long_failures=True,
                pattern_config=pattern_config
            )

            print(f"Found {len(all_patterns)} patterns using analyze_multiple_windows")

            # Detect trends
            trends = detect_multiple_timeframe_trends(prices)
            print(f"Detected trends: {trends}")

            # Format response
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
                    'pattern_config': pattern_config
                },
                'analysis_summary': {
                    'total_patterns_found': len(all_patterns),
                    'patterns_returned': len(all_patterns[:top_patterns_count])
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


def filter_data_by_time(prices, dates, df, time_str, time_window_minutes):
    """Filter data by specific time window"""
    from datetime import datetime, time

    try:
        # Parse target time
        if len(time_str.split(':')) == 2:
            target_time = datetime.strptime(time_str, '%H:%M').time()
        else:
            target_time = datetime.strptime(time_str, '%H:%M:%S').time()

        # Convert timestamps to time objects
        if isinstance(dates[0], str):
            time_objects = []
            for d in dates:
                try:
                    if ':' in d and len(d.split(':')) >= 2:
                        time_objects.append(datetime.strptime(d, '%H:%M:%S').time())
                    else:
                        time_objects.append(datetime.strptime(d, '%H:%M').time())
                except:
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
                                   1440 - abs(time_minutes - target_minutes))

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
    """Format pattern data for frontend consumption - simple coordinate handling"""
    formatted_patterns = []
    seen_patterns = set()

    print(f"Formatting {len(all_patterns)} patterns for frontend...")

    for pattern_data in all_patterns:
        result, analysis, window_info = pattern_data

        # Handle both single pattern and list of patterns
        patterns_to_process = [result] if not isinstance(result, list) else result

        for pattern in patterns_to_process:
            # Create signature to avoid duplicates
            pattern_signature = (
                pattern.get('A', [0, 0])[0],
                pattern.get('B', [0, 0])[0],
                pattern.get('C', [0, 0])[0],
                pattern.get('D', [0, 0])[0],
                pattern.get('direction')
            )

            if pattern_signature in seen_patterns:
                continue
            seen_patterns.add(pattern_signature)

            # Handle coordinate conversion for window-based patterns
            window_start_idx = window_info.get('start_idx', 0)

            def convert_point_coordinate(point, window_start):
                """Convert window-local coordinates to global coordinates"""
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    local_index, price = point[0], point[1]

                    # If this is a window-based pattern (start_idx > 0), convert coordinates
                    if window_start > 0:
                        global_index = window_start + local_index
                        print(
                            f"  Converting: local {local_index} + window_start {window_start} = global {global_index}")
                    else:
                        # Comprehensive pattern - coordinates are already global
                        global_index = local_index

                    return [global_index, price]
                return point

            # Convert all pattern coordinates
            converted_A = convert_point_coordinate(pattern.get('A', [0, 0]), window_start_idx)
            converted_B = convert_point_coordinate(pattern.get('B', [0, 0]), window_start_idx)
            converted_C = convert_point_coordinate(pattern.get('C', [0, 0]), window_start_idx)
            converted_D = convert_point_coordinate(pattern.get('D', [0, 0]), window_start_idx)

            formatted_pattern = {
                'direction': pattern.get('direction', 'unknown'),
                'status': pattern.get('status', 'unknown'),
                'A': converted_A,
                'B': converted_B,
                'C': converted_C,
                'D': converted_D,
                'initial_move_pct': pattern.get('initial_move_pct', 0),
                'retracement_pct': pattern.get('retracement_pct', 0),
                'failure_level': pattern.get('failure_level'),
                'completion_level': pattern.get('completion_level'),
                'target_level': pattern.get('target_level'),
                'pattern_type': pattern.get('pattern_type', 'Fibonacci'),
                'analysis': analysis,
                'accurate_failure_point': pattern.get('accurate_failure_point', False),
                'window_info': {
                    'window_size': window_info.get('window_size', 0),
                    'start_idx': window_start_idx
                }
            }

            print(f"Formatted pattern: {pattern['direction']} {pattern['status']}")
            print(f"  A: {converted_A}, B: {converted_B}, C: {converted_C}, D: {converted_D}")
            if pattern.get('accurate_failure_point'):
                print(f"  ✓ Accurate failure point found")

            formatted_patterns.append(formatted_pattern)

    # Sort by quality and limit
    formatted_patterns.sort(key=lambda p: p.get('initial_move_pct', 0), reverse=True)
    final_patterns = formatted_patterns[:10]  # Return top 10 like main.py

    print(f"Returning {len(final_patterns)} formatted patterns")
    return final_patterns


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
                        datetime.strptime(date_str, '%Y-%m-%d')
                        available_dates.append(date_str)
                    except ValueError:
                        continue

        # Also check current directory
        for filename in os.listdir('.'):
            if filename.startswith('btc_minute_data_') and filename.endswith('.csv'):
                date_str = filename.replace('btc_minute_data_', '').replace('.csv', '')
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                    if date_str not in available_dates:
                        available_dates.append(date_str)
                except ValueError:
                    continue

        available_dates.sort(reverse=True)
        return jsonify({'dates': available_dates})

    except Exception as e:
        print(f"Error getting available dates: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    """Get current analysis parameters"""
    params = load_parameters()
    return jsonify(params)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    params = load_parameters()
    return jsonify({
        'status': 'healthy',
        'data_dir_exists': os.path.exists(DATA_DIR),
        'parameters_loaded': bool(params),
        'min_change': params.get('min_change'),
        'window_sizes': params.get('window_sizes'),
        'only_completed_failed': True
    })


if __name__ == '__main__':
    # Make sure required directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Load and display parameters
    params = load_parameters()

    print(f"=== BITCOIN ANALYZER WEB SERVER ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Parameters loaded:")
    print(f"  Min change: {params.get('min_change')}")
    print(f"  Window sizes: {params.get('window_sizes')}")
    print(f"  Top patterns: {params.get('top_patterns')}")
    print(f"  Only completed/failed patterns: YES")
    print(f"Open your browser to: http://localhost:5000")
    print("=" * 40)

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)