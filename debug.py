#!/usr/bin/env python3
"""
Coordinate Debug Script - Helps debug pattern coordinate mapping issues
"""

import os
import json
from datetime import datetime
from analyze import load_and_prepare_data, find_significant_price_patterns, analyze_multiple_windows


def debug_coordinates(date_str, data_file=None):
    """Debug coordinate mapping for pattern detection"""

    print(f"🔍 DEBUGGING COORDINATES FOR {date_str}")
    print("=" * 60)

    # Find data file
    if not data_file:
        possible_files = [
            f"C:/Users/admin/Desktop/btc_minute_data/btc_minute_data_{date_str}.csv",
            f"btc_minute_data_{date_str}.csv",
            f"./btc_minute_data_{date_str}.csv"
        ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                data_file = file_path
                break

        if not data_file:
            print(f"❌ No data file found for {date_str}")
            return

    print(f"📁 Using data file: {data_file}")

    # Load parameters
    try:
        with open('parameters.json', 'r') as f:
            params = json.load(f)
        pattern_config = params.get('pattern_detection', {})
        min_change = params.get('min_change', 0.005)
        window_sizes = params.get('window_sizes', [600])
    except:
        print("⚠️ Using default parameters")
        pattern_config = {
            "retracement_target": 0.5,
            "retracement_tolerance": 0.015,
            "completion_extension": 0.236,
            "failure_level": 0.764,
            "min_move_multiplier": 3.0
        }
        min_change = 0.008
        window_sizes = [600]

    # Load data
    try:
        prices, dates, df = load_and_prepare_data(data_file)
        print(f"✅ Loaded {len(prices)} data points")
        print(f"📈 Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        print(f"📅 Date range: {dates[0]} to {dates[-1]}")
        print(f"🔢 Sample indices: 0, {len(prices) // 4}, {len(prices) // 2}, {3 * len(prices) // 4}, {len(prices) - 1}")
        print(
            f"🕐 Sample times: {dates[0]}, {dates[len(dates) // 4]}, {dates[len(dates) // 2]}, {dates[3 * len(dates) // 4]}, {dates[-1]}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    print(f"\n🔍 TESTING DIFFERENT DETECTION METHODS")
    print("-" * 60)

    # Method 1: Comprehensive detection (full dataset)
    print(f"METHOD 1: Comprehensive Detection (Full Dataset)")
    print("-" * 40)

    try:
        comprehensive_patterns = find_significant_price_patterns(
            prices, dates, min_change, config=pattern_config
        )

        print(f"Found {len(comprehensive_patterns)} comprehensive patterns")

        for i, pattern in enumerate(comprehensive_patterns):
            print(f"\nComprehensive Pattern {i + 1}:")
            print(f"  Direction: {pattern['direction']}, Status: {pattern['status']}")

            for point_name in ['A', 'B', 'C', 'D']:
                if point_name in pattern:
                    idx, price = pattern[point_name]
                    # Validate index
                    if 0 <= idx < len(dates):
                        time_str = dates[idx].strftime('%H:%M:%S') if hasattr(dates[idx], 'strftime') else str(
                            dates[idx])
                        print(f"  {point_name}: Index {idx:4d}, Price ${price:8.2f}, Time {time_str}")
                    else:
                        print(f"  {point_name}: Index {idx:4d} ❌ OUT OF RANGE, Price ${price:8.2f}")

    except Exception as e:
        print(f"❌ Comprehensive detection failed: {e}")
        comprehensive_patterns = []

    # Method 2: Window-based detection
    print(f"\nMETHOD 2: Window-Based Detection")
    print("-" * 40)

    try:
        all_patterns = analyze_multiple_windows(
            prices,
            dates,
            window_sizes=window_sizes,
            overlap_percent=70,
            min_change_threshold=min_change,
            allow_multiple_patterns=False,
            detect_long_failures=True,
            pattern_config=pattern_config
        )

        print(f"Found {len(all_patterns)} window-based patterns")

        for i, (pattern, analysis, window_info) in enumerate(all_patterns):
            print(f"\nWindow Pattern {i + 1}:")
            print(f"  Direction: {pattern['direction']}, Status: {pattern['status']}")
            print(f"  Window: start_idx={window_info['start_idx']}, size={window_info['window_size']}")

            for point_name in ['A', 'B', 'C', 'D']:
                if point_name in pattern:
                    local_idx, price = pattern[point_name]

                    # Calculate global index
                    window_start = window_info['start_idx']
                    global_idx = window_start + local_idx

                    # Validate both local and global indices
                    local_valid = 0 <= local_idx < window_info['window_size']
                    global_valid = 0 <= global_idx < len(dates)

                    if global_valid:
                        time_str = dates[global_idx].strftime('%H:%M:%S') if hasattr(dates[global_idx],
                                                                                     'strftime') else str(
                            dates[global_idx])
                        status = "✅" if local_valid and global_valid else "❌"
                        print(
                            f"  {point_name}: Local {local_idx:3d} → Global {global_idx:4d} {status}, Price ${price:8.2f}, Time {time_str}")
                    else:
                        print(
                            f"  {point_name}: Local {local_idx:3d} → Global {global_idx:4d} ❌ OUT OF RANGE, Price ${price:8.2f}")

    except Exception as e:
        print(f"❌ Window-based detection failed: {e}")
        all_patterns = []

    # Summary and recommendations
    print(f"\n💡 COORDINATE ANALYSIS SUMMARY")
    print("-" * 60)

    total_comprehensive = len(comprehensive_patterns)
    total_windowed = len(all_patterns)

    print(f"Comprehensive patterns: {total_comprehensive}")
    print(f"Window-based patterns: {total_windowed}")

    if total_comprehensive > 0:
        print("✅ Comprehensive detection working - use this method")
        print("   Coordinates are already global, no conversion needed")
    elif total_windowed > 0:
        print("⚠️ Only window-based detection working")
        print("   ⚡ CRITICAL: Need proper coordinate conversion!")
        print("   Local window indices must be converted to global indices")
        print("   Formula: global_index = window_start_index + local_index")
    else:
        print("❌ No patterns detected with current settings")
        print("   Try reducing min_change or adjusting thresholds")

    print(f"\n🎯 COORDINATE MAPPING VALIDATION")
    print("-" * 60)

    # Test coordinate mapping for window patterns
    if all_patterns:
        for i, (pattern, analysis, window_info) in enumerate(all_patterns[:2]):  # Test first 2
            print(f"\nTesting Pattern {i + 1} coordinate mapping:")
            window_start = window_info['start_idx']
            window_size = window_info['window_size']

            print(f"  Window: [{window_start}:{window_start + window_size}]")

            for point_name in ['A', 'B', 'C', 'D']:
                if point_name in pattern:
                    local_idx, price = pattern[point_name]
                    global_idx = window_start + local_idx

                    # Check if this makes sense
                    if 0 <= global_idx < len(prices):
                        actual_price = prices[global_idx]
                        price_match = abs(actual_price - price) < 0.01
                        match_status = "✅ MATCH" if price_match else f"❌ MISMATCH (actual: ${actual_price:.2f})"

                        print(
                            f"    {point_name}: {local_idx} + {window_start} = {global_idx} → ${price:.2f} {match_status}")
                    else:
                        print(f"    {point_name}: {local_idx} + {window_start} = {global_idx} ❌ OUT OF RANGE")


def test_frontend_coordinate_format():
    """Test the coordinate format that should be sent to frontend"""
    print(f"\n🌐 FRONTEND COORDINATE FORMAT TEST")
    print("-" * 60)

    # Simulate what the frontend expects
    sample_pattern = {
        'direction': 'up',
        'status': 'completed',
        'A': [100, 85000.50],  # [index, price]
        'B': [200, 86500.75],
        'C': [300, 85750.25],
        'D': [400, 87000.00]
    }

    print("Expected format for frontend:")
    print("  Pattern points should be: [global_index, price]")
    print("  Where global_index corresponds to position in the full price/timestamp arrays")
    print()
    print("Example:")
    for point_name, (index, price) in sample_pattern.items():
        if point_name in ['A', 'B', 'C', 'D']:
            print(f"  {point_name}: [{index}, {price}] ← index {index} in full dataset")

    print("\n⚠️ Common mistakes:")
    print("  ❌ Using window-local indices without conversion")
    print("  ❌ Index >= timestamp array length")
    print("  ❌ Negative indices")
    print("  ❌ Price doesn't match price array at that index")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python debug_coordinates.py YYYY-MM-DD [data_file.csv]")
        print("Example: python debug_coordinates.py 2025-04-15")
        return

    date_str = sys.argv[1]
    data_file = sys.argv[2] if len(sys.argv) > 2 else None

    debug_coordinates(date_str, data_file)
    test_frontend_coordinate_format()


if __name__ == "__main__":
    main()