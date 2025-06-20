def extract_completed_patterns(all_patterns):
    completed_patterns = []

    for pattern_data in all_patterns:
        result, analysis, window_info = pattern_data

        # Handle both single pattern and list of patterns
        patterns_to_check = [result] if not isinstance(result, list) else result

        for pattern in patterns_to_check:
            if pattern.get('status') == 'completed':
                completed_patterns.append({
                    'pattern': pattern,
                    'analysis': analysis,
                    'window_info': window_info
                })

    return completed_patterns


def calculate_fan_extensions_from_382_anchor(fan_analysis):

    anchor_382_price = fan_analysis['anchor_100_percent']
    highest_price = fan_analysis['initial_0_percent']

    fan_range = highest_price - anchor_382_price

    fan_levels = [0.236, 0.382, 0.5, 0.618, 0.764]
    fan_extension_prices = {}

    for level in fan_levels:
        extension_price = anchor_382_price - (fan_range * level)
        fan_extension_prices[f"fan_{level}"] = extension_price

    return fan_extension_prices, fan_range