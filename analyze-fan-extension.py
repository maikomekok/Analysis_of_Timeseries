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