import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def detect_trend(prices):
    """Automatically detects if the trend is upward or downward."""
    if prices[-1] > prices[0]:
        return "up"
    else:
        return "down"


def find_best_starting_point(prices, window=50):
    """
    Find the best starting point for Fibonacci retracement analysis.
    Ensures A is the lowest point for downtrend or highest point for uptrend.

    Args:
    prices (list): Price series
    window (int): Size of the rolling window to search for local extrema

    Returns:
    int: Index of the best starting point
    """
    direction = detect_trend(prices)

    # Use numpy for efficient rolling window calculation
    prices_array = np.array(prices)

    if direction == "up":
        # For uptrend, look for a local minimum
        local_mins = []
        for i in range(window, len(prices) - window):
            local_window = prices_array[i - window:i + window]
            if prices_array[i] == np.min(local_window):
                local_mins.append((i, prices_array[i]))

        # Select the local minimum with the lowest price
        if local_mins:
            return min(local_mins, key=lambda x: x[1])[0]
    else:
        # For downtrend, look for the absolute lowest point
        return np.argmin(prices_array)


def find_retracement_extension(prices, threshold=0.3, nested_threshold=0.2):
    """
    Find Fibonacci retracement points in price data dynamically.
    Improved version with better starting point selection.
    """
    # Find the best starting point
    A = find_best_starting_point(prices)
    direction = detect_trend(prices[A:])

    # Adjust prices to start from the new A point
    prices_subset = prices[A:]

    B, C, D = None, None, None

    # Find B (Swing High/Low)
    if direction == "up":
        B = max(range(1, len(prices_subset) - 1), key=lambda i: prices_subset[i])  # Find highest point
    else:
        B = min(range(1, len(prices_subset) - 1), key=lambda i: prices_subset[i])  # Find lowest point

    if B is None:
        return None

    # Find C based on retracement threshold
    if direction == "up":
        move = prices_subset[B] - prices_subset[0]
        retracement_threshold = prices_subset[B] - (move * threshold)
        for j in range(B + 1, len(prices_subset) - 1):
            if prices_subset[j] <= retracement_threshold:
                C = j
                break
    else:
        move = prices_subset[0] - prices_subset[B]
        retracement_threshold = prices_subset[B] + (move * threshold)
        for j in range(B + 1, len(prices_subset) - 1):
            if prices_subset[j] >= retracement_threshold:
                C = j
                break

    if C is None:
        return None

    # Find D (Final Move Completion)
    if direction == "up":
        D = max(range(C + 1, len(prices_subset)), key=lambda i: prices_subset[i])  # Find new high
    else:
        D = min(range(C + 1, len(prices_subset)), key=lambda i: prices_subset[i])  # Find new low

    # Adjust indices to original price series
    result = {
        "A": (A, prices[A]),
        "B": (A + B, prices[A + B]),
        "C": (A + C, prices[A + C]),
        "D": (A + D, prices[A + D]),
        "direction": direction
    }
    return result


def draw_fibonacci_levels(ax, prices, direction):
    """
    Draw Fibonacci retracement levels on a diagnostic plot
    based on min and max values
    """
    min_idx = np.argmin(prices)
    max_idx = np.argmax(prices)

    # Determine A and B for Fibonacci levels based on trend
    if direction == "up":
        A_value = prices[min_idx]
        B_value = prices[max_idx]
    else:
        A_value = prices[max_idx]
        B_value = prices[min_idx]

    # Calculate the main move
    main_move = B_value - A_value

    # Draw retracement levels
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0, -0.236, -0.618]
    level_colors = {
        0: '#FF0000',  # Red
        0.236: '#FF7F00',  # Orange
        0.382: '#FFFF00',  # Yellow
        0.5: '#00FF00',  # Green
        0.618: '#0000FF',  # Blue
        0.764: '#4B0082',  # Indigo
        1.0: '#8F00FF',  # Violet
        -0.236: '#FFC0CB',  # Pink
        -0.618: '#800080'  # Purple
    }

    for level in fib_levels:
        fib_price = B_value - (main_move * level) if direction == "up" else B_value + (main_move * level)
        ax.axhline(
            y=fib_price,
            color=level_colors.get(level, 'gray'),
            linestyle='--',
            alpha=0.6,
            linewidth=1.5,
            label=f'Fib {level * 100:.1f}%'
        )
        ax.text(
            0,
            fib_price,
            f'{level * 100:.1f}%',
            fontsize=10,
            verticalalignment='center',
            color=level_colors.get(level, 'gray')
        )

    return ax


def plot_diagnostic_graph(prices, result=None):
    """
    Create a diagnostic plot of the price series.
    If a retracement result is provided, mark the key points.
    Always draw Fibonacci levels based on min and max prices.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(prices, marker='o', linestyle='-', color='black', alpha=0.7, label='Price Series')

    # Title and basic info
    direction = result['direction'] if result else detect_trend(prices)
    move_type = "Upward" if direction == "up" else "Downward"
    ax.set_title(f'Price Series Diagnostic Plot - {move_type} Move', fontsize=16)

    # If result exists, mark key points
    if result:
        points = {'A': ('black', result['A']), 'B': ('red', result['B']),
                  'C': ('green', result['C']), 'D': ('blue', result['D'])}

        for label, (color, point) in points.items():
            ax.scatter(point[0], point[1], color=color, s=100, zorder=3)
            ax.text(point[0], point[1], label, fontsize=14, fontweight='bold',
                    ha='right', va='bottom', color=color)

        # Draw Fibonacci retracement levels based on A and B
        A, B = result['A'][1], result['B'][1]
        main_move = B - A if direction == "up" else A - B
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0, -0.236, -0.618]
        level_colors = {
            0: '#FF0000',  # Red
            0.236: '#FF7F00',  # Orange
            0.382: '#FFFF00',  # Yellow
            0.5: '#00FF00',  # Green
            0.618: '#0000FF',  # Blue
            0.764: '#4B0082',  # Indigo
            1.0: '#8F00FF',  # Violet
            -0.236: '#FFC0CB',  # Pink
            -0.618: '#800080'  # Purple
        }

        for level in fib_levels:
            fib_price = B - (main_move * level) if direction == "up" else B + (main_move * level)
            ax.axhline(
                y=fib_price,
                color=level_colors.get(level, 'gray'),
                linestyle='--',
                alpha=0.6,
                linewidth=1.5,
                label=f'Fib {level * 100:.1f}%'
            )
            ax.text(
                0,
                fib_price,
                f'{level * 100:.1f}%',
                fontsize=10,
                verticalalignment='center',
                color=level_colors.get(level, 'gray')
            )
    else:
        # Draw Fibonacci levels based on min and max when no pattern is found
        ax = draw_fibonacci_levels(ax, prices, direction)

    # Statistical annotations
    ax.axhline(y=np.mean(prices), color='purple', linestyle='-', linewidth=2, label='Mean Price')

    # Min and Max points
    min_idx = np.argmin(prices)
    max_idx = np.argmax(prices)
    ax.scatter(min_idx, prices[min_idx], color='darkred', label='Min Price', s=100, zorder=4)
    ax.scatter(max_idx, prices[max_idx], color='darkgreen', label='Max Price', s=100, zorder=4)

    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    plt.tight_layout()


def analyze_move(result):
    """Analyze the retracement and extension ratios."""
    if result is None:
        return "No valid pattern found."

    A, B, C, D = result['A'][1], result['B'][1], result['C'][1], result['D'][1]
    direction = result['direction']

    if direction == "up":
        move_distance = B - A
        retracement_ratio = (B - C) / move_distance if move_distance != 0 else 0
        extension_ratio = (D - C) / move_distance if move_distance != 0 else None
    else:
        move_distance = A - B
        retracement_ratio = (C - B) / move_distance if move_distance != 0 else 0
        extension_ratio = (C - D) / move_distance if move_distance != 0 else None

    # Classify move
    if retracement_ratio < 0.382:
        status = "Progressive move (shallow retracement)"
    elif retracement_ratio >= 0.618:
        status = "Failed move (deep retracement)"
    else:
        status = "Successful move (moderate retracement)"

    if extension_ratio is not None:
        if extension_ratio > 1.0:
            status += " with strong extension"
        elif extension_ratio < 0.5:
            status += " with weak extension"

    return status


# Example usage
def main():
    # Load BTC data
    df = pd.read_csv("BTCUSDT-hourly-historical-price.csv")
    Prices = df["close"].dropna().iloc[0:100]
    prices = Prices.tolist()

    # Run Analysis on BTC Data
    result = find_retracement_extension(prices)

    # Always plot the diagnostic graph
    plot_diagnostic_graph(prices, result)

    if result:
        analysis = analyze_move(result)

        print(f"Detected trend: {result['direction']}")
        print(f"Identified points:")
        for point, point_info in result.items():
            if point != 'direction':
                print(f"{point}: Index {point_info[0]}, Price {point_info[1]}")
        print(f"Analysis: {analysis}")

        # Print key Fibonacci ratios
        A, B = result['A'][1], result['B'][1]
        main_move = B - A if result['direction'] == "up" else A - B
        print("\nKey Fibonacci levels:")
        for level in [0, 0.236, 0.382, 0.5, 0.618, 0.764, 1.0, -0.236, -0.618]:
            fib_price = B - (main_move * level) if result['direction'] == "up" else B + (main_move * level)
            print(f"{level * 100:.1f}%: {fib_price:.2f}")
    else:
        print("Could not find a valid Fibonacci retracement pattern.")
        print("A diagnostic graph has been generated with Fibonacci levels based on min/max prices.")

    plt.show()


if __name__ == "__main__":
    main()