import matplotlib.pyplot as plt
import pandas as pd

# Load BTC data
df = pd.read_csv("BTCUSDT-hourly-historical-price.csv")
Prices = df["close"].dropna().iloc[200:800]
prices = Prices.tolist()



def detect_trend(prices):
    """Automatically detects if the trend is upward or downward."""
    if prices[-1] > prices[0]:
        return "up"
    else:
        return "down"

def find_retracement_extension(prices, threshold=0.3, nested_threshold=0.2):
    """
    Find Fibonacci retracement points in price data dynamically.
    Now automatically detects trend direction.
    """
    direction = detect_trend(prices)
    A = 0
    B, C, D = None, None, None

    # Find B (Swing High/Low)
    if direction == "up":
        B = max(range(1, len(prices) - 1), key=lambda i: prices[i])  # Find highest point
    else:
        B = min(range(1, len(prices) - 1), key=lambda i: prices[i])  # Find lowest point

    if B is None:
        return None

    # Find C based on retracement threshold
    if direction == "up":
        move = prices[B] - prices[A]
        retracement_threshold = prices[B] - (move * threshold)
        for j in range(B + 1, len(prices) - 1):
            if prices[j] <= retracement_threshold:
                C = j
                break
    else:
        move = prices[A] - prices[B]
        retracement_threshold = prices[B] + (move * threshold)
        for j in range(B + 1, len(prices) - 1):
            if prices[j] >= retracement_threshold:
                C = j
                break

    if C is None:
        return None

    # Find D (Final Move Completion)
    if direction == "up":
        D = max(range(C + 1, len(prices)), key=lambda i: prices[i])  # Find new high
    else:
        D = min(range(C + 1, len(prices)), key=lambda i: prices[i])  # Find new low

    # Ensure -23.6% level is crossed
    move_extension = (prices[D] - prices[C]) / (prices[B] - prices[A])
    if move_extension < -0.236:
        print(f"Move complete: crossed -23.6% level at index {D}, price: {prices[D]}")
    else:
        print("Move not completed yet.")

    return {"A": (A, prices[A]), "B": (B, prices[B]), "C": (C, prices[C]), "D": (D, prices[D]), "direction": direction}

def plot_fib_retracement(prices, result):
    if not result:
        print("No valid retracement found.")
        return

    direction = result.get("direction", "up")
    plt.figure(figsize=(14, 8))
    plt.plot(prices, marker='o', linestyle='-', color='black', alpha=0.7, label='Price Series')
    move_type = "Upward" if direction == "up" else "Downward"
    plt.title(f'Fibonacci Retracement Analysis - {move_type} Move', fontsize=16)

    # Mark key points
    points = {'A': ('black', result['A']), 'B': ('red', result['B']), 'C': ('green', result['C']), 'D': ('blue', result['D'])}
    for label, (color, (idx, price)) in points.items():
        plt.scatter(idx, price, color=color, s=100, zorder=3)
        plt.text(idx, price, label, fontsize=14, fontweight='bold', ha='right', va='bottom', color=color)

    # Plot Fibonacci retracement levels
    A, B, C = result['A'][1], result['B'][1], result['C'][1]
    main_move = B - A if direction == "up" else A - B
    fib_levels = [0, 0.382, 0.5, 0.618, 0.764, 1.0, -0.236, -0.618]
    for level in fib_levels:
        fib_price = B - (main_move * level) if direction == "up" else B + (main_move * level)
        plt.axhline(y=fib_price, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        plt.text(0, fib_price, f'{level * 100:.1f}%', fontsize=8, verticalalignment='center', color='gray')

    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
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

# Run Analysis on BTC Data
result = find_retracement_extension(prices)
plot_fib_retracement(prices, result)
analysis = analyze_move(result)

print(f"Detected trend: {result['direction']}")
print(f"Identified points: {result}")
print(f"Analysis: {analysis}")

plt.show()
