import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def detect_trend(prices):
    return "up" if prices[-1] > prices[0] else "down"

def find_extreme_point(prices):
    direction = detect_trend(prices)
    if direction == "up":
        return np.argmin(prices)  # Lowest point for an uptrend
    else:
        return np.argmax(prices)  # Highest point for a downtrend

def calculate_fibonacci_levels(high, low):
    levels = [0.0, 0.382, 0.5, 0.618, 0.764, 1.0, -0.236, -0.618]
    return {level: high - (high - low) * level for level in levels}

def find_fibonacci_points(prices, levels):
    A = find_extreme_point(prices)
    direction = detect_trend(prices)
    if direction == "up":
        B = np.argmax(prices[A:]) + A  # Highest after A
        C = next((i for i in range(B, len(prices)) if prices[i] <= levels[0.618]), None)
        D = np.argmax(prices[C:]) + C if C else None  # New high after C
    else:
        B = np.argmin(prices[A:]) + A  # Lowest after A
        C = next((i for i in range(B, len(prices)) if prices[i] >= levels[0.618]), None)
        D = np.argmin(prices[C:]) + C if C else None  # New low after C
    return {"A": A, "B": B, "C": C, "D": D}

def analyze_pattern(points, prices):
    if None in points.values():
        return "Pattern is incomplete or failed."
    A, B, C, D = points['A'], points['B'], points['C'], points['D']
    AB = prices[B] - prices[A] if prices[B] > prices[A] else prices[A] - prices[B]
    BC = prices[C] - prices[B] if prices[C] > prices[B] else prices[B] - prices[C]
    CD = prices[D] - prices[C] if prices[D] > prices[C] else prices[C] - prices[D]
    retracement_ratio = BC / AB if AB != 0 else 0
    extension_ratio = CD / AB if AB != 0 else 0
    if 0.618 <= retracement_ratio <= 0.764 and extension_ratio > 1.0:
        return "Successful Fibonacci pattern detected."
    else:
        return "Fibonacci pattern failed."

def plot_fibonacci(prices, levels, points, conclusion):
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Price Data', color='black')
    for level, price in levels.items():
        plt.axhline(price, linestyle='dashed', color='gray', label=f'Fib {level*100:.1f}%')
        plt.text(0, price, f'{level*100:.1f}%', fontsize=10, verticalalignment='bottom', color='blue')
    colors = {'A': 'red', 'B': 'green', 'C': 'blue', 'D': 'purple'}
    for point, idx in points.items():
        if idx is not None:
            plt.scatter(idx, prices[idx], color=colors[point], label=point, zorder=3)
            plt.text(idx, prices[idx], point, fontsize=12, fontweight='bold', verticalalignment='bottom', color=colors[point])
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    plt.title(f"Fibonacci Retracement Levels with Key Points - {conclusion}")
    plt.grid()
    plt.show()

def main():
    df = pd.read_csv("BTCUSDT-hourly-historical-price.csv")
    Prices = df["close"].dropna().iloc[0:100]
    prices = Prices.tolist()
    extreme_index = find_extreme_point(prices)
    high, low = max(prices), min(prices)
    fib_levels = calculate_fibonacci_levels(high, low)
    points = find_fibonacci_points(prices, fib_levels)
    conclusion = analyze_pattern(points, prices)
    plot_fibonacci(prices, fib_levels, points, conclusion)
    print(conclusion)

if __name__ == "__main__":
    main()
