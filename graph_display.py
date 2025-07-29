import numpy as np
import matplotlib.pyplot as plt
import random


class BearishFanExtensions:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.original_abcd = None
        self.fan_systems = []
        self.failed_patterns = []
        self.completed_patterns = []

    def generate_bearish_fan_pattern(self, data_points=2000):
        base_price = 50000
        prices = []
        x_values = list(range(data_points))

        # Remove the crazy noise - just use flat base
        for i in range(data_points):
            prices.append(base_price)

        self.original_abcd = self._create_bearish_abcd(prices, x_values, 200, 550)
        fan_system1 = self._create_bearish_fan_extension_from_original(prices, x_values, 550, 1200, "Bear Fan S1")
        fan_system2 = self._create_bearish_fan_with_retrace_support(prices, x_values, 1200, 1500, "Bear Fan S2")
        fan_system3 = self._create_bearish_fan_with_break_remeasure(prices, x_values, 1500, 1900, "Bear Fan S3")
        self.fan_systems = [fan_system1, fan_system2, fan_system3]

        # Evaluate each fan system for failure/completion
        self._evaluate_fan_patterns()

        return {
            'x_values': x_values,
            'prices': prices,
            'original_abcd': self.original_abcd,
            'fan_systems': self.fan_systems,
            'failed_patterns': self.failed_patterns,
            'completed_patterns': self.completed_patterns
        }

    def _evaluate_fan_patterns(self):
        """Evaluate each fan pattern for failure or completion status"""
        self.failed_patterns = []
        self.completed_patterns = []

        for fan_system in self.fan_systems:
            failure_info = self._check_fan_failure(fan_system)
            if failure_info['failed']:
                self.failed_patterns.append({
                    'system': fan_system,
                    'failure_info': failure_info
                })
            else:
                self.completed_patterns.append({
                    'system': fan_system,
                    'completion_info': self._get_completion_info(fan_system)
                })

    def _check_fan_failure(self, fan_system):
        """
        Check if a bearish fan system has failed based on:
        1. 88.6% is the failure point for each fan (upside break in downtrend)
        2. ABCD pattern 38.2% level is the final brake of the fan system (upside break)
        """
        failure_info = {
            'failed': False,
            'failure_type': None,
            'failure_level': None,
            'failure_price': None,
            'failure_point': None
        }

        # Get the base move for calculating failure levels
        if 'AB_move' in fan_system:
            base_move = fan_system['AB_move']
        else:
            base_move = fan_system['original_AB_move']

        # Calculate failure levels for bearish patterns
        if 's2_point' in fan_system:
            # For bearish fan extensions, 88.6% failure level from S2 (upward break)
            s2_price = fan_system['s2_point'][1]
            s1_price = fan_system['s1_point'][1]
            s1_s2_move = s1_price - s2_price  # Note: reversed for bearish
            fan_886_failure = s2_price + s1_s2_move * 0.886  # Upward break in downtrend

            # Check if price broke above 88.6% level (failure in bearish)
            if 'failure_point' in fan_system:
                failure_price = fan_system['failure_point'][1]
                if failure_price > fan_886_failure:
                    failure_info['failed'] = True
                    failure_info['failure_type'] = 'Bearish Fan 88.6% Break'
                    failure_info['failure_level'] = '88.6%'
                    failure_info['failure_price'] = failure_price
                    failure_info['failure_point'] = fan_system['failure_point']

        # Check for ABCD 38.2% level break (final brake) - upward break in bearish
        original_abcd_382 = self.original_abcd['points']['B'][1] + self.original_abcd['AB_move'] * 0.382

        if 'final_break_point' in fan_system:
            final_break_price = fan_system['final_break_point'][1]
            if final_break_price > original_abcd_382:
                failure_info['failed'] = True
                failure_info['failure_type'] = 'Bearish ABCD 38.2% Final Break'
                failure_info['failure_level'] = 'ABCD 38.2%'
                failure_info['failure_price'] = final_break_price
                failure_info['failure_point'] = fan_system['final_break_point']

        return failure_info

    def _get_completion_info(self, fan_system):
        """Get completion information for successful bearish fan patterns"""
        completion_info = {
            'completion_level': fan_system.get('completion_level', 'Unknown'),
            'completion_point': fan_system.get('completion_point', None),
            'lowest_extension': None
        }

        if 'new_s3_point' in fan_system:
            completion_info['lowest_extension'] = fan_system['new_s3_point']

        return completion_info

    def _create_bearish_abcd(self, prices, x_values, start_idx, end_idx):
        A_idx = start_idx + 80  # 280
        B_idx = start_idx + 180  # 380
        C_idx = start_idx + 280  # 480
        D_idx = end_idx  # 550

        A_price = 55000  # Higher starting point for bearish
        B_price = 45000  # Lower trough
        AB = A_price - B_price  # 10000 - bearish move down
        C_price = B_price + AB * 0.5  # 45000 + 10000 * 0.5 = 50000
        D_price = C_price - AB  # 50000 - 10000 = 40000

        # D completes at -23.6% level (below B which is 0% in bearish)
        d_completion_price = B_price - AB * 0.236  # 45000 - 10000 * 0.236 = 42640
        prices[D_idx] = d_completion_price

        prices[A_idx] = A_price
        prices[B_idx] = B_price
        prices[C_idx] = C_price

        self._smooth_transition(prices, A_idx, B_idx, A_price, B_price)
        self._smooth_transition(prices, B_idx, C_idx, B_price, C_price)
        self._smooth_transition(prices, C_idx, D_idx, C_price, d_completion_price)

        reversal_length = 50
        reversal_slope = (prices[D_idx] - prices[C_idx]) / reversal_length * 0.5

        for i in range(D_idx + 1, min(D_idx + reversal_length + 1, len(prices))):
            progress = (i - D_idx) / reversal_length
            prices[i] = prices[D_idx] - reversal_slope * progress
            # Remove the random noise here

        for i in range(D_idx + reversal_length + 1, len(prices)):
            prices[i] = prices[i - 1]
            # Remove the random noise here too

        AB_move = A_price - B_price  # Bearish move

        # Fixed Fibonacci levels for bearish - B is now 0%, D is -23.6%
        fib_levels = {
            '100%': A_price,  # A at 100%
            '50%': C_price,  # C at 50%
            '0%': B_price,  # B at 0%
            '-23.6%': d_completion_price,  # D at -23.6%
            '+38.2%': B_price + AB_move * 0.382  # ABCD 38.2% failure level (upward)
        }

        return {
            'name': 'Bearish ABCD',
            'color': 'purple',
            'points': {'A': (A_idx, A_price), 'B': (B_idx, B_price),
                       'C': (C_idx, C_price), 'D': (D_idx, d_completion_price)},
            'AB_move': AB_move,
            'fib_levels': fib_levels,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'failure_levels': {
                'abcd_382': B_price + AB_move * 0.382  # Upward break
            }
        }

    def _create_bearish_fan_extension_from_original(self, prices, x_values, start_idx, end_idx, system_name):
        original_D_price = self.original_abcd['points']['D'][1]
        original_AB_move = self.original_abcd['AB_move']
        D_idx = self.original_abcd['points']['D'][0]
        B_price = self.original_abcd['points']['B'][1]

        # S1 at positive 38.2% level (up from B in bearish)
        s1_price = B_price + original_AB_move * 0.382  # 45000 + 10000 * 0.382 = 48820
        s1_idx = D_idx + 80  # 630
        prices[s1_idx] = s1_price

        # S2: Lowest price after S1 (in bearish trend)
        s2_price = 40000  # Lower for bearish moves
        s2_idx = s1_idx + 80  # 710
        prices[s2_idx] = s2_price

        # S1-S2 move (bearish)
        s1_s2_move = s1_price - s2_price  # Bearish move down

        # Calculate failure levels for bearish
        fan_886_failure = s2_price + s1_s2_move * 0.886  # 88.6% failure level (upward break)
        abcd_382_failure = self.original_abcd['failure_levels']['abcd_382']

        # New Fibonacci levels for bearish S1-S2
        fan_levels = {
            'S1 (100%)': s1_price,
            'S2 (0%)': s2_price,
            '38.2%': s2_price + s1_s2_move * 0.382,  # Upward retracement
            '78.6%': s2_price + s1_s2_move * 0.786,  # Upward retracement
            '88.6%': fan_886_failure,  # Fan failure level (upward)
            '161.8%': s1_price - s1_s2_move * 1.618  # Downward extension
        }

        # S3 at 38.2% retracement of S1-S2 (upward)
        s3_price = fan_levels['38.2%']
        s3_idx = s2_idx + 80  # 790
        prices[s3_idx] = s3_price

        # New S3 at 161.8% of S1-S2 (downward extension)
        new_s3_price = fan_levels['161.8%']
        new_s3_idx = s3_idx + 80  # 870
        prices[new_s3_idx] = new_s3_price

        # First failure: Break above 88.6% level (bullish break in bearish trend)
        failure_price = fan_886_failure + 1000  # Above 88.6% level - failure in bearish
        failure_idx = new_s3_idx + 80  # 950
        prices[failure_idx] = failure_price

        # Recovery attempt (back down in bearish)
        recovery_price = 38000  # Lower recovery in bearish
        recovery_idx = failure_idx + 80  # 1030
        prices[recovery_idx] = recovery_price

        # Final break: ABCD 38.2% level break (upward break)
        final_price = abcd_382_failure + 500  # Break ABCD 38.2% upward (final brake)
        final_idx = recovery_idx + 80  # 1110
        prices[final_idx] = final_price

        final2_price = 60000  # Much higher final price (failed bearish)
        final2_idx = final_idx + 10  # 1120
        prices[final2_idx] = final2_price

        # Smooth transitions
        self._smooth_transition(prices, D_idx, s1_idx, original_D_price, s1_price)
        self._smooth_transition(prices, s1_idx, s2_idx, s1_price, s2_price)
        self._smooth_transition(prices, s2_idx, s3_idx, s2_price, s3_price)
        self._smooth_transition(prices, s3_idx, new_s3_idx, s3_price, new_s3_price)
        self._smooth_transition(prices, new_s3_idx, failure_idx, new_s3_price, failure_price)
        self._smooth_transition(prices, failure_idx, recovery_idx, failure_price, recovery_price)
        self._smooth_transition(prices, recovery_idx, final_idx, recovery_price, final_price)
        self._smooth_transition(prices, final_idx, final2_idx, final_price, final2_price)
        self._smooth_transition(prices, final2_idx, end_idx, final2_price, prices[end_idx])

        return {
            'name': system_name,
            'color': 'green',
            'fan_levels': fan_levels,
            'original_AB_move': original_AB_move,
            'AB_move': s1_s2_move,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'completion_level': '161.8%',
            'completion_point': (new_s3_idx, new_s3_price),
            's1_point': (s1_idx, s1_price),
            's2_point': (s2_idx, s2_price),
            's3_point': (s3_idx, s3_price),
            'new_s3_point': (new_s3_idx, new_s3_price),
            'failure_point': (failure_idx, failure_price),  # This breaks 88.6% upward
            'recovery_point': (recovery_idx, recovery_price),
            'final_break_point': (final_idx, final_price),
            'failure_levels': {
                'fan_886': fan_886_failure,
                'abcd_382': abcd_382_failure
            },
            'is_fan_extension': True
        }

    def _create_bearish_fan_with_retrace_support(self, prices, x_values, start_idx, end_idx, system_name):
        original_AB_move = self.original_abcd['AB_move']
        # Use B price as reference (0% level)
        original_B_price = self.original_abcd['points']['B'][1]

        fan_levels = {
            'B (0%)': original_B_price,
            '-23.6%': original_B_price - original_AB_move * 0.236,
            '-38.2%': original_B_price - original_AB_move * 0.382,
            '-50%': original_B_price - original_AB_move * 0.5,
            '-61.8%': original_B_price - original_AB_move * 0.618,
            '-78.6%': original_B_price - original_AB_move * 0.786,
            '+88.6%': original_B_price + original_AB_move * 0.886,  # Fan failure level (upward)
            '-100%': original_B_price - original_AB_move * 1.0,
            '-161.8%': original_B_price - original_AB_move * 1.618,
            '-261.8%': original_B_price - original_AB_move * 2.618,
            '-423.6%': original_B_price - original_AB_move * 4.236
        }

        phase1_end = start_idx + int((end_idx - start_idx) * 0.4)
        start_price = fan_levels['-61.8%']
        target_price1 = fan_levels['-78.6%'] - 2000  # Lower targets in bearish
        self._smooth_transition(prices, start_idx, phase1_end, start_price, target_price1)

        phase2_end = start_idx + int((end_idx - start_idx) * 0.7)
        support_price = fan_levels['-38.2%'] - 1000  # Lower support in bearish
        self._smooth_transition(prices, phase1_end, phase2_end, target_price1, support_price)

        final_target = fan_levels['-161.8%'] - 3000  # Much lower final target
        self._smooth_transition(prices, phase2_end, end_idx, support_price, final_target)

        return {
            'name': system_name,
            'color': 'blue',
            'fan_levels': fan_levels,
            'original_AB_move': original_AB_move,
            'AB_move': original_AB_move,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'support_retest': (phase2_end, support_price),
            'completion_level': '161.8%',
            'completion_point': (end_idx, final_target),
            'failure_levels': {
                'fan_886': fan_levels['+88.6%'],
                'abcd_382': self.original_abcd['failure_levels']['abcd_382']
            },
            'is_fan_extension': True
        }

    def _create_bearish_fan_with_break_remeasure(self, prices, x_values, start_idx, end_idx, system_name):
        original_AB_move = self.original_abcd['AB_move']
        # Use B price as reference (0% level)
        original_B_price = self.original_abcd['points']['B'][1]
        original_fan_382 = original_B_price - original_AB_move * 0.382

        break_idx = start_idx + int((end_idx - start_idx) * 0.2)
        start_price = original_B_price - original_AB_move * 1.618
        break_price = original_fan_382 * 1.015  # Small upward break
        self._smooth_transition(prices, start_idx, break_idx, start_price, break_price)

        support_idx = start_idx + int((end_idx - start_idx) * 0.4)
        new_support_price = break_price * 1.002
        self._smooth_transition(prices, break_idx, support_idx, break_price, new_support_price)

        new_fan_levels = {
            'New Support': new_support_price,
            '-23.6%': new_support_price - original_AB_move * 0.236,
            '-38.2%': new_support_price - original_AB_move * 0.382,
            '-50%': new_support_price - original_AB_move * 0.5,
            '-61.8%': new_support_price - original_AB_move * 0.618,
            '-78.6%': new_support_price - original_AB_move * 0.786,
            '+88.6%': new_support_price + original_AB_move * 0.886,  # Fan failure level (upward)
            '-100%': new_support_price - original_AB_move * 1.0,
            '-161.8%': new_support_price - original_AB_move * 1.618,
            '-261.8%': new_support_price - original_AB_move * 2.618,
            '-423.6%': new_support_price - original_AB_move * 4.236
        }

        resistance_idx = start_idx + int((end_idx - start_idx) * 0.7)
        resistance_price = new_fan_levels['-61.8%'] - 1500  # Lower resistance in bearish
        self._smooth_transition(prices, support_idx, resistance_idx, new_support_price, resistance_price)

        next_pattern_target = new_fan_levels['-100%'] - 2000  # Much lower target
        self._smooth_transition(prices, resistance_idx, end_idx, resistance_price, next_pattern_target)

        return {
            'name': system_name,
            'color': 'red',
            'fan_levels': new_fan_levels,
            'original_fan_levels': {
                'B (0%)': original_B_price,
                '-38.2%': original_fan_382
            },
            'original_AB_move': original_AB_move,
            'AB_move': original_AB_move,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'break_point': (break_idx, break_price),
            'new_support': (support_idx, new_support_price),
            'resistance_point': (resistance_idx, resistance_price),
            'completion_point': (end_idx, next_pattern_target),
            'completion_level': '100%',
            'failure_levels': {
                'fan_886': new_fan_levels['+88.6%'],
                'abcd_382': self.original_abcd['failure_levels']['abcd_382']
            },
            'remeasured_from_support': True,
            'is_fan_extension': True
        }

    def _smooth_transition(self, prices, start_idx, end_idx, start_price, end_price):
        if start_idx >= end_idx or start_idx >= len(prices) or end_idx >= len(prices):
            return
        for i in range(start_idx, end_idx + 1):
            if i >= len(prices):
                break
            progress = (i - start_idx) / (end_idx - start_idx) if end_idx != start_idx else 0
            prices[i] = start_price + (end_price - start_price) * progress

    def plot_bearish_fan_extensions(self, data):
        self.fig, self.ax = plt.subplots(figsize=(28, 16))
        self.ax.plot(data['x_values'], data['prices'],
                     color='black', linewidth=1.5, alpha=0.8, label='Price')
        zone_colors = ['#FFE6FF', '#FFE6E6', '#E6F0FF', '#E6FFE6']

        original = data['original_abcd']
        self.ax.axvspan(original['start_idx'], original['end_idx'],
                        color=zone_colors[0], alpha=0.3, zorder=0)

        for i, fan_system in enumerate(data['fan_systems']):
            zone_color = zone_colors[i + 1]
            self.ax.axvspan(fan_system['start_idx'], fan_system['end_idx'],
                            color=zone_color, alpha=0.3, zorder=0)

        self._plot_bearish_abcd(data['original_abcd'])
        for fan_system in data['fan_systems']:
            self._plot_bearish_fan_system(fan_system)
        self._plot_bearish_failure_levels(data)
        self._add_comprehensive_info_box(data)

        self.ax.set_title('Bearish Cascading Fan Extensions - Downtrend',
                          fontsize=16, fontweight='bold', pad=20)
        self.ax.set_xlabel('Time Index', fontsize=12)
        self.ax.set_ylabel('Price', fontsize=12)
        self.ax.grid(True, alpha=0.3, linestyle='--')

        # Simple legend - just the main systems
        self.ax.plot([], [], color='purple', label='Bearish ABCD')
        for fan_system in data['fan_systems']:
            self.ax.plot([], [], color=fan_system['color'], label=fan_system['name'])

        self.ax.legend(loc='upper right', fontsize=10)  # Upper right for bearish
        plt.tight_layout()
        plt.show()
        return self.fig, self.ax

    def _plot_bearish_failure_levels(self, data):
        """Plot bearish failure levels"""
        # Only show ABCD 38.2% failure level (upward break) - no annotations
        abcd_382_failure = data['original_abcd']['failure_levels']['abcd_382']
        self.ax.axhline(y=abcd_382_failure, color='darkred', linestyle=':',
                        alpha=0.8, linewidth=2)

    def _plot_bearish_abcd(self, original_abcd):
        color = original_abcd['color']
        point_markers = {'A': 'o', 'B': 's', 'C': '^', 'D': 'd'}
        for label, (idx, price) in original_abcd['points'].items():
            self.ax.scatter(idx, price, color=color, marker=point_markers[label],
                            s=150, zorder=5, edgecolors='black', linewidth=2)
            self.ax.annotate(f'{label}', (idx, price),
                             xytext=(5, 15), textcoords='offset points',
                             fontsize=12, fontweight='bold', color=color)
        abcd_x = [original_abcd['points'][p][0] for p in ['A', 'B', 'C', 'D']]
        abcd_y = [original_abcd['points'][p][1] for p in ['A', 'B', 'C', 'D']]
        self.ax.plot(abcd_x, abcd_y, color=color, linewidth=4, alpha=0.8,
                     linestyle='-', label='Bearish ABCD Pattern')

        # Show ABCD 38.2% failure level (upward break)
        abcd_382_failure = original_abcd['failure_levels']['abcd_382']
        self.ax.axhline(y=abcd_382_failure, color='darkred', linestyle=':',
                        alpha=0.8, linewidth=3)

    def _plot_bearish_fan_system(self, fan_system):
        color = fan_system['color']
        annotation_offset = 20

        if 's1_point' in fan_system:
            s1_idx, s1_price = fan_system['s1_point']
            self.ax.scatter(s1_idx, s1_price, color=color, marker='o',
                            s=200, zorder=6, edgecolors='black', linewidth=2)
            self.ax.annotate('S1', (s1_idx, s1_price), xytext=(5, annotation_offset),
                             textcoords='offset points', fontsize=14, color=color, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color))
        if 's2_point' in fan_system:
            s2_idx, s2_price = fan_system['s2_point']
            self.ax.scatter(s2_idx, s2_price, color=color, marker='s',
                            s=200, zorder=6, edgecolors='black', linewidth=2)
            self.ax.annotate('S2', (s2_idx, s2_price), xytext=(5, annotation_offset),
                             textcoords='offset points', fontsize=14, color=color, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color))
        if 's3_point' in fan_system:
            s3_idx, s3_price = fan_system['s3_point']
            self.ax.scatter(s3_idx, s3_price, color=color, marker='^',
                            s=200, zorder=6, edgecolors='black', linewidth=2)
            self.ax.annotate('S3', (s3_idx, s3_price), xytext=(5, annotation_offset),
                             textcoords='offset points', fontsize=14, color=color, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color))
        if 'new_s3_point' in fan_system:
            new_s3_idx, new_s3_price = fan_system['new_s3_point']
            self.ax.scatter(new_s3_idx, new_s3_price, color=color, marker='d',
                            s=200, zorder=6, edgecolors='black', linewidth=2)
            self.ax.annotate('New S3\n(161.8%)', (new_s3_idx, new_s3_price), xytext=(5, annotation_offset),
                             textcoords='offset points', fontsize=12, color=color, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color))
        if 'failure_point' in fan_system:
            failure_idx, failure_price = fan_system['failure_point']
            self.ax.scatter(failure_idx, failure_price, color='red', marker='X',
                            s=300, zorder=8, edgecolors='black', linewidth=3)
            self.ax.annotate('88.6%\nFAILURE', (failure_idx, failure_price), xytext=(10, annotation_offset + 15),
                             textcoords='offset points', fontsize=12, color='red', fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='pink', alpha=0.9, edgecolor='red'))
        if 'recovery_point' in fan_system:
            recovery_idx, recovery_price = fan_system['recovery_point']
            self.ax.scatter(recovery_idx, recovery_price, color=color, marker='v',
                            s=200, zorder=6, edgecolors='black', linewidth=2)
            self.ax.annotate('Recovery', (recovery_idx, recovery_price), xytext=(5, -annotation_offset - 10),
                             textcoords='offset points', fontsize=12, color=color, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color))
        if 'final_break_point' in fan_system:
            final_idx, final_price = fan_system['final_break_point']
            self.ax.scatter(final_idx, final_price, color='darkred', marker='X',
                            s=350, zorder=9, edgecolors='black', linewidth=3)
            self.ax.annotate('ABCD 38.2%\nFINAL BREAK', (final_idx, final_price), xytext=(10, annotation_offset + 20),
                             textcoords='offset points', fontsize=12, color='darkred', fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='mistyrose', alpha=0.9, edgecolor='darkred'))

    def _add_comprehensive_info_box(self, data):
        # Set y-axis limits to focus on the action
        self.ax.set_ylim(25000, 65000)  # Range for bearish patterns

    def save_bearish_pattern_analysis(self, data, filename="bearish_fan_pattern_analysis.txt"):
        """Save detailed analysis of failed and completed bearish patterns to file"""
        with open(filename, 'w') as f:
            f.write("🎯 BEARISH FAN EXTENSIONS PATTERN ANALYSIS\n")
            f.write("=" * 50 + "\n\n")

            # Original ABCD info
            original = data['original_abcd']
            f.write(f"📋 ORIGINAL BEARISH ABCD PATTERN:\n")
            f.write(f"   AB Move: {original['AB_move']:.0f} points (bearish)\n")
            f.write(f"   A: ${original['points']['A'][1]:.0f} (100%)\n")
            f.write(f"   B: ${original['points']['B'][1]:.0f} (0%)\n")
            f.write(f"   C: ${original['points']['C'][1]:.0f} (50%)\n")
            f.write(f"   D: ${original['points']['D'][1]:.0f} (-23.6%)\n")
            f.write(f"   ABCD 38.2% Failure Level: ${original['failure_levels']['abcd_382']:.0f} (upward break)\n\n")

            # Failed patterns
            f.write(f"❌ FAILED BEARISH PATTERNS ({len(data['failed_patterns'])}):\n")
            f.write("-" * 30 + "\n")
            for failed in data['failed_patterns']:
                system = failed['system']
                failure_info = failed['failure_info']
                f.write(f"System: {system['name']}\n")
                f.write(f"Failure Type: {failure_info['failure_type']}\n")
                f.write(f"Failure Level: {failure_info['failure_level']}\n")
                f.write(f"Failure Price: ${failure_info['failure_price']:.0f}\n")
                f.write(f"Time Index: {failure_info['failure_point'][0]}\n")

                if 'failure_levels' in system:
                    f.write(f"88.6% Level: ${system['failure_levels']['fan_886']:.0f} (upward break)\n")
                    f.write(f"ABCD 38.2% Level: ${system['failure_levels']['abcd_382']:.0f} (upward break)\n")
                f.write("\n")

            # Completed patterns
            f.write(f"✅ COMPLETED BEARISH PATTERNS ({len(data['completed_patterns'])}):\n")
            f.write("-" * 30 + "\n")
            for completed in data['completed_patterns']:
                system = completed['system']
                completion_info = completed['completion_info']
                f.write(f"System: {system['name']}\n")
                f.write(f"Completion Level: {completion_info['completion_level']}\n")
                if completion_info['completion_point']:
                    f.write(f"Completion Price: ${completion_info['completion_point'][1]:.0f}\n")
                    f.write(f"Time Index: {completion_info['completion_point'][0]}\n")

                if 'failure_levels' in system:
                    f.write(f"88.6% Level: ${system['failure_levels']['fan_886']:.0f} (upward break)\n")
                    f.write(f"ABCD 38.2% Level: ${system['failure_levels']['abcd_382']:.0f} (upward break)\n")
                f.write("\n")

            # Summary statistics
            total_systems = len(data['fan_systems'])
            failed_count = len(data['failed_patterns'])
            completed_count = len(data['completed_patterns'])

            f.write(f"📊 SUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Bearish Fan Systems: {total_systems}\n")
            f.write(f"Failed Systems: {failed_count}\n")
            f.write(f"Completed Systems: {completed_count}\n")
            f.write(f"Success Rate: {(completed_count / total_systems * 100):.1f}%\n")
            f.write(f"Failure Rate: {(failed_count / total_systems * 100):.1f}%\n\n")

            f.write(f"🚨 BEARISH FAILURE CRITERIA SUMMARY:\n")
            f.write("-" * 25 + "\n")
            f.write(f"1. 88.6% Level Break: Bearish fan system failure (upward break)\n")
            f.write(f"2. ABCD 38.2% Level Break: Final bearish system failure (upward break)\n")
            f.write(f"   ABCD 38.2% Price: ${original['failure_levels']['abcd_382']:.0f}\n")

        print(f"💾 Bearish pattern analysis saved to: {filename}")


def main():
    print("🎯 Generating BEARISH Fan Extensions with Failure Level Tracking...")
    print("📊 Key Bearish Failure Criteria:")
    print("   • 88.6% = Bearish fan failure level (upward break)")
    print("   • ABCD 38.2% = Final brake of bearish fan system (upward break)")

    analyzer = BearishFanExtensions()
    data = analyzer.generate_bearish_fan_pattern()

    # Generate the plot
    fig, ax = analyzer.plot_bearish_fan_extensions(data)

    # Save detailed analysis
    analyzer.save_bearish_pattern_analysis(data)

    # Print summary to console
    print(f"\n📋 ORIGINAL BEARISH ABCD PATTERN:")
    original = data['original_abcd']
    print(f"   AB Move: {original['AB_move']:.0f} points (bearish)")
    print(f"   A: ${original['points']['A'][1]:.0f} (100%)")
    print(f"   B: ${original['points']['B'][1]:.0f} (0%)")
    print(f"   C: ${original['points']['C'][1]:.0f} (50%)")
    print(f"   D: ${original['points']['D'][1]:.0f} (-23.6%)")
    print(f"   ABCD 38.2% Failure: ${original['failure_levels']['abcd_382']:.0f} (upward break)")

    print(f"\n🔄 BEARISH FAN SYSTEMS ANALYSIS:")
    for failed in data['failed_patterns']:
        system = failed['system']
        failure_info = failed['failure_info']
        print(f"\n   ❌ {system['name']}: FAILED")
        print(f"      Type: {failure_info['failure_type']}")
        print(f"      Level: {failure_info['failure_level']}")
        print(f"      Price: ${failure_info['failure_price']:.0f}")

    for completed in data['completed_patterns']:
        system = completed['system']
        completion_info = completed['completion_info']
        print(f"\n   ✅ {system['name']}: COMPLETED")
        print(f"      Level: {completion_info['completion_level']}")
        if completion_info['completion_point']:
            print(f"      Price: ${completion_info['completion_point'][1]:.0f}")

    # Summary statistics
    total = len(data['fan_systems'])
    failed = len(data['failed_patterns'])
    completed = len(data['completed_patterns'])

    print(f"\n📊 SUMMARY:")
    print(f"   Total Systems: {total}")
    print(f"   Failed: {failed}")
    print(f"   Completed: {completed}")
    print(f"   Success Rate: {(completed / total * 100):.1f}%")


if __name__ == "__main__":
    main()