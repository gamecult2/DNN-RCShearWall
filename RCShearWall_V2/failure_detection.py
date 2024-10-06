import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats


class ShearWallAnalyzer:
    def __init__(self):
        self.disp = None
        self.force = None
        self.energy = None
        self.stiffness = None

    def load_data(self, disp_file, force_file):
        """Load data from OpenSees output files"""
        self.disp = np.loadtxt(disp_file)[:, 1]  # Skip time column
        self.force = np.loadtxt(force_file)[:, 1]  # Skip time column

    def calculate_energy_dissipation(self):
        """Calculate cumulative energy dissipation"""
        self.energy = np.zeros_like(self.disp)
        for i in range(1, len(self.disp)):
            delta_d = self.disp[i] - self.disp[i - 1]
            avg_force = (self.force[i] + self.force[i - 1]) / 2
            self.energy[i] = self.energy[i - 1] + abs(delta_d * avg_force)

    def calculate_cycle_stiffness(self):
        """Calculate stiffness for each cycle"""
        # Find peaks in displacement (cycle ends)
        peaks, _ = find_peaks(abs(self.disp), height=0.5 * max(abs(self.disp)))

        self.stiffness = []
        for i in range(len(peaks) - 1):
            cycle_disp = self.disp[peaks[i]:peaks[i + 1]]
            cycle_force = self.force[peaks[i]:peaks[i + 1]]

            # Linear regression to get stiffness
            slope, _, _, _, _ = stats.linregress(cycle_disp, cycle_force)
            self.stiffness.append(abs(slope))

        return self.stiffness

    def detect_failure_mode(self):
        """
        Detect failure mode based on multiple criteria
        Returns: Dictionary with failure probabilities and key indicators
        """
        # Calculate key indicators
        self.calculate_energy_dissipation()
        stiffness_history = self.calculate_cycle_stiffness()

        # 1. Strength Degradation Rate
        peak_forces = []
        for i in range(0, len(self.force), 100):  # Sample every 100 points
            peak_forces.append(max(abs(self.force[i:i + 100])))
        strength_deg_rate = np.polyfit(range(len(peak_forces)), peak_forces, 1)[0]

        # 2. Energy Dissipation Rate
        energy_rate = np.diff(self.energy) / np.diff(np.arange(len(self.energy)))
        avg_energy_rate = np.mean(energy_rate)

        # 3. Stiffness Degradation
        stiffness_deg_rate = np.polyfit(range(len(stiffness_history)),
                                        stiffness_history, 1)[0]

        # 4. Displacement Ductility
        yield_disp = self.find_yield_displacement()
        max_disp = max(abs(self.disp))
        ductility = max_disp / yield_disp if yield_disp > 0 else 0

        # 5. Hysteresis Shape Analysis
        pinching_factor = self.calculate_pinching_factor()

        # Calculate failure mode probabilities
        indicators = {
            'strength_degradation': strength_deg_rate,
            'energy_dissipation_rate': avg_energy_rate,
            'stiffness_degradation': stiffness_deg_rate,
            'ductility': ductility,
            'pinching_factor': pinching_factor
        }

        probs = self.calculate_failure_probabilities(indicators)

        return {
            'failure_probabilities': probs,
            'indicators': indicators
        }

    def find_yield_displacement(self):
        """Find yield displacement using bilinear approximation"""
        max_force = max(abs(self.force))
        max_disp = max(abs(self.disp))

        # Initial elastic stiffness (using first 10% of data)
        n_initial = int(len(self.disp) * 0.1)
        k_initial, _ = np.polyfit(self.disp[:n_initial],
                                  self.force[:n_initial], 1)

        # Iterate to find best bilinear fit
        min_error = float('inf')
        yield_disp = 0

        for i in range(len(self.disp)):
            d_y = self.disp[i]
            f_y = self.force[i]

            # Define bilinear curve
            d_points = np.array([0, d_y, max_disp])
            f_points = np.array([0, f_y, f_y])

            # Calculate error
            error = 0
            for j in range(len(self.disp)):
                f_bilinear = np.interp(self.disp[j], d_points, f_points)
                error += (self.force[j] - f_bilinear) ** 2

            if error < min_error:
                min_error = error
                yield_disp = d_y

        return abs(yield_disp)

    def calculate_pinching_factor(self):
        """Calculate pinching factor based on hysteresis shape"""
        # Find zero-crossing points
        zero_crossings = np.where(np.diff(np.signbit(self.force)))[0]

        if len(zero_crossings) < 2:
            return 0

        # Calculate average force at 50% of max displacement
        max_disp = max(abs(self.disp))
        mid_disp_mask = (abs(self.disp) > 0.45 * max_disp) & (abs(self.disp) < 0.55 * max_disp)
        avg_mid_force = np.mean(abs(self.force[mid_disp_mask]))

        # Calculate average maximum force
        avg_max_force = np.mean(abs(self.force[abs(self.disp) > 0.9 * max_disp]))

        return avg_mid_force / avg_max_force if avg_max_force > 0 else 0

    def calculate_failure_probabilities(self, indicators):
        """
        Calculate probability of each failure mode based on indicators
        Returns: Dictionary with probabilities for each failure mode
        """
        # Normalize indicators
        i = indicators
        probs = {
            'flexural': 0.0,
            'shear': 0.0,
            'sliding': 0.0
        }

        # Flexural failure indicators
        if (i['ductility'] > 4.0 and
                i['strength_degradation'] > -0.3 and
                i['pinching_factor'] > 0.5):
            probs['flexural'] += 0.4

        if i['energy_dissipation_rate'] > 0.6:
            probs['flexural'] += 0.3

        if i['stiffness_degradation'] > -0.4:
            probs['flexural'] += 0.3

        # Shear failure indicators
        if (i['ductility'] < 2.0 and
                i['strength_degradation'] < -0.5 and
                i['pinching_factor'] < 0.3):
            probs['shear'] += 0.5

        if i['energy_dissipation_rate'] < 0.3:
            probs['shear'] += 0.3

        if i['stiffness_degradation'] < -0.6:
            probs['shear'] += 0.2

        # Sliding failure indicators
        if (2.0 <= i['ductility'] <= 4.0 and
                -0.5 <= i['strength_degradation'] <= -0.3 and
                i['pinching_factor'] < 0.4):
            probs['sliding'] += 0.4

        if 0.3 <= i['energy_dissipation_rate'] <= 0.6:
            probs['sliding'] += 0.3

        if -0.6 <= i['stiffness_degradation'] <= -0.4:
            probs['sliding'] += 0.3

        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            for mode in probs:
                probs[mode] /= total

        return probs

    def plot_analysis_results(self):
        """Generate comprehensive plots of analysis results"""
        plt.figure(figsize=(15, 10))

        # Force-Displacement Hysteresis
        plt.subplot(2, 2, 1)
        plt.plot(self.disp, self.force, 'b-', linewidth=1)
        plt.xlabel('Displacement')
        plt.ylabel('Force')
        plt.title('Force-Displacement Hysteresis')
        plt.grid(True)

        # Cumulative Energy Dissipation
        plt.subplot(2, 2, 2)
        plt.plot(self.energy, 'r-', linewidth=1)
        plt.xlabel('Step')
        plt.ylabel('Cumulative Energy')
        plt.title('Energy Dissipation')
        plt.grid(True)

        # Stiffness Degradation
        plt.subplot(2, 2, 3)
        plt.plot(self.stiffness, 'g-', linewidth=1)
        plt.xlabel('Cycle')
        plt.ylabel('Stiffness')
        plt.title('Stiffness Degradation')
        plt.grid(True)

        # Peak Forces
        peak_forces = []
        for i in range(0, len(self.force), 100):
            peak_forces.append(max(abs(self.force[i:i + 100])))
        plt.subplot(2, 2, 4)
        plt.plot(peak_forces, 'm-', linewidth=1)
        plt.xlabel('Cycle Group')
        plt.ylabel('Peak Force')
        plt.title('Strength Degradation')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('analysis_results.png')
        plt.show()
        plt.close()


def main():
    # Example usage
    analyzer = ShearWallAnalyzer()
    analyzer.load_data('plot/node_disp.out', 'plot/node_force.out')

    # Perform analysis
    results = analyzer.detect_failure_mode()

    # Print results
    print("\nFailure Mode Analysis Results:")
    print("-" * 50)
    print("\nFailure Probabilities:")
    for mode, prob in results['failure_probabilities'].items():
        print(f"{mode.capitalize():10}: {prob:.2%}")

    print("\nKey Indicators:")
    for indicator, value in results['indicators'].items():
        print(f"{indicator.replace('_', ' ').capitalize():25}: {value:.3f}")

    # Generate plots
    analyzer.plot_analysis_results()
    print("\nPlots have been saved as 'analysis_results.png'")


if __name__ == "__main__":
    main()