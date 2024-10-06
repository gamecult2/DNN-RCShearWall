import openseespy.opensees as ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ShearWallAnalysis:
    def __init__(self):
        self.wall_height = 3650
        self.wall_length = 1220
        self.wall_thickness = 102
        # Material properties
        self.fc = -30.0  # Concrete compressive strength (MPa)
        self.ec = -0.002  # Strain at maximum strength
        self.ecu = -0.004  # Ultimate strain
        self.ft = 3.0  # Tensile strength (MPa)
        self.Ec = 4700 * np.sqrt(abs(self.fc))  # Concrete elastic modulus

        # Steel properties
        self.fy = 500.0  # Yield strength (MPa)
        self.Es = 200000.0  # Steel elastic modulus (MPa)
        self.bs = 0.01  # Strain hardening ratio

    def analyze_failure_modes(self, strain_data, stress_data, disp_data):
        """
        Analyze recorded data to determine failure modes
        Returns dictionary with failure assessments
        """
        failure_modes = {
            'flexural': False,
            'shear': False,
            'sliding': False,
            'crushing': False
        }

        # Define failure criteria
        max_strain = np.max(np.abs(strain_data))
        max_stress = np.max(np.abs(stress_data))
        max_disp = np.max(np.abs(disp_data))

        # Failure checks based on ACI 318 and common criteria
        # Flexural failure
        if max_strain > abs(self.ecu):
            failure_modes['flexural'] = True

        # Shear failure (simplified check)
        allowable_shear = 0.2 * abs(self.fc)  # Simplified ACI criteria
        if max_stress > allowable_shear:
            failure_modes['shear'] = True

        # Sliding shear check
        slide_limit = self.wall_height / 100  # Common drift limit
        if max_disp > slide_limit:
            failure_modes['sliding'] = True

        # Web crushing check
        crushing_limit = 0.85 * abs(self.fc)
        if max_stress > crushing_limit:
            failure_modes['crushing'] = True

        return failure_modes

    def plot_results(self, strain_data, stress_data, disp_data):
        """Plot analysis results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Plot strains
        ax1.plot(strain_data)
        ax1.set_title('Strain History')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Strain')
        ax1.grid(True)

        # Plot stresses
        ax2.plot(stress_data)
        ax2.set_title('Stress History')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Stress (MPa)')
        ax2.grid(True)

        # Plot displacements
        ax3.plot(disp_data)
        ax3.set_title('Displacement History')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Displacement (mm)')
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig('wall_analysis_results.png')

    def run_analysis(self):
        """Run the complete analysis"""
        try:
            # Read results
            strain_data = np.loadtxt('results/element_strain.out')
            stress_data = np.loadtxt('results/element_forces.out')
            disp_data = np.loadtxt('results/node_disp.out')

            # Analyze failure modes
            failure_modes = self.analyze_failure_modes(strain_data, stress_data, disp_data)

            # Plot results
            self.plot_results(strain_data, stress_data, disp_data)

            # Generate report
            report = self.generate_report(failure_modes, strain_data, stress_data, disp_data)

            return report

        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            return None

    def generate_report(self, failure_modes, strain_data, stress_data, disp_data):
        """Generate analysis report"""
        report = {
            'max_strain': np.max(np.abs(strain_data)),
            'max_stress': np.max(np.abs(stress_data)),
            'max_displacement': np.max(np.abs(disp_data)),
            'failure_modes': failure_modes,
            'primary_failure_mode': None
        }

        # Determine primary failure mode
        if any(failure_modes.values()):
            report['primary_failure_mode'] = next(
                mode for mode, failed in failure_modes.items() if failed
            )

        return report


# Example usage
if __name__ == "__main__":
    # Create analysis object
    wall_analysis = ShearWallAnalysis()

    # Run analysis
    results = wall_analysis.run_analysis()

    if results:
        print("\nAnalysis Results:")
        print(f"Maximum Strain: {results['max_strain']:.6f}")
        print(f"Maximum Stress: {results['max_stress']:.2f} MPa")
        print(f"Maximum Displacement: {results['max_displacement']:.2f} mm")
        print("\nFailure Modes:")
        for mode, failed in results['failure_modes'].items():
            print(f"{mode.title()}: {'Yes' if failed else 'No'}")
        if results['primary_failure_mode']:
            print(f"\nPrimary Failure Mode: {results['primary_failure_mode'].title()}")