import numpy as np


class AxialLoadRatioCalculator:
    def __init__(self):
        """Initialize calculator with standard parameters"""
        self.phi = 0.65  # Strength reduction factor for compression

    def calculate_axial_capacity(self, fc_prime, Ag, Ast=0, fy=420):
        """
        Calculate nominal axial capacity (Pn) according to ACI 318-19

        Parameters:
        -----------
        fc_prime : float
            Specified compressive strength of concrete (MPa)
        Ag : float
            Gross cross-sectional area (mm²)
        Ast : float
            Total area of longitudinal reinforcement (mm²)
        fy : float
            Specified yield strength of reinforcing steel (MPa)

        Returns:
        --------
        float : Nominal axial capacity (N)
        """
        # Calculate areas
        Ac = Ag - Ast  # Net concrete area

        # Calculate nominal axial capacity (ACI 318-19 Section 22.4.2.2)
        Pn = 0.85 * fc_prime * Ac + fy * Ast

        return Pn

    def calculate_axial_load_ratio(self, Pu, fc_prime, Ag, Ast=0, fy=420):
        """
        Calculate axial load ratio (ALR) according to ACI 318-19

        Parameters:
        -----------
        Pu : float
            Factored axial load (N)
        fc_prime : float
            Specified compressive strength of concrete (MPa)
        Ag : float
            Gross cross-sectional area (mm²)
        Ast : float
            Total area of longitudinal reinforcement (mm²)
        fy : float
            Specified yield strength of reinforcing steel (MPa)

        Returns:
        --------
        dict : Dictionary containing axial load ratios and checks
        """
        # Calculate nominal axial capacity
        Pn = self.calculate_axial_capacity(fc_prime, Ag, Ast, fy)

        # Calculate design axial strength
        Pn_design = self.phi * Pn

        # Calculate different axial load ratios
        alr_gross = Pu / (Ag * fc_prime)  # Based on gross area
        alr_nominal = Pu / Pn  # Based on nominal capacity
        alr_design = Pu / Pn_design  # Based on design strength

        # Check limits according to ACI 318-19
        max_axial_load = 0.80 * Pn  # Maximum axial load limit (ACI 318-19 Section 22.4.2.1)
        min_reinforcement_ratio = 0.01  # Minimum reinforcement ratio
        max_reinforcement_ratio = 0.08  # Maximum reinforcement ratio
        reinforcement_ratio = Ast / Ag if Ag > 0 else 0

        # Perform checks
        checks = {
            'within_max_load': Pu <= max_axial_load,
            'reinforcement_ratio_min': reinforcement_ratio >= min_reinforcement_ratio,
            'reinforcement_ratio_max': reinforcement_ratio <= max_reinforcement_ratio
        }

        return {
            'alr_gross': alr_gross,
            'alr_nominal': alr_nominal,
            'alr_design': alr_design,
            'nominal_capacity': Pn,
            'design_capacity': Pn_design,
            'reinforcement_ratio': reinforcement_ratio,
            'checks': checks
        }

    def check_slenderness(self, k, lu, r):
        """
        Check if slenderness effects need to be considered

        Parameters:
        -----------
        k : float
            Effective length factor
        lu : float
            Unsupported length (mm)
        r : float
            Radius of gyration (mm)

        Returns:
        --------
        bool : True if slenderness effects need to be considered
        """
        slenderness_ratio = (k * lu) / r
        return slenderness_ratio > 22  # ACI 318-19 Section 6.2.5


def example_calculation():
    """Example usage of the AxialLoadRatioCalculator"""

    # Initialize calculator
    calculator = AxialLoadRatioCalculator()

    # Example parameters
    fc_prime = 41.75  # MPa
    Ag = 102 * 3650  # mm² (300mm x 300mm column)
    Ast = 0  # mm² (4-#20 bars)
    Pu = 353 * 1000  # N (1000 kN)
    Pu = 132.11996 * 100000

    # Calculate axial load ratio
    results = calculator.calculate_axial_load_ratio(Pu, fc_prime, Ag, Ast)

    # Print results
    print("Axial Load Ratio Analysis Results:")
    print(f"ALR (Gross Area): {results['alr_gross']:.3f}")
    print(f"ALR (Nominal Capacity): {results['alr_nominal']:.3f}")
    print(f"ALR (Design Capacity): {results['alr_design']:.3f}")
    print(f"\nNominal Capacity: {results['nominal_capacity'] / 1000:.1f} kN")
    print(f"Design Capacity: {results['design_capacity'] / 1000:.1f} kN")
    print(f"Reinforcement Ratio: {results['reinforcement_ratio']:.3f}")

    print("\nCode Checks:")
    for check, result in results['checks'].items():
        print(f"{check}: {'Pass' if result else 'Fail'}")


if __name__ == "__main__":
    example_calculation()
