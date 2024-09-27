def generate_peaks(Dmax, max_points=1000, CycleType="Full", Fact=1, Ncycles=1, output_file_path="data/tmpDsteps.txt"):
    with open(output_file_path, 'w') as outFileID:
        total_points = min(max_points, len(Dmax))  # Ensure the total points do not exceed the number of peak displacements
        num_points_per_peak = total_points // len(Dmax)  # Calculate points per peak
        remaining_points = total_points % len(Dmax)  # Calculate any remaining points

        for i, peak_displacement in enumerate(Dmax):
            for _ in range(Ncycles):  # Repeat cycles
                Disp = 0

                Dmax_scaled = peak_displacement * Fact

                if Dmax_scaled < 0:
                    dx = -Dmax_scaled / num_points_per_peak
                else:
                    dx = Dmax_scaled / num_points_per_peak

                points_to_generate = num_points_per_peak
                if i < remaining_points:
                    points_to_generate += 1  # Distribute remaining points

                for _ in range(points_to_generate):
                    Disp += dx
                    outFileID.write(f'{Disp}\n')  # Write only the numeric value

                if CycleType != "Push":
                    for _ in range(points_to_generate):
                        Disp -= dx
                        outFileID.write(f'{Disp}\n')  # Write only the numeric value

                    if CycleType != "Half":
                        for _ in range(points_to_generate):
                            Disp -= dx
                            outFileID.write(f'{Disp}\n')  # Write only the numeric value

                        for _ in range(points_to_generate):
                            Disp += dx
                            outFileID.write(f'{Disp}\n')  # Write only the numeric value

# Input parameters
Dmax = [12.7, 17.78, 25.4, 35.56, 54.61]
max_points = 1000  # Maximum total number of points
CycleType = "Full"
Fact = 1
Ncycles = 2  # Specify the number of repeated cycles at each peak

# Generate and append displacement increments to the same file
generate_peaks(Dmax, max_points, CycleType, Fact, Ncycles)
