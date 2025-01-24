import numpy as np
import pandas as pd
from pathlib import Path


def select_all_extreme_specimens(data_folder, data_size, input_parameters=17, percentile_threshold=20, middle_percentage=20):
    # Parameters to exclude (Ag, protocol, ar, loadF)
    exclude_params = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # Indices of the parameters to exclude (0-based index)
    # exclude_params = [15, 16]
    # Adjust input_parameters to ignore the excluded parameters
    selected_params = [i for i in range(input_parameters) if i not in exclude_params]

    # Read the parquet file and convert to numpy array with selected parameters
    InParams = pd.read_parquet(data_folder / "InputParameters.parquet").iloc[:data_size, selected_params].to_numpy(dtype='float32')

    # Calculate percentile bounds for each parameter
    lower_bounds = np.percentile(InParams, percentile_threshold, axis=0)
    upper_bounds = np.percentile(InParams, 100 - percentile_threshold, axis=0)

    # Define bounds for the middle 20% range (between the 40th and 60th percentiles)
    middle_lower_bounds = np.percentile(InParams, 50-(middle_percentage/2), axis=0)
    middle_upper_bounds = np.percentile(InParams, 50+(middle_percentage/2), axis=0)

    # Create masks for lower, middle, and upper ranges
    lower_mask = InParams <= lower_bounds
    upper_mask = InParams >= upper_bounds
    middle_mask = (InParams >= middle_lower_bounds) & (InParams <= middle_upper_bounds)

    # Find specimens where ALL parameters are in either extreme or middle ranges
    all_lower = np.all(lower_mask, axis=1)
    all_upper = np.all(upper_mask, axis=1)
    all_middle = np.all(middle_mask, axis=1)

    # Combine masks to get specimens that are either all low, all middle, or all high
    selected_mask = all_lower | all_upper | all_middle
    selected_indices = np.where(selected_mask)[0]

    # Create DataFrame with selected specimens
    selected_specimens = pd.DataFrame(
        InParams[selected_indices],
        columns=[f'Param_{i + 1}' for i in selected_params]  # Adjust column names for selected parameters
    )
    selected_specimens.index = selected_indices

    # Add category (lower, middle, or upper)
    categories = []
    for idx in selected_indices:
        if all_lower[idx]:
            categories.append('Lower')
        elif all_upper[idx]:
            categories.append('Upper')
        else:
            categories.append('Middle')
    selected_specimens['Category'] = categories

    return selected_specimens, selected_indices


# Example usage
if __name__ == "__main__":
    data_folder = Path("RCWall_Data/Run_Final_Full/FullData")  # Replace with actual path
    data_size = 100000000  # Replace with your actual data size

    selected_specimens, indices = select_all_extreme_specimens(data_folder, data_size)

    print(f"\nTotal number of selected specimens: {len(indices)}")
    if len(indices) > 0:
        print("\nBreakdown:")
        print(f"Upper specimens: {sum(selected_specimens['Category'] == 'Upper')}")
        print(f"Lower specimens: {sum(selected_specimens['Category'] == 'Lower')}")
        print(f"Middle specimens: {sum(selected_specimens['Category'] == 'Middle')}")
        print("\nSelected specimens:")
        print(selected_specimens)
    else:
        print("\nNo specimens found where all parameters are in extreme or middle ranges.")
