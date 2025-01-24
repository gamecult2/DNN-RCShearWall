import pandas as pd

analysis = 'CyclicData'
folder = 'Run_Full2'
# Load the Parquet files
input_parameters_df = pd.read_parquet(f'RCWall_Data/{folder}/{analysis}/InputParameters.parquet')
input_displacement_df = pd.read_parquet(f'RCWall_Data/{folder}/{analysis}/InputDisplacement.parquet')
output_shear_df = pd.read_parquet(f'RCWall_Data/{folder}/{analysis}/OutputShear.parquet')

# Inverse the sign of the values
input_displacement_df = -input_displacement_df
output_shear_df = -output_shear_df


# Save the modified data to new Parquet files 260860
input_parameters_df.to_parquet(f'RCWall_Data/{folder}/{analysis}/InputParameters2.parquet')
input_displacement_df.to_parquet(f'RCWall_Data/{folder}/{analysis}/InputDisplacement2.parquet')
output_shear_df.to_parquet(f'RCWall_Data/{folder}/{analysis}/OutputShear2.parquet')

print("Parquet files saved successfully with inverted values.")

# Load the Parquet files
input_displacement_df = pd.read_parquet(f'RCWall_Data/{folder}/{analysis}/InputDisplacement.parquet')
input_displacement2_df = pd.read_parquet(f'RCWall_Data/{folder}/{analysis}/InputDisplacement2.parquet')

output_shear_df = pd.read_parquet(f'RCWall_Data/{folder}/{analysis}/OutputShear.parquet')
output_shear2_df = pd.read_parquet(f'RCWall_Data/{folder}/{analysis}/OutputShear2.parquet')

input_parameters_df = pd.read_parquet(f'RCWall_Data/{folder}/{analysis}/InputParameters.parquet')
input_parameters2_df = pd.read_parquet(f'RCWall_Data/{folder}/{analysis}/InputParameters2.parquet')

# Concatenate the DataFrames
input_displacement_combined = pd.concat([input_displacement_df, input_displacement2_df], ignore_index=True)
output_shear_combined = pd.concat([output_shear_df, output_shear2_df], ignore_index=True)
input_parameters_combined = pd.concat([input_parameters_df, input_parameters2_df], ignore_index=True)

# Save the concatenated DataFrames to new Parquet files
input_displacement_combined.to_parquet(f'RCWall_Data/Run_FullM/{analysis}/InputDisplacement.parquet')
output_shear_combined.to_parquet(f'RCWall_Data/Run_FullM/{analysis}/OutputShear.parquet')
input_parameters_combined.to_parquet(f'RCWall_Data/Run_FullM/{analysis}/InputParameters.parquet')

print("Parquet files saved successfully.")
