import pandas as pd

# Load the Parquet files
input_parameters_df = pd.read_parquet('RCWall_Data/Run_Full/FullData/InputParameters.parquet')
input_displacement_df = pd.read_parquet('RCWall_Data/Run_Full/FullData/InputDisplacement.parquet')
output_shear_df = pd.read_parquet('RCWall_Data/Run_Full/FullData/OutputShear.parquet')

# Inverse the sign of the values
input_displacement_df = -input_displacement_df
output_shear_df = -output_shear_df

# Save the modified data to new Parquet files
input_parameters_df.to_parquet('RCWall_Data/Run_Full/FullData/InputParameters2.parquet')
input_displacement_df.to_parquet('RCWall_Data/Run_Full/FullData/InputDisplacement2.parquet')
output_shear_df.to_parquet('RCWall_Data/Run_Full/FullData/OutputShear2.parquet')

print("Parquet files saved successfully with inverted values.")

# Load the Parquet files
input_displacement_df = pd.read_parquet('RCWall_Data/Run_Full/FullData/InputDisplacement.parquet')
input_displacement2_df = pd.read_parquet('RCWall_Data/Run_Full/FullData/InputDisplacement2.parquet')

output_shear_df = pd.read_parquet('RCWall_Data/Run_Full/FullData/OutputShear.parquet')
output_shear2_df = pd.read_parquet('RCWall_Data/Run_Full/FullData/OutputShear2.parquet')

input_parameters_df = pd.read_parquet('RCWall_Data/Run_Full/FullData/InputParameters.parquet')
input_parameters2_df = pd.read_parquet('RCWall_Data/Run_Full/FullData/InputParameters2.parquet')

# Concatenate the DataFrames
input_displacement_combined = pd.concat([input_displacement_df, input_displacement2_df], ignore_index=True)
output_shear_combined = pd.concat([output_shear_df, output_shear2_df], ignore_index=True)
input_parameters_combined = pd.concat([input_parameters_df, input_parameters2_df], ignore_index=True)

# Save the concatenated DataFrames to new Parquet files
input_displacement_combined.to_parquet('RCWall_Data/Run_Full3/FullData/InputDisplacement.parquet')
output_shear_combined.to_parquet('RCWall_Data/Run_Full3/FullData/OutputShear.parquet')
input_parameters_combined.to_parquet('RCWall_Data/Run_Full3/FullData/InputParameters.parquet')

print("Parquet files saved successfully.")
