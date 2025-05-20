
# Imports
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import os
import glob
import warnings
import time
import concurrent.futures

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT_DIR : Path = Path().resolve().parents[1]
print(ROOT_DIR)


def process_drone_flight_data(input_file, final_output_file, columns, overall_start_time, overall_end_time, aggregation_interval, drone_time_windows=None):
    """
    Process the flight data by gathering the correct columns and flight time. 

    Parameters:
    - input_file: The flight data file. 
    - final_output_file: The processed drone data file. 
    - columns: The columns to be gatherd from the flight data. 
    - overall_start_time: Start time for all the drones.
    - overall_end_time: End time for all the drones.
    - aggregation_interval: Spesifick flighttimes for each of the drones. 
    """
    # Load data
    input_file = Path(input_file)
    final_output_file = Path(final_output_file)

    df = pd.read_csv(input_file)
    first_col = df.columns[0]
    if first_col.startswith("Unnamed:"):
        df.rename(columns={first_col: "time"}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    
    # Sets start and end time.
    overall_start_dt = pd.to_datetime(overall_start_time)
    overall_end_dt = pd.to_datetime(overall_end_time)
    df = df[(df['time'] >= overall_start_dt) & (df['time'] <= overall_end_dt)]
    
    # Checks over columns
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
      print("Warning: The following columns are not found in the dataset:", missing_cols)
        
    df = df[columns]
    df = df.loc[:, ~df.columns.duplicated()]

    # Ensure 'time' is structured correcty.
    df['bin_time'] = (df['time'] + pd.Timedelta(seconds=aggregation_interval)).dt.floor(f'{aggregation_interval}S')
    agg_df = df.groupby('bin_time', as_index=False).mean(numeric_only=True)
    agg_df.rename(columns={'bin_time': 'time'}, inplace=True)

    cols = agg_df.columns.tolist()
    cols.remove('time')
    agg_df = agg_df[['time'] + cols]

  
    def extract_drone_data(drone, utm_x, utm_y, alt, theta, windspeed=None, winddir=None):
        """
        Extracting the drone data for only the wanted columns and in the wanted format.

        Parameters:
         - drone: The spresifick drone (21, 23, 25 or 90)
         - utm_x, utm_y: The UTM coordintates in x and y direction. 
         - alt: The hight in z direction, in meter.
        """

        # Set columns
        cols = ['time', utm_x, utm_y, alt, theta]
        rename_map = {
            utm_x: 'utm_x',
            utm_y: 'utm_y',
            alt: 'utm_z',
            theta: 'theta_temp'
        }
        # Windspeed and Windrection for drone 90.
        if drone == "drone90" and windspeed and winddir:
            cols.extend([windspeed, winddir])
            rename_map[windspeed] = 'windspeed'
            rename_map[winddir] = 'wind_direction'
        df_drone = agg_df[cols].rename(columns=rename_map)
        
        # Convert theta_temp from Celsius to Kelvin.
        df_drone['theta_temp'] = df_drone['theta_temp'] + 273.15
        
        df_drone['drone'] = drone

        # For drone90, ensure altitude is positive.
        if drone == "drone90":
            df_drone['utm_z'] = df_drone['utm_z'].abs()
        return df_drone

    # Extract drone data for the spesific drones. 
    df_drone90 = extract_drone_data("drone90", "drone90_utm_x", "drone90_utm_y", "drone90_z",
                                    "drone90_mpcTemp_pot", "drone90_WindEstimate_Magnitude__mDs", "drone90_WindEstimate_Direction__rad")
    df_drone21 = extract_drone_data("drone21", "drone21_utm_x", "drone21_utm_y", "drone21_z", "drone21_mpcTemp_pot")
    df_drone23 = extract_drone_data("drone23", "drone23_utm_x", "drone23_utm_y", "drone23_z", "drone23_mpcTemp_pot")
    df_drone25 = extract_drone_data("drone25", "drone25_utm_x", "drone25_utm_y", "drone25_z", "drone25_mpcTemp_pot")

    long_df = pd.concat([df_drone90, df_drone21, df_drone23, df_drone25], ignore_index=True)
    long_df['time'] = pd.to_datetime(long_df['time'])

    # Dropp all rows without of the time_windows set. 
    if drone_time_windows is not None:
        mask = pd.Series(True, index=long_df.index)
        for drone, (start_str, end_str) in drone_time_windows.items():
            start_dt = pd.to_datetime(start_str)
            end_dt = pd.to_datetime(end_str)
            drone_mask = long_df['drone'] == drone
            mask[drone_mask] = long_df.loc[drone_mask, 'time'].between(start_dt, end_dt)
        long_df = long_df[mask]

    # Sort the data into a set structure with drone 90 first then 21, 23 and lastly 25. 
    drone_order = {"drone90": 0, "drone21": 1, "drone23": 2, "drone25": 3}
    long_df['drone_order'] = long_df['drone'].map(drone_order)
    long_df.sort_values(by=['drone_order', 'time'], inplace=True)
    long_df.drop(columns=['drone_order'], inplace=True)
    long_df.reset_index(drop=True, inplace=True)

    for col in ['utm_x', 'utm_y', 'altitude']:
        if col in long_df.columns:
            long_df[col] = long_df[col].round().astype(int)

    long_df['seconds_after_takeoff'] = (long_df['time'] - overall_start_dt).dt.total_seconds().astype(int)

    # Save to csv file.
    long_df.to_csv(final_output_file, index=False)
    print(f"Final long-format data saved to {final_output_file}")
    return long_df



def utm_coordinates_to_gridpoints(input_file, output_file, utm_x, utm_y, utm_z, center_utm_x=513200, center_utm_y=8265250, center_grid_x=47, center_grid_y=119, cell_size=16):
    """
    Matches the UTM coordinates to the corresponding gridcell in the LES griddsystem.

    Parameters:
    - input_file: The dataframe with the UTM corner coordinates.
    - output_file: The output path and file for the gridded data.  
    - center_utm_x: The center of the UTM coordinates in the x direction (default is put to 513200).
    - center_utm_y: The center of the UTM coordinates in the y direction (default is put to 8265250).
    - center_grid_x: The center of the gridpoint system in the x direction (default is put to 47).
    - center_grid_y: The center of the gridpoint system in y direction (default is put to 119).
    - cell_size: The size of the cells used in the gridpoint (default is put to 16).
    """

    # Calculate grid coordinates
    input_file['grid_x'] = ((input_file[utm_x] - center_utm_x) / cell_size) + center_grid_x
    input_file['grid_y'] = ((input_file[utm_y] - center_utm_y) / cell_size) + center_grid_y
    # Correct rounding and type
    input_file['grid_x'] = np.ceil(input_file['grid_x']).astype(int)
    input_file['grid_y'] = np.ceil(input_file['grid_y']).astype(int)

    # The hights for z. 
    height_boundaries = [0, 8, 24, 40, 56, 72, 88, 104, 120, 136, 152]

    # Function to convert utm_z to grid_z
    def convert_z_to_grid(utm_z):
        adjusted_utm_z = utm_z + 40
        closest_index = min(range(len(height_boundaries)), key=lambda i: abs(height_boundaries[i] - adjusted_utm_z))
        return closest_index

    # Conversion of utm_z to grid_z
    input_file['grid_z'] = input_file[utm_z].apply(convert_z_to_grid)

    input_file.to_csv(output_file, index=False)
    print(f"Final long-format data saved to {output_file}")

    return input_file


# The New, Not parrallelised one. 
def process_les_files(output_dir, start_file_index, end_file_index, time_interval):
    """
    Process LES data from NetCdf files into smaller csv files by only taking data from the wanted gridpoints, with a given time intervall. 
    
    Parameters:
    - output_dir: The directory where the output CSV files will be saved.
    - start_file_index: The starting index for the LES files.
    - end_file_index: The ending index for the LES files.
    - time_interval: The resampling interval in seconds.
    """
    
    # Loads grid points
    grid_points = pd.read_csv('Data/Processed_Drone_data/flight_19_1_Processed_utm_and_grid.csv')

    # Loop through LES_files
    for i in range(start_file_index, end_file_index + 1):
        les_file = f'/cluster/projects/nn9774k/palm/v24.04/JOBS/zac_shf_{i}/OUTPUT/zac_shf_{i}_masked_N02_M001.000.nc'
        
        if not os.path.exists(les_file):
            print(f'File {les_file} does not exist. Skipping index {i}.')
            continue

        print(f'Processing file for index: {i}')

        file_start_time = time.time()

        # Open the LES file
        src = xr.open_dataset(les_file)
        theta = src['theta']
        u_var = src['u']
        v_var = src['v']

        # Convert the time variable to datetime
        base_time = pd.Timestamp('00:00:00')
        times = base_time + pd.to_timedelta(src['time'].values)

        # List to accumulate dataframes for gridpoints
        df_list = []

        # Loop over each gridpoint
        for idx, row in grid_points.iterrows():
            x_idx = int(row['grid_x'])
            y_idx = int(row['grid_y'])
            z_idx = int(row['grid_z'])
            drone_val = row['drone']

            try:
                # Extracct theta, u and v for drone 90 and theta for the rest of the drones
                theta_vals = theta[:, z_idx, y_idx, x_idx].values
                if drone_val == 'drone90':
                    u_vals = u_var[:, z_idx, y_idx, x_idx].values
                    v_vals = v_var[:, z_idx, y_idx, x_idx].values
                else:
                    u_vals = np.full_like(theta_vals, fill_value=np.nan, dtype=float)
                    v_vals = np.full_like(theta_vals, fill_value=np.nan, dtype=float)
            except IndexError:
                print(f'Index error at grid point {idx}.')
                continue

            # Add the values into a dataframe
            df_point = pd.DataFrame({
                'time': times,
                'theta': theta_vals,
                'u': u_vals,
                'v': v_vals
            }).set_index('time')

            # Compute the mean for each time interval
            df_resampled_mean = df_point.resample(
                f'{time_interval}S', label='right', closed='right'
            ).mean().reset_index()

            # Compute the amount of timesteps in each time_interval
            df_resampled_count = df_point.resample(
                f'{time_interval}S', label='right', closed='right'
            ).count().reset_index()

            # Add the timestep count column
            df_resampled_mean['timestep_counter'] = df_resampled_count['theta']

            df_resampled_mean["time"] = df_resampled_mean["time"].dt.strftime('%H:%M:%S')

            # Add gridpoint data
            df_resampled_mean['drone'] = drone_val
            df_resampled_mean['grid_x'] = row['grid_x']
            df_resampled_mean['grid_y'] = row['grid_y']
            df_resampled_mean['grid_z'] = row['grid_z']
            df_resampled_mean['seconds_after_takeoff'] = (df_resampled_mean.index + 1) * time_interval

            df_list.append(df_resampled_mean)

        # Load and save the data 
        if df_list:
            final_df = pd.concat(df_list, ignore_index=True)
            merge_keys = ['drone', 'seconds_after_takeoff']
            final_df_unique = final_df.drop_duplicates(subset=merge_keys, keep='first')

            df_merged = pd.merge(grid_points, 
                     final_df_unique[['time', 'theta', 'u', 'v', 'timestep_counter'] + merge_keys],
                     on=merge_keys,
                     how='left')
            
            df_final = df_merged[['time_y', 'theta', 'u', 'v', 'timestep_counter', 'drone', 'grid_x', 'grid_y', 'grid_z', 'seconds_after_takeoff']].copy()
            df_final.rename(columns={'time_y': 'time'}, inplace=True)

            # Save the final DataFrame to a CSV file
            output_file = os.path.join(output_dir, f'zac_shf_{i}_output.csv')
            df_final.to_csv(output_file, index=False)

            print(f'Data from file index {i} saved to {output_file} in {time.time() - file_start_time:.2f} seconds.')
        else:
            print(f'No data processed for file index {i}.')
