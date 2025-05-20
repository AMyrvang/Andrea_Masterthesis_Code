# Imports
import warnings
import concurrent.futures
import os
import xarray as xr
import multiprocessing as mp
import pandas as pd
from pathlib import Path
from datetime import datetime

# Local dependencies
from Data_processing import process_drone_flight_data, utm_coordinates_to_gridpoints, process_les_files

warnings.filterwarnings('ignore', category=FutureWarning)

ROOT_DIR : Path = Path().resolve().parents[1]


def main() -> None:
    """
    Runs the data processing for both the drone data (flighe 21) and the LES datasets. 
    """

    # Spresify which columns i want from the drone data. 
    columns_to_ceep_drone_data= [
        'time', 
        'drone21_utm_x', 'drone21_utm_y', 'drone21_z', 'drone21_mpcTemp_pot',
        'drone23_utm_x', 'drone23_utm_y', 'drone23_z', 'drone23_mpcTemp_pot',
        'drone25_utm_x', 'drone25_utm_y', 'drone25_z', 'drone25_mpcTemp_pot',
        'drone90_utm_x', 'drone90_utm_y', 'drone90_z', 'drone90_mpcTemp_pot',
        'drone90_WindEstimate_Magnitude__mDs', 'drone90_WindEstimate_Direction__rad'
    ]
    
    # Set paths.
    input_path_drone_data = (ROOT_DIR / 'Github' / 'Masterthings-git-' / 'Data' / 'Drone_flight_data' / 'flight_21_1s.csv')
    final_output_path_drone_data = (ROOT_DIR / 'Github' / 'Masterthings-git-' / 'Data' / 'Processed_Drone_data' / 'flight_21_1_Processed.csv')

    # Sets the start and end time for each drone. 
    overall_start_time = '2022-08-24 10:25:00'
    overall_end_time = '2022-08-24 11:17:00'

    time_interval = 5

    drone_time_windows = {
        'drone90': ('2022-08-24 10:25:25', '2022-08-24 11:15:25'),
        'drone21': ('2022-08-24 10:26:00', '2022-08-24 10:51:00'),
        'drone23': ('2022-08-24 10:26:00', '2022-08-24 10:51:00'),
        'drone25': ('2022-08-24 10:26:00', '2022-08-24 10:51:00'),
    }

    # Runs the data processing for flight 21. 
    process_drone_flight_data(input_path_drone_data, final_output_path_drone_data, columns_to_ceep_drone_data, overall_start_time, overall_end_time, time_interval, drone_time_windows)
    
    # Set the paths and runs the utm to gridpoints. 
    output_path_utm_to_gridpoints = ROOT_DIR / 'Github' / 'Masterthings-git-' / 'Data' / 'Processed_Drone_data' / 'flight_21_1_Processed_utm_and_grid.csv'
    data_utm_coordinates = pd.read_csv(ROOT_DIR / 'Github' / 'Masterthings-git-' / 'Data' / 'Processed_Drone_data' / 'flight_21_1_Processed.csv')
    utm_coordinates_to_gridpoints(data_utm_coordinates, output_path_utm_to_gridpoints , 'utm_x','utm_y','utm_z')


    # Runs the LES shortening. 
    process_les_files(
        output_dir = '/cluster/projects/nn9774k/palm/v24.04/JOBS/Processed_files', 
        start_file_index = 1,
        end_file_index = 64,
        time_interval = 5
    )
   
   
if __name__ == "__main__":
    start = datetime.now()
    print(f'starting {start}')
    main()
    print(f'Finished {datetime.now() - start}')

