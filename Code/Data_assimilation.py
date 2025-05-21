"""Main running script for inferring LES parameters from synthetic and real drone observations using Bayesian data assimilation"""
# Imports
import pandas as pd
import numpy as np
import itertools
#import glob
from pathlib import Path
from typing import List

# Local dependencies
from EnKS import tsubspaceEnKA
from setup_logging import setup_logging
from config import variables_to_assimilate

PROJECT_ROOT: Path = Path(__file__).parent.parent

def get_synthetic_truth_data() -> pd.DataFrame:
    """ 
    Returns a pandas.DataFrame containing the LES synthetic truth data.
    """
    FNAME = "zac_shf_15_output_perturbed.csv"
    perturbed_st_data_path: Path = Path('/cluster/projects/nn9774k/palm/v24.04/JOBS/Processed_files_flight_21')/FNAME

    return pd.read_csv(perturbed_st_data_path, index_col=False) 


def create_perturbed_obs_matrix(obs_df, number_of_ensemble_members):
    """
    Returns a perturbed observation matrix for the ensamble members (LES data).
    """
    avg_number_of_aggregated_obs = obs_df['timestep_counter'].mean()
    number_of_kalman_iterations = 1 # This is for the non iterative EnKS, change if iterations. 
    number_of_obs = obs_df.shape[0]
    obs_vector = np.zeros(shape = (len(variables_to_assimilate) * number_of_obs, )) 
    obs_error_perturbations = np.zeros(shape = (len(variables_to_assimilate) * number_of_obs, number_of_ensemble_members))

    for i, var in enumerate(variables_to_assimilate.keys()):
        obs_vector[i * number_of_obs: (i + 1) * number_of_obs] = obs_df[var].values

        obs_error_perturbations[i * number_of_obs: (i + 1) * number_of_obs, : ] = \
        np.random.randn(number_of_obs, number_of_ensemble_members) * (variables_to_assimilate[var]['obs_error_std'] / np.sqrt(avg_number_of_aggregated_obs)) * np.sqrt(number_of_kalman_iterations)

    perturbed_obs_matrix = obs_error_perturbations + obs_vector.reshape((len(obs_vector), 1))
    print (perturbed_obs_matrix)

    return perturbed_obs_matrix

def get_les_data() -> List[pd.DataFrame]:
    """
    Returns LES data as a list of pandas.DataFrames, one df per ensemble member.
    """
    result_df = pd.DataFrame()
    les_data_paths = [f'/cluster/projects/nn9774k/palm/v24.04/JOBS/Processed_files_flight_21/zac_shf_{i}_output.csv' for i in itertools.chain(range(1,14+1), range((16), 64+1))]
   
    for i, les_path in enumerate(les_data_paths):
        print(les_path)
        current_df = pd.read_csv(les_path, index_col = False)
        theta_values = current_df['theta']

        wind_speed_values = u_and_v_to_wind_speed_and_wind_direction(current_df['u'], current_df['v'])[0]
        wind_direction_values = u_and_v_to_wind_speed_and_wind_direction(current_df['u'], current_df['v'])[1]

        combined_values = np.concatenate((theta_values.values, wind_speed_values.values,  wind_direction_values.values))
        result_df[f'ens_member_{i}'] = combined_values
    return result_df

 
def get_les_params() -> pd.DataFrame:
    """ 
    Returns the LES paramteres as a pandas.DataFrames.
    """
    FNAME = "priors_shfl_shfs_ug_vg_pt_128n_without_15.csv"
    PARAM_CSV_PATH: Path = Path(PROJECT_ROOT / "Data" / "les_params" / FNAME)
    return pd.read_csv(PARAM_CSV_PATH, index_col=False)


def convert_to_datetime(target_array: pd.Series) -> pd.Series:
    return pd.to_datetime(target_array)


def align_data(model_dfs: List[pd.DataFrame], target_df: pd.DataFrame, model_datetime_col: str, target_datetime_col: str):
    """
    Aligns data to ensure datetime coulumns have date time data type.
    """
    target_df[target_datetime_col] = convert_to_datetime(target_df[target_datetime_col])
    for model_df in model_dfs:
        model_df[model_datetime_col] = convert_to_datetime(model_df[model_datetime_col])

def u_and_v_to_wind_speed_and_wind_direction(u, v):
    """
    Calculates the wind speed to wind driection from the u and v values. 
    """
    wind_speed = np.sqrt(u**2 + v**2)  
    wind_direction = np.arctan2(-u, -v)  
    return wind_speed, wind_direction


def main() -> None:

    # Set up logger
    logger = setup_logging("st_experiment")

    # Load synthetic truth data
    synth_truth_df = get_synthetic_truth_data()
    #print(synth_truth_df)

    # Load LES data
    les_df = get_les_data()
    #print(f"{les_df=}")
    
    perturbed_obs_matrix = create_perturbed_obs_matrix(obs_df = synth_truth_df, number_of_ensemble_members= les_df.shape[1])
    #print(f'{perturbed_obs_matrix=}')

    # Load LES parameters
    les_params = get_les_params()

    # Transpose to make it compatible with DA algorythm
    les_params = les_params.drop(columns = ['run_id'])
    les_params = les_params.iloc[0:63, :] #This is hardcoding stuff
    les_params = les_params.transpose()

    print(f'Printing les_params: {les_params.shape}')
    print(f'Printing perturbed: {perturbed_obs_matrix.shape}')
    print(f'Printing les_df: {les_df.shape}')

    # Add a mask over NaN values to drop them. 
    les_df = les_df.to_numpy()
    
    valid_mask = ~(
        np.isnan(perturbed_obs_matrix).any(axis=1) |
        np.isnan(les_df).any(axis=1)
    )

    perturbed_obs_matrix = perturbed_obs_matrix[valid_mask, :]
    les_df = les_df[valid_mask, :]

    invalid_mask = ~valid_mask

    print(f"Valid rows: {np.sum(valid_mask)}")
    print(f"Invalid rows: {np.sum(invalid_mask)}")

    #Runs the EnKS 
    updated_parameter_matrix = tsubspaceEnKA(theta_mat = les_params, y_observed_peturbed_mat=perturbed_obs_matrix, y_predicted_mat=les_df)

    # Prints the results 
    print(updated_parameter_matrix)
    print(f'shape = {updated_parameter_matrix.shape}')
    logger.info("Loaded data, starting to align synthetic observations and les output.")

    # Saves data as a parameter matrix
    updated_parameter_matrix.to_csv('../Data/Syntetic_run_DA/syntetic_updated_parameter_matrix_flight21.csv')
    print('File saved')


if __name__ == "__main__":
    main()
