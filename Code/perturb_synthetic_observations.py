import pandas as pd
import numpy as np

from pathlib import Path

np.random.seed(2025)

PROJECT_ROOT = Path(__file__).parent.parent
PATH_TO_SYNTH_TRUTH = Path('/cluster/projects/nn9774k/palm/v24.04/JOBS/Processed_files_flight_21')
   
def SYNTH_TRUTH_FILE_STEM(i):
    SYNTH_TRUTH_FILE_STEM = f"zac_shf_{i}_output"
    return SYNTH_TRUTH_FILE_STEM


# Assumed observation error model, standard deviation of a normal dist. with mean 0.
OBS_ERROR_STDS: dict = {
    "theta": 0.3, 
    "wind_speed":1.0,
    "wind_direction":0.8, 
}

# Converts u and v to wind speed and wind direction.
def u_and_v_to_wind_speed_and_wind_direction(u, v):
    wind_speed = np.sqrt(u**2 + v**2)  
    wind_direction = np.arctan2(-u, -v)
    return wind_speed, wind_direction


def main():

    # Perturbes LES files by adding noise, for synthetic truth and cross validation. 
    for i in range(1,65):
        # Load data
        st_df = pd.read_csv(
        PATH_TO_SYNTH_TRUTH / f"{SYNTH_TRUTH_FILE_STEM(i)}.csv",
        index_col=False,
        )

        # Compute wind speed and direction from u, v
        st_df["wind_speed"], st_df["wind_direction"] = zip(*st_df.apply(lambda row: u_and_v_to_wind_speed_and_wind_direction(row["u"], row["v"]), axis=1))

        # Apply noise only to the specified variables in the error dictionary
        for var, error in OBS_ERROR_STDS.items():
            print(f'{var} for index {i}')
            if var == "wind_speed":
                st_df["wind_speed"] = st_df["wind_speed"] + np.random.normal(loc=0, scale=error, size=st_df.shape[0])
            elif var == "wind_direction":
                st_df["wind_direction"] = (st_df["wind_direction"] + np.random.normal(loc=0, scale=error, size=st_df.shape[0]))
            else:
                st_df[var] = st_df[var] + np.random.normal(loc=0, scale=error, size=st_df.shape[0])

        # Save the modified DataFrame if needed
            st_df.to_csv(
            PATH_TO_SYNTH_TRUTH / f"{SYNTH_TRUTH_FILE_STEM(i)}_perturbed.csv",
            index=False,
        )


if __name__ == "__main__":
    main()








