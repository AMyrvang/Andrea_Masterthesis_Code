# Andrea_Masterthesis_Code
This is all the code and most of the datasets used for my masterproject titeld "". Which implements the "Calibrate, Emulate, Sample" to infer LES data. 

Please note that not all of the data is saved at this repository because of space and file size. Most of the processed files can be located in the `Data` folder. You might also need to modify the file path in the script to correctly access those files. 


### Requirements
To run the Python programs, the following Python packages must be installed:
- Numpy
- Pandas

- Scikit-learn
- Matplotlib
- Seaborn
- 

### Structure
- `Code`Containes all the code files
- `Code/Data_processing.py`: Script for preprocessing the dataset,  `main.py` runs the data processing functions for flight 21. 
- `Code/Emulator_and_MCMC.py`: Containes the NLL emulator and the MCMC, covering the emulate and sample part of the framework. 
- `Code/perturb_synthetic_observations.py`: Perturbed the synthetic truth file by adding noise. 

- `Data`: Most of the data files used in this project. 
- `Tables_and_Figures`: Most of the figures produced in this project. 

### Run code
To successfully execute the code, ensure that all required packages are installed, and then enter the following command in the terminal to run the codes: 

```bash
python Code/main.py
```
```bash
python Code/Data_assimilation.py
```
```bash
python Code/Emulator_and_MCMC.py
```

