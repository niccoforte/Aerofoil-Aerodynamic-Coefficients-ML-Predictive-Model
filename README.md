# Aerofoil Aerodynamic Coefficient ML Predictive Model

By Niccolò Forte  
Supervised by Dr. Jens-Dominik Mueller

This repository contains the software developed for a dissertation thesis submitted to the School of Engineering and 
Material Science in the Queen Mary University of London for the fulfillment of the Bachelor’s degree in Mechanical 
Engineering.

The purpose of this repository is to produce a Machine Learning model, built using the Tensorflow Keras Python API, 
which is capable of predicting the aerodynamic coefficients of aerofoils at a large range of angles of attack (AoA) - 
including high AoA where common simulations tend to be inaccurate or time and computationally expensive. The model is
designed to be trained on large datasets of simulation data and limited amounts of experimental data to produce accurate
experimental lift and drag coefficient predictions for aerofoils at various combinations of Reynolds number (Re) and AoA.

All necessary python packages required to run this software are detailed in the
[`requiremments.txt`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/requirements.txt).

Please refer to the specifications included in the 
[`LICENSE.md`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/LICENSE.md) 
for information on the reuse of this software. It is kindly asked to cite the original software as detailed by the 
[`CITATION.cff`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/CITATION.cff).

## Scripts

###  [`main.py`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/main.py)

The `main.py` script contains the set-up to run the software. Changes can be applied to the sofrware's final outcome by 
changing attributes for the `set_up()` function or for functions imported from other scripts as these are called.  
A working scenario is set-up as default.

### [`aerofoils.py`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/aerofoils.py)

Coordinate files to represent aerofoil geometries are processes in the `aerofoils.py` script. Files are downloaded from 
online databases including the [UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads/coord_database.html) 
and the [Airfoil Tools Airfoil Database](http://airfoiltools.com/search/airfoils). The coordinates within each `.dat` 
file are read, reformatted, and y-coordiantes are reproduced at consistent cosine-spaced x-locations by fitting 
of a cubic spline for all aerofoils. Profile objects with geometrical information are created to represent each aerofoil 
profile, which are collected in a dictionary and Pandas DataFrame.

Functionalities of this script include:
- Download coordinate `.dat` files for 1,600+ aerofoils.
- Reshape coordinates between (0, 0) and (1, 0).
- Reproduce $y$-coordinates at consistent cosined space x-locations.
- Fit smooth cubic spline through coordinates.
- Structure data into dictionary of Profile objects & Pandas DataFrame.

###  [`cases.py`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/cases.py)

Aerodynamic coefficient data is processed by the `cases.py` script. Xfoil produced aerodynamic coefficients for a large 
variety of combinations of aerofoils, Re, and AoA are achieved from the 
[Airfoil Tools Airfoil Database](http://airfoiltools.com/search/airfoils). Alternatively, experimentally achieved 
aerodynamic coefficients are digitised from original publications or achieved from past digitisations. The script reads 
each `.csv` coefficient file, collectes individual "case" information including combinations of aerofoil, Re, AoA, and 
their resulting lift (Cl) and drag (Cd) coefficients, and format each case into rows of a Pandas DataFrame.

Functionalities of this script include:
- Download and read Xfoil Cl and Cd `.csv` files for 830,000+ individual combinations of aerofoil, Re, and AoA.
- Read Experimenal Cl and Cd `.csv` files.
- Structure data into Pantas DataFrame.
- Save and read full dataset to and from structured `.csv` file.

###  [`data.py`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/data.py)

The Pandas DataFrames produced by the `aerofoils.py` and `cases.py` scripts are merged together in the `data.py` script
in order to produce one single data DataFrame including the aerofoil profile coordinates, Re, AoA, Cl and Cd for each 
case. The cases in the data DataFrame are processed into testing and training data, and further split into inputs and 
outputs (or targets) for a Multi-Layer Perceptron (MLP) Neural Network (NN) model.

Functionalities of this script include:
- Merging of aerofoil coordinate and aerodynamic coefficient DataFrames along a shared column.
- Processing of data into Training and Testing NN data.
- Separation of Training and Testing NN data into Inputs and Outputs/Targets.

###  [`nnetwork.py`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/nnetwork.py)

The ML model is built as an MLP NN by the `nnetwork.py` script. `Model` class objects are created to represent NN 
models, initiated by calling the `nnetwork.run_Model()` function in `main.py` where some hyperparameters relevant to 
their building and training are defined. Models are built by the the `build_MLP()` class function which details all 
remaining architectural hyperparameters, trained according to additional hyperparameters in the `train()` class 
function, evaluated over batches of training and testing data by the `evaluate()` class function, and finally used to 
made prediction on testing data by the `predict()` class function - where metrics are evaluated between predicted and 
targert Cl and Cd outputs. Functions are included to allow the visualisation of training and prediction metrics, along 
with plotted predicted and target Cl, Cd, and Lift-to-Drag ratio (L/D).

Functionalities of this script include:
- Build and train MLP NN models with developer-defined hyperparameters.
- Evaluate the model over batches of training and testing data.
- Make predictions on unknown testing inputs.
- Evaluate metrics between predictions and targets.
- Visualize training and prediction metrics.
- Plot predicted and target Cl, Cd, and L/D and relevant aerofoil profile for testing data.


## Jupyer Notebooks

Jupyter Notebooks in the 
[`jupyter`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/tree/main/jupyter) 
directory are provided containing the same functionalities as the `.py` scripts but subdivided into cells to allow for 
iterations of functions without the need to run the full software. This can prove useful and time-saving when developing 
new functionalities and defining optimal hyper parameters through iterative training and testing of NN models. 

Only three jupyter notebooks are provided - `aerofoils.ipynb`, `cases.ipynb`, and `nnetwork.ipynb` - which build 
upon each other's outputs by running preceeding notebooks. Scripts can be run in order of cells, where functions that 
would be called by the `main.py` script are simply called in standalone cells of respective notebooks. The 
`aerofoils.ipynb` notebooks retains the same abilities as the `aerofoils.py` script, while functionalities from the
`data.py` script are split into the `cases.ipynb` notebooks - inheriting the ability to merge aerofoil geometry and 
case DataFrames - and the `nnewtwork.ipynb` notebook - producing testing and training input and output data.


## Data

Raw data is available in the 
[`dat`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/tree/main/dat) directory, 
including the aerofoil coordinate files, Xfoil simulation coefficient data, and experimental coefficient data. Given 
the high quantity of raw data files, subsets of up to 1,000 files are available in the `dat` directory. To download 
all/missing files from the online databases, set the `_reset` parameter of the `set_up()` function in the `main.py` 
script to True (only needed once).

Structured `.csv` files containing case DataFrames saved by the `cases.py` script are stored in the
[`dat-saved`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/tree/main/dat-saved)
directory. These are read to create case DataFrames when the `_reset` parameter of the `set_up()` function in the 
`main.py`script is set to False, whereas they are overwritten when it is set to True.
