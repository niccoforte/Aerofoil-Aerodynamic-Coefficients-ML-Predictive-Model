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

The `main.py` script contains the set-up to run the software. Changes can be applied to the final outcome by commenting 
or uncommenting functions or by changing function attributes as these are called.  
A working scenario is set-up as default.

### [`aerofoils.py`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/aerofoils.py)

Coordinate files to represent aerofoil geometries are processes in the `aerofoils.py` script. Files are downloaded from 
online databases including the [UIUC Airfoil Coordinates Database](https://m-selig.ae.illinois.edu/ads/coord_database.html) 
and the [Airfoil Tools Airfoil Database](http://airfoiltools.com/search/airfoils). The coordinates within each `.dat` 
file are read, reformatted, and y-coordiantes are reproduced at consistent cosine-spaced x-locations by fitting of a 
cubic spline for all aerofoils. Profile objects with geometrical information are created to represent each aerofoil 
profile, which are collected in a dictionary and Pandas DataFrame.

Functionalities of this script include:
- Download coordinate files for 1,600+ aerofoils.
- Reshape coordinates between (0, 0) and (1, 0).
- Reproduce y-coordinates at consistent cosined space x-locations.
- Fit smooth cubic spline through coordinates.

###  [`cases.py`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/cases.py)

Aerodynamic coefficient data is processed by the `cases.py` script. Xfoil produced aerodynamic coefficients for a large 
variety of combinations of aerofoils, Re, and AoA are achieved from the 
[Airfoil Tools Airfoil Database](http://airfoiltools.com/search/airfoils). Alternatively, experimental aerodynamic 
coefficients are digitised from original publications or achieved from past digitisations. This script reads each 
`.csv` file, collectes individual "case" information including aerofoil, Re, AoA, and resulting lift (Cl) and drag (Cd)
coefficients, and format each case into rows of a Pandas DataFrame.

Functionalities of this script include:
- Download resulting Cl and Cd for 830,000+ individual combinations of aerofoil, Re, and AoA.
- Structure data.
- Save full dataset to structured `.csv` file.

###  [`data.py`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/data.py)

The Pandas DataFrames produced by the `aerofoils.py` and `cases.py` scripts are merged together in the `data.py` script
in order to produce one single DataFrame including the aerofoil profile coordinates, Re, AoA, Cl and Cd for each case. 
The cases in the final DataFrame are processed into testing and training data, and further into inputs and output (or 
target) data for a Multi-Layer Perceptron (MLP) Neural Network (NN) model.

Functionalities of this script include:
- Merging of aerofoil coordinate and aerodynamic coefficient DataFrames along a shared column.
- Separation of data into Training and Testing NN data.
- Separation of Training and Testing NN data into inputs and targets.

###  [`nnetwork.py`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/blob/main/nnetwork.py)

The ML model is built as an MLP NN by the `nnetwork.py` script. `Model` class objects are created to represent NN 
models, which are created by calling the `nnetwork.run_Model()` function in `main.py` where some hyperparameters 
relevant to their building and training are defined. Models are then built by the the `build_MLP()` class function 
which details all remaining hyperparameters, trained according to further hyperparameters in the `train()` class 
function, evaluated over batches of training and testing data by the `evaluate()` class function, and finally used to 
predict on testing data by the `predict()` class function - where metrics are evaluated between predicted and targert 
Cl and Cd outputs. Functions are included to allow the visualisation of training and prediction metrics, along with 
plotted predicted and target Cl, Cd, and Lift-to-Drag ratio (L/D).

Functionalities of this script include:
- Build and train MLP NN models with developer-defined hyperparameters.
- Evaluate the model over training and testing data.
- Make prediction on unknown testing data and evaluate metrics between predictions and targets.
- Visualize training and prediction metrics.
- Plot predicted and target Cl, Cd, and L/D.


## Jupyer Notebooks

Jupyter Notebooks in the 
[`jupyter`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/tree/main/jupyter) 
directory are provided containing the same functionalities as the `.py` scripts subdivided into cells to allow for 
iterations of certain functions without having to run the full software each time. This can prove useful and time-saving 
when developing new functionalities and defining optimal hyper parameters through iterative training and testing of 
models. 

Only three jupyter notebooks are provided - `aerofoils.ipynb`, `cases.ipynb`, and `nnetwork.ipynb` - which build 
upon each other's outputs by running preceeding notebooks. Scripts can be run in order of cells, where functions that 
would be called by the `main.py` script are simply called in standalone cells of respective notebooks. The 
`aerofoils.ipynb` notebooks retains the same abilities as the `aerofoils.py` script, while functionalities from the
`data.py` script are split into the `cases.ipynb` notebooks - inheriting the ability to merge aerofoil geometry and 
case DataFrames - and the `nnewtwork.ipynb` notebook - producing testing and training input and output data.


## Data

Raw data is abailable in the 
[`dat`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/tree/main/dat) directory, 
including the aerofoil coordinate files, Xfoil simulation coefficient data, and experimental coefficient data.

Structured `.csv` files containing case DataFrames saved by the `cases.py` script are stored in the
[`dat-saved`](https://github.com/niccoforte/Aerofoil-Aerodynamic-Coefficients-ML-Predictive-Model/tree/main/dat-saved)
directory.



## Results