import pandas as pd
import numpy as np
import math
import random
from sklearn.utils import shuffle

import aerofoils


def merge(aerofoils_df, cases_df):
    """Merges two Pandas DataFrames, one containing aerofoil profile geometry information and the other containing
    aerodynamic coefficient cases, on the 'file' column which is present in both DataFrames and contains the filename of
    the relevant aerofoil. As several cases (at different Re and AoA) exist for a signle aerofoil geometry, the rows in
    the geometry DataFrame are duplicated and merged with the matching rows of the coefficient DataFrame.

    Parameters
    ----------
    aerofoils_df : pandas.DataFrame
        Pandas DataFrame containing aerofoil profile geometry information. Format is adapted for DataFrames output by
        the aerofoils.create_profiles() function.
    cases_df : pandas.DataFrame
        Pandas DataFrame aerodynamic coefficit information. Format is adapted for DataFrames output by the
        cases.create_cases() function.
    """

    print('Merging Aerofoils and Cases DataFrames...')

    cases_df['right_index'] = cases_df.index.tolist()

    totdata_df = pd.merge(aerofoils_df, cases_df, on='file', how='outer', indicator=True)
    data_df = totdata_df.loc[totdata_df._merge == 'both'].drop(columns=['_merge'])

    dup, inds = list(data_df.right_index.duplicated()), data_df.index.tolist()
    dup_inds = [i for i, d in zip(inds, dup) if d is True]
    data_df = data_df[~data_df.index.isin(dup_inds)].drop(columns=['right_index'])

    print(f'-Done. DataFrames merged successfully with a total of {len(data_df)} final cases.')

    return data_df


def get_data(df1, df2=None, nTrain=100, nTest=50, p_foil=None, p_Re=None):
    """Arranges input and output data for the training and testing of an MLP Neural Network.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Pandas DataFrame containing Xfoil generated cases.
        Adapted for DataFrame output by data.merge() function.
    df2 : pandas.DataFrame, optional
        Pandas DataFrame containing experimental cases.
        Adapted for DataFrame output by data.merge() function.
    nTrain : int, default 100
        Number of randomly generated aerofoils for which all cases are used to train NN.
    nTest : int, optional, default 50
        Number of randomly generated aerofoils for which all cases are used to test NN.
        Only relevant if df2 and p_foil not defined.
    p_foil : str, optional
        Specific aerofoil for which all cases are used to test NN.
    p_Re : float, optional
        Specific Reynolds number for which all cases for p_foil are used to test NN.

    Returns
    -------
    train_df : pandas.DataFrame
        Shuffled Pandas DataFrame containing the cases to train NN.
    test_df : pandas.DataFrame
        Pandas DataFrame containing the cases to test NN.
    sample_weights : list
        List of int weights assigned to cases during training.
    """

    print('Arranging and preparing data for NN inputs and outputs...')

    NNdata_df = df1.drop(columns=['spline', 'xy_profile'])

    if df2 is not None:
        NNexp_df = df2.drop(columns=['spline', 'xy_profile'])

        if p_foil:
            test_df = NNexp_df[NNexp_df.file == p_foil]
        else:
            test_df = NNexp_df

        if p_Re:
            test_df = test_df[test_df.Re == p_Re]

        t_indx = [random.randint(0, len(set(NNdata_df.file)) - 1) for i in range(nTrain)]
        t_foils = [list(set(NNdata_df.file))[indx] for indx in t_indx]
        if p_foil:
            t_foils = [i for i in t_foils if aerofoils.aerofoil_difference(df=df1, name1=i, name2=p_foil) > 0.5]

        train_df = shuffle(NNdata_df[NNdata_df.file.str.contains("|".join(t_foils))])
        if p_foil:
            train_df = train_df[train_df.file != p_foil]
        if p_Re:
            train_df = train_df[train_df.Re != p_Re]

        train_df['weights'] = [1] * len(train_df)

        if p_foil:
            Texp_df = NNexp_df[NNexp_df.file != p_foil]
        else:
            Texp_df = NNexp_df

        if p_Re:
            Texp_df = Texp_df[Texp_df.Re != p_Re]

        Texp_df['weights'] = [5] * len(Texp_df)
        Texp_df.loc[Texp_df.alpha > 10, 'weights'] = 10
        Texp_df.loc[Texp_df.alpha < -10, 'weights'] = 10

        train_df = train_df[~train_df.file.str.contains("|".join(list(set(Texp_df.file))))]

        train_df = shuffle(pd.concat([train_df, Texp_df]))

        print(f' N. Xfoil Training Aerofoils: {len(set(train_df.file.tolist()))}')
        print(f' N. Xfoil in Training: {len(train_df)}')
        print(f' N. Exp. Training Aerofoils: {len(set(Texp_df.file.tolist()))}')
        print(f' N. Experimental in Training: {len(Texp_df)}')

    else:
        if p_foil:
            test_df = NNdata_df[NNdata_df.file == p_foil]
            if p_Re:
                test_df = test_df[test_df.Re == p_Re]

            t_indx = [random.randint(0, len(set(NNdata_df.file))-1) for i in range(nTrain)]
            t_foils = [list(set(NNdata_df.file))[indx] for indx in t_indx]
            t_foils = [i for i in t_foils if aerofoils.aerofoil_difference(df=df1, name1=i, name2=p_foil) > 0.5]

            train_df = shuffle(NNdata_df[NNdata_df.file.str.contains("|".join(t_foils))])
            train_df = train_df[train_df.file != p_foil]
            if p_Re:
                train_df = train_df[train_df.Re != p_Re]

            print(f' N. Xfoil Training Aerofoils: {len(t_foils)}')
            print(f' N. Xfoil in Training: {len(train_df)}')

        else:
            p_indx = [random.randint(0, len(set(NNdata_df.file)) - 1) for i in range(nTest)]
            p_foils = [list(set(NNdata_df.file))[indx] for indx in p_indx]

            test_df = NNdata_df[NNdata_df['file'].str.contains(f"|".join(p_foils))]
            if p_Re:
                test_df = test_df[test_df.Re == p_Re]

            t_indx = [random.randint(0, len(set(df1.file)) - 1) for i in range(nTrain)]
            t_foils = [list(set(df1.file))[indx] for indx in t_indx if indx not in p_indx]

            train_df = shuffle(NNdata_df[NNdata_df.file.str.contains("|".join(t_foils))])
            if p_Re:
                train_df = train_df[train_df.Re != p_Re]

            print(f' N. Xfoil Training Aerofoils: {len(t_foils)}')
            print(f' N. Xfoil in Training: {len(train_df)}')

        train_df['weights'] = [1] * len(train_df)

    sample_weights = np.array(train_df.weights.tolist())

    return train_df, test_df, sample_weights


def prep_data(data):
    """Transforms training and testing data from Pandas DataFrames into arrays of inputs and outputs compatible for
    testing and training NN.

    Parameters
    ----------
    data : list
        List of two Pandas DataFrames containing data to train and test NN.
        Adapted for DataFrames output by data.get(data) function.

    Returns
    -------
    train_in : numpy.array
        Array of input training data of length number of training cases, each containing the aerofoil profile
        coordinates, Re, and AoA for a specific case.
    train_out : numpy.array
        Array of output training data of length number of training cases, each containing the Cl and Cd for a specific
        case.
    test_in : numpy.array
        Array of input testing data of length number of testing cases, each containing the aerofoil profile
        coordinates, Re, and AoA for a specific case.
    test_out : numpy.array
        Array of output testing data of length number of testing cases, each containing the Cl and Cd for a specific
        case.
    """

    train_in = np.array([[0.0 if math.isnan(y) else y for y in ys_up] +
                         [0.0 if math.isnan(y) else y for y in ys_low] +
                         [float(Re)] + [float(alpha)] for ys_up, ys_low, Re, alpha in
                         zip(data[0].y_up.tolist(), data[0].y_low.tolist(), data[0].Re.tolist(),
                             data[0].alpha.tolist())],
                        dtype='float32')

    train_out = np.array([[float(cl), float(cd)] for cl, cd in zip(data[0].Cl.tolist(), data[0].Cd.tolist())],
                         dtype='float32')

    test_in = np.array([[0.0 if math.isnan(y) else y for y in ys_up] +
                        [0.0 if math.isnan(y) else y for y in ys_low] +
                        [float(Re)] + [float(alpha)] for ys_up, ys_low, Re, alpha in
                        zip(data[1].y_up.tolist(), data[1].y_low.tolist(), data[1].Re.tolist(),
                            data[1].alpha.tolist())],
                       dtype='float32')

    test_out = np.array([[float(cl), float(cd)] for cl, cd in zip(data[1].Cl.tolist(), data[1].Cd.tolist())],
                        dtype='float32')

    print(f'-Done. Data successfully formatted into {len(train_in)} inputs & {len(train_out)} outputs.')
    return train_in, train_out, test_in, test_out
