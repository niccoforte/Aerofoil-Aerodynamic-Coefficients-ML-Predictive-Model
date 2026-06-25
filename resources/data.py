import pandas as pd
import numpy as np
import math
from sklearn.utils import shuffle

from . import aerofoils


def _unique_foils(df):
    return sorted(df.file.dropna().unique().tolist())


def _sample_foils(foils, count, rng):
    foils = list(foils)
    if count <= 0:
        return []
    if count >= len(foils):
        return foils
    return rng.choice(foils, size=count, replace=False).tolist()


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

    aerofoils_unique = aerofoils_df.drop_duplicates(subset='file', keep='first').copy()
    n_duplicates = len(aerofoils_df) - len(aerofoils_unique)
    if n_duplicates:
        print(f' Warning: ignored {n_duplicates} duplicate aerofoil profile row(s) before merging.')

    data_df = pd.merge(
        aerofoils_unique,
        cases_df,
        on='file',
        how='inner',
        validate='one_to_many',
    )

    print(f'-Done. DataFrames merged successfully with a total of {len(data_df)} final cases.')

    return data_df


def get_data(
    df1,
    df2=None,
    nTrain=100,
    nTest=50,
    p_foil=None,
    p_Re=None,
    random_state=42,
    include_exp_in_train=False,
):
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
    random_state : int, default 42
        Seed used for reproducible train/test aerofoil selection and row shuffling.
    include_exp_in_train : bool, default False
        If True and df2 is provided without p_foil, experimental rows are also included in training. The default avoids
        using the same experimental rows for both testing and weighted training.

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

    rng = np.random.default_rng(random_state)
    NNdata_df = df1.drop(columns=['spline', 'xy_profile']).copy()

    if df2 is not None:
        NNexp_df = df2.drop(columns=['spline', 'xy_profile']).copy()

        if p_foil:
            test_df = NNexp_df[NNexp_df.file == p_foil].copy()
        else:
            test_df = NNexp_df.copy()

        if p_Re:
            test_df = test_df[test_df.Re == p_Re].copy()

        candidate_foils = _unique_foils(NNdata_df)
        if p_foil:
            candidate_foils = [
                foil for foil in candidate_foils
                if foil != p_foil and aerofoils.aerofoil_difference(df=df1, name1=foil, name2=p_foil) > 0.5
            ]

        t_foils = _sample_foils(candidate_foils, nTrain, rng)
        train_df = NNdata_df[NNdata_df.file.isin(t_foils)].copy()
        if p_Re:
            train_df = train_df[train_df.Re != p_Re].copy()

        train_df['weights'] = [1] * len(train_df)

        if p_foil:
            Texp_df = NNexp_df[NNexp_df.file != p_foil].copy()
        elif include_exp_in_train:
            Texp_df = NNexp_df.copy()
        else:
            Texp_df = NNexp_df.iloc[0:0].copy()
            print(
                ' Warning: df2 was provided without p_foil, so experimental rows are reserved for testing only. '
                'Set include_exp_in_train=True to include them in weighted training.'
            )

        if p_Re:
            Texp_df = Texp_df[Texp_df.Re != p_Re].copy()

        Texp_df['weights'] = [5] * len(Texp_df)
        Texp_df.loc[Texp_df.alpha > 10, 'weights'] = 10
        Texp_df.loc[Texp_df.alpha < -10, 'weights'] = 10

        train_df = train_df[~train_df.file.isin(_unique_foils(Texp_df))].copy()

        train_df = shuffle(pd.concat([train_df, Texp_df], ignore_index=True), random_state=random_state)

        print(f' N. Xfoil Training Aerofoils: {len(set(train_df.file.tolist()))}')
        print(f' N. Xfoil in Training: {len(train_df)}')
        print(f' N. Exp. Training Aerofoils: {len(set(Texp_df.file.tolist()))}')
        print(f' N. Experimental in Training: {len(Texp_df)}')

    else:
        if p_foil:
            test_df = NNdata_df[NNdata_df.file == p_foil].copy()
            if p_Re:
                test_df = test_df[test_df.Re == p_Re].copy()

            candidate_foils = [
                foil for foil in _unique_foils(NNdata_df)
                if foil != p_foil and aerofoils.aerofoil_difference(df=df1, name1=foil, name2=p_foil) > 0.5
            ]
            t_foils = _sample_foils(candidate_foils, nTrain, rng)

            train_df = NNdata_df[NNdata_df.file.isin(t_foils)].copy()
            if p_Re:
                train_df = train_df[train_df.Re != p_Re].copy()
            train_df = shuffle(train_df, random_state=random_state)

            print(f' N. Xfoil Training Aerofoils: {len(t_foils)}')
            print(f' N. Xfoil in Training: {len(train_df)}')

        else:
            candidate_foils = _unique_foils(NNdata_df)
            p_foils = _sample_foils(candidate_foils, nTest, rng)
            p_foils_set = set(p_foils)
            train_candidates = [foil for foil in candidate_foils if foil not in p_foils_set]
            t_foils = _sample_foils(train_candidates, nTrain, rng)

            test_df = NNdata_df[NNdata_df.file.isin(p_foils)].copy()
            if p_Re:
                test_df = test_df[test_df.Re == p_Re].copy()

            train_df = NNdata_df[NNdata_df.file.isin(t_foils)].copy()
            if p_Re:
                train_df = train_df[train_df.Re != p_Re].copy()
            train_df = shuffle(train_df, random_state=random_state)

            print(f' N. Xfoil Training Aerofoils: {len(t_foils)}')
            print(f' N. Xfoil in Training: {len(train_df)}')

        train_df['weights'] = [1] * len(train_df)

    if train_df.empty:
        raise ValueError('No training rows were selected. Check nTrain, p_foil, p_Re, and available input data.')
    if test_df.empty:
        raise ValueError('No testing rows were selected. Check nTest, p_foil, p_Re, and available input data.')

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
