import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import math
import random


def merge(aerofoils_df, cases_df):
    """ """

    print('Merging Aerofoils and Cases DataFrames...')

    cases_df['right_index'] = cases_df.index.tolist()

    totdata_df = pd.merge(aerofoils_df, cases_df, on='file', how='outer', indicator=True)
    data_df = totdata_df.loc[totdata_df._merge == 'both'].drop(columns=['_merge'])

    dup, inds = list(data_df.right_index.duplicated()), data_df.index.tolist()
    dup_inds = [i for i, d in zip(inds,dup) if d == True]
    data_df = data_df[~data_df.index.isin(dup_inds)].drop(columns=['right_index'])

    print(f'-Done. DataFrames merged successfully with a total of {len(data_df)} fianl cases.')

    return data_df


def get_data(df, profiles, nTrain=100, nTest=50):
    print('Arranging and preparing data for NN inputs and outputs...')

    NNdata_df = df.drop(columns=['spline', 'xy_profile'])

    if len(df) > 50000:
        indxs1 = [random.randint(0, len(profiles)-1) for i in range(nTrain)]
        names1 = [list(profiles.keys())[indx] for indx in indxs1]

        # naca_df = NNdata_df[NNdata_df.file.str.contains('naca')]
        train_df = shuffle(NNdata_df[NNdata_df.file.str.contains("|".join(names1))])  # [:35515] #.sample(frac=0.8))
        train_df = train_df[~train_df.file.str.contains('goe')]
        # train_df = train_df[train_df.file != 'goe233']
        train_df = train_df[train_df.Re != 200000]

        # indxs2 = [random.randint(0, len(profiles-1)) for i in range(nTest)]
        # names2 = [list(profiles.keys())[indx] for indx in indxs2 if indx not in indxs1]

        # test_df = NNdata_df[~NNdata_df.index.isin(train_df.index)]
        test_df = NNdata_df[NNdata_df['file'].str.contains('goe')]  # "|".join(names2))]
        # test_df = NNdata_df[NNdata_df.file == 'goe233']
        test_df = test_df[test_df.Re == 200000]

    elif len(df) < 50000:
        train_df = NNdata_df
        test_df = NNdata_df  # [NNdata_df.file == 'tilt']

    return train_df, test_df


def prep_data(data):
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
