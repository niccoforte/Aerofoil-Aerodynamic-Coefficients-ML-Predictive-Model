import pandas as pd


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