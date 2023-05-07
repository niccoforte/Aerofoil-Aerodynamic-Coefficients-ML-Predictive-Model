import aerofoils
import cases
import data
import nnetwork
import pandas as pd


def set_up(_reset=False, _ren=False):
    """ Set-up options for running main.py.

    Parameters
    ----------
    _reset : bool, default False
        If True checks for new files to download from online databases to 'dat/' directories, creates case DataFrames
        from scratch, and saves them by overwriting files in 'dat-saved/' directory. Otherwise uses files already
        present in these directories.
    _ren : bool, default False
        If True runs the script for the profile corrdinates and coefficients downloaded from the study conducted by
        the Université de Rennes. Note NN training and testing will be done using this data too. Otherwise the script
        use profile coordinates and coefficients from UIUC and Airfoil Tools.
    """

    return _reset, _ren


reset, ren = set_up(_reset=False, _ren=False)


# Download aerofoil .dat files to 'dat/aerofoil-dat' directory and case .csv files to 'dat/case-dat' directory.
if reset is True:
    aerofoils.get_UIUC_foils(directory='dat/aerofoil-dat')
    aerofoils.get_AFT_foils(directory='dat/aerofoil-dat')
    cases.get_AFT_cases(directory='dat/case-dat')

    if ren is True:
        aerofoils.get_RENNES_foils(directory='dat/rennes-dat/aerofoil-dat')
        cases.get_RENNES_cases(directory='dat/rennes-dat/case-dat')


# Create dictionary of Profile objects and Aerofoils DataFrame.
profiles, aerofoils_df = aerofoils.create_profiles(directory='dat/aerofoil-dat', points=51, prnt=False)

if ren is True:
    ren_profiles, ren_aerofoils_df = aerofoils.create_profiles(directory='dat/rennes-dat/aerofoil-dat', points=51, prnt=False)


# Create DataFrame of case data.
if reset is True:
    cases_df = cases.create_cases(directory='dat/case-dat', ext='csv')
    cases.save_cases(df=cases_df, file='dat-saved/cases-df.csv')
    exp_cases_df = cases.create_cases(directory='dat/exp-dat/case-dat', ext='csv')
    cases.save_cases(df=exp_cases_df, file='dat-saved/exp-cases-df.csv')

    if ren is True:
        ren_cases_df = cases.create_cases(directory='dat/rennes-dat/case-dat', ext='txt')
        cases.save_cases(df=ren_cases_df, file='dat-saved/ren-cases-df.csv')

if reset is False:
    cases_df = cases.df_from_csv(file='dat-saved/cases-df.csv')
    ren_cases_df = cases.df_from_csv(file='dat-saved/ren-cases-df.csv')
    exp_cases_df = cases.df_from_csv(file='dat-saved/exp-cases-df.csv')


# Merge aerofoils and cases dataframes.
data_df = data.merge(aerofoils_df, cases_df)
exp_data_df = data.merge(aerofoils_df, exp_cases_df)

if ren is True:
    ren_data_df = data.merge(ren_aerofoils_df, ren_cases_df)


# Prepare & get NN inputs & outputs.
train_df, test_df, sample_weights = data.get_data(df1=data_df, df2=exp_data_df, nTrain=250, nTest=50, p_foil='n0012', p_Re=1000000)
train_in, train_out, test_in, test_out = data.prep_data(data=[train_df, test_df])
dat = [train_in, train_out, test_in, test_out]

if ren is True:
    ren_train_df, ren_test_df, sample_weights = data.get_data(df1=ren_data_df, df2=exp_data_df, nTrain=250, nTest=50, p_foil='n0012', p_Re=1000000)
    ren_train_in, ren_train_out, ren_test_in, ren_test_out = data.prep_data(data=[ren_train_df, ren_test_df])
    dat = [ren_train_in, ren_train_out, ren_test_in, ren_test_out]


# Build model(s), train using training data, & predict using testing data.
neurons = [[512, 256, 128, 64, 32, 16], [1, 2, 3, 4]]
activations = ['sigmoid']  # , 'hard_sigmoid', 'softmax', 'softsign', 'leakyrelu', 'selu', 'elu', 'prelu', 'tanh']
activationz = ['swish', 'softplus', 'relu', 'gelu']
models = nnetwork.run_Model(data=dat,
                            neurons=neurons,
                            activation=activations,
                            weights=sample_weights,
                            test_df=test_df,
                            EPOCHS=10,
                            BATCH=256,
                            lr=0.001,
                            verbose=0,
                            callbacks=True)
model = list(models.values())[0]
pred, Pmetrics_df, output_df = model.pred, model.Pmetrics_df, model.output_df


# Predict using testing data.
#pred, Pmetrics_df, output_df = nnetwork.model_predict(model=model, test_in=test_in, test_out=test_out, test_df=test_df)
#pred, Pmetrics_df, output_df = nnetwork.model_predict(model=model, test_in=ren_test_in, test_out=ren_test_out, test_df=ren_test_df)


#  Training and prediction metrics.
fitHistory = nnetwork.train_metrics(models=models,
                                    mets=['loss', 'ACC'],
                                    df_from='current',
                                    prnt=True,
                                    plot=True)
pred_metrics_df = nnetwork.pred_metrics(Pmetrics_df,
                                        models,
                                        file='results/metrics/prediction-mets.csv',
                                        df_from='current',
                                        add=None,
                                        df_save=False,
                                        prnt=True,
                                        plot=True)

# Predictions.
plot_df = nnetwork.predictions(output=output_df,
                               name='n0012',
                               re=1000000,
                               file='results/predictions.csv',
                               df_from='current',
                               aerofoils_df=aerofoils_df,
                               model_add=False,
                               df_save=False,
                               plot=True,
                               err=True)


####################################################################
# ----- Print DF
#pd.set_option('display.max_columns', None)
#print(data_df)

# ----- Aerofoil Plots
#aindx = aerofoils_df.loc[aerofoils_df.file == 'goe10k'].index[0]
#aerofoils.plot_profile(aerofoils_df, aindx, scatt=True, x_val=0.3, pltfig=1, prnt=True)
####################################################################
