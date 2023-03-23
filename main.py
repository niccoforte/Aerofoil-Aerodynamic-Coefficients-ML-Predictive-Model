import aerofoils
import cases
import data
import nnetwork
import pandas as pd


# Download aerofoil .dat files to 'dat/aerofoil-dat' directory and case .csv files to 'dat/case-dat' directory.
aerofoils.get_UIUC_foils(directory='dat/aerofoil-dat')
aerofoils.get_AFT_foils(directory='dat/aerofoil-dat')
aerofoils.get_RENNES_foils(directory='dat/rennes-dat/aerofoil-dat')
cases.get_AFT_cases(directory='dat/case-dat')
cases.get_RENNES_cases(directory='dat/rennes-dat/case-dat')


# Create dictionary of Profile objects and Aerofoils DataFrame.
profiles, aerofoils_df = aerofoils.create_profiles(directory='dat/aerofoil-dat', points=51, prnt=False)
ren_profiles, ren_aerofoils_df = aerofoils.create_profiles(directory='dat/rennes-dat/aerofoil-dat', points=51, prnt=False)


# Create DataFrame of case data.
#cases_df = cases.create_cases(directory='dat/case-dat', ext='csv')
#cases.save_cases(df=cases_df, file='dat-saved/cases-df')
#ren_cases_df = cases.create_cases(directory='dat/rennes-dat/case-dat', ext='txt')
#cases.save_cases(df=ren_cases_df, file='dat-saved/ren-cases-df')
cases_df = cases.df_from_csv(file='dat-saved/cases-df')
ren_cases_df = cases.df_from_csv(file='dat-saved/ren-cases-df')


# Merge aerofoils and cases dataframes.
data_df = data.merge(aerofoils_df, cases_df)
ren_data_df = data.merge(ren_aerofoils_df, cases_df)


# Prepare & get NN inputs & outputs.
train_df, test_df = nnetwork.get_data(df=data_df, profiles=profiles, nTrain=250, nTest=50)
train_in, train_out, test_in, test_out = nnetwork.prep_data(data=[train_df, test_df])
data = [train_in, train_out, test_in, test_out]

ren_train_df, ren_test_df = nnetwork.get_data(df=ren_data_df, profiles=ren_profiles)
ren_train_in, ren_train_out, ren_test_in, ren_test_out = nnetwork.prep_data(data=[ren_train_df, ren_test_df])
ren_data = [ren_train_in, ren_train_out, ren_test_in, ren_test_out]


# Build model(s), train using training data, & predict using testing data.
neurons = [[512, 256, 128, 64, 32, 16], [1, 2, 3, 4, 5]]
activations = ['hard_sigmoid']  # , 'sigmoid', 'softmax', 'softsign', 'leakyrelu', 'selu', 'elu', 'prelu', 'tanh']
activationz = ['swish', 'softplus', 'relu', 'gelu']
models = nnetwork.run_Model(data=data, neurons=neurons, activation=activations, test_df=test_df, mod='mlp_tf',
                            EPOCHS=100, BATCH=256, lr=0.1, verbose=0)
model = list(models.values())[0]
pred, Pmetrics_df, output_df = model.pred, model.Pmetrics_df, model.output_df
train_in, train_out = model.train_in, model.train_out
test_in, test_out = model.test_in, model.test_out


# Predict using testing data.
pred, Pmetrics_df, output_df = nnetwork.model_predict(model=model, test_in=test_in, true=test_out, test_df=test_df)
#pred, Pmetrics_df, output_df = nnetwork.model_predict(model=model, test_in=ren_test_in, true=ren_test_out, test_df=ren_test_df)


# Prediction and training metrics.
fitHistory = nnetwork.train_metrics(models=models, mets=['loss', 'ACC'], df_from='current', prnt=True, plot=True)
metrics_df = nnetwork.pred_metrics(Pmetrics_df, models, file='results/model-metrics.csv', df_from='current',
                                   models_add=False, df_save=False, prnt=True, plot=True)

# Predictions.
NAMEs, REs, plot_df = nnetwork.predictions(aerofoils_df=aerofoils_df, output=output_df, name='goe565', re=200000.0, file='results/predictions.csv',
                                           df_from='current', model_add=False, df_save=False, plot=True, err=True)


# ===========================================================================
# ----- Print DF
#pd.set_option('display.max_columns', None)
#print(data_df)

# ----- Aerofoil Plots
#aindx = aerofoils_df.loc[aerofoils_df.file == 'goe10k'].index[0]
#aerofoils.plot_profile(aerofoils_df, aindx, scatt=True, x_val=0.3, pltfig=1, prnt=True)
