import pandas as pd
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import aerofoils

import tensorflow as tf
from tensorflow import keras
from keras import Sequential, optimizers, initializers
from keras.layers import Dense, BatchNormalization, Activation, InputLayer, LeakyReLU, PReLU, Dropout, Conv2D
from keras.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def accuracy(y_true, y_pred):
    indx = 0
    for t, p in zip(y_true, y_pred):
        if np.abs(t - p) < 0.01:
            indx += 1
        else:
            pass
    score = indx / len(y_pred)
    return score


def standardize(train, test):
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, test


class Model:

    def __init__(self, data, neurons, activation, name, test_df, mod='mlp_tf', EPOCHS=50, BATCH=256, lr=0.1, verbose=0):
        self.train_in = data[0]
        self.train_out = data[1]
        self.test_in = data[2]
        self.test_out = data[3]
        self.neurons = neurons
        self.activation = activation
        self.mod = mod
        self.name = name
        self.test_df = test_df
        self.lr = lr
        self.EPOCHS = EPOCHS
        self.BATCH = BATCH
        self.verbose = verbose

        self.model = None
        self.fitHistory = None
        self.trainEv = None
        self.testEv = None

        print(' Building model...')
        self.model = self.build_MLP_tf()
        print(' -Done. Model successfully built.')
        print(' Training model...')
        self.fitHistory = self.train()
        print(' -Done. Model successully trained.')
        print(' Evaluating model on training and testing data...')
        self.trainEv, self.testEv = self.evaluate()
        print(' -Done. Model evaluations printed above.')
        print(' Predicting on testing data...')
        self.pred, self.Pmetrics_df, self.output_df = self.predict()
        print(' -Done. Predictions made and metrics on predictions evaluated.')

    def build_MLP_tf(self):

        if self.mod == 'mlp_tf':
            model = Sequential()

            model.add(InputLayer(input_shape=len(self.train_in[0])))
            model.add(BatchNormalization())

            for n in self.neurons[1]:
                model.add(Dense(self.neurons[0][n]))
                model.add(BatchNormalization())

                if self.activation == 'leakyrelu':
                    model.add(LeakyReLU(alpha=0.3))
                elif self.activation == 'prelu':
                    model.add(PReLU(alpha_initializer='zeros'))
                else:
                    model.add(Activation(self.activation))

            model.add(Dense(len(self.test_out[0])))

            # model.add(Conv2D(256, 3, 3))
            # model.add(Dropout(0.2))

            OPT = optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            METS = ['ACC', 'MAE', 'MSE', 'MAPE']
            model.compile(optimizer=OPT, loss='MSE', metrics=METS)

            if self.verbose == 1:
                print(model.summary())

        elif self.mod == 'mlp_skl':
            model = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), activation='logistic', solver='adam',
                                 alpha=0.0001, batch_size='auto',
                                 learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5, max_iter=self.EPOCHS,
                                 shuffle=True,
                                 random_state=None, tol=0.0001, verbose=self.verbose, warm_start=False, momentum=0.9,
                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                                 beta_2=0.999,
                                 epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

            self.train_in, self.test_in = standardize(self.train_in, self.test_in)

        elif self.mod == 'rfr':
            model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=None)

        return model

    def train(self):

        if self.mod == 'mlp_tf':
            early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=self.verbose, mode='min')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=self.verbose,
                                          min_delta=1e-4, mode='min')

            fitHistory = self.model.fit(self.train_in, self.train_out, epochs=self.EPOCHS, batch_size=self.BATCH,
                                        validation_split=0.1,
                                        verbose=self.verbose, callbacks=[reduce_lr, early_stop],
                                        class_weight={0: 1, 1: 1})

        elif self.mod == 'mlp_skl' or 'rfr':
            fitHistory = self.model.fit(self.train_in, self.train_out)

        return fitHistory

    def evaluate(self):
        if self.mod == 'mlp_tf':
            trainEv = self.model.evaluate(self.train_in, self.train_out, batch_size=self.BATCH)
            testEv = self.model.evaluate(self.test_in, self.test_out, batch_size=self.BATCH)

        elif self.mod == 'mlp_skl' or 'rfr':
            trainEv = self.model.score(self.train_in, self.train_out)
            print(f'Train Score: {trainEv}.')
            testEv = self.model.score(self.test_in, self.test_out)
            print(f'Test Score: {testEv}.')

        return trainEv, testEv

    def predict(self):
        pred = self.model.predict(self.test_in)

        clp = [p[0] for p in pred]
        cdp = [p[1] for p in pred]
        ldp = [l / d for l, d in zip(clp, cdp)]
        clt = [t[0] for t in self.test_out]
        cdt = [t[1] for t in self.test_out]
        ldt = [l / d for l, d in zip(clt, cdt)]
        p = clp + cdp
        t = clt + cdt

        ACC_cl = accuracy(clt, clp)
        ACC_cd = accuracy(cdt, cdp)
        ACC = accuracy(t, p)
        R2_cl = r2_score(clt, clp)
        R2_cd = r2_score(cdt, cdp)
        MSE_cl = mean_squared_error(clt, clp)
        MSE_cd = mean_squared_error(cdt, cdp)
        MSE = mean_squared_error(t, p)
        MAE_cl = mean_absolute_error(clt, clp)
        MAE_cd = mean_absolute_error(cdt, cdp)
        MAE = mean_absolute_error(t, p)
        MAPE_cl = mean_absolute_percentage_error(clt, clp)
        MAPE_cd = mean_absolute_percentage_error(cdt, cdp)
        MAPE = mean_absolute_percentage_error(t, p)
        Pmetrics_df = pd.DataFrame({'name': [str(self.name)],
                                    'ACC_cl': [float(ACC_cl)], 'ACC_cd': [float(ACC_cd)], 'ACC': [float(ACC)],
                                    'MAE_cl': [float(MAE_cl)], 'MAE_cd': [float(MAE_cd)], 'MAE': [float(MAE)],
                                    'R2_cl': [float(R2_cl)], 'R2_cd': [float(R2_cd)],
                                    'MSE_cl': [float(MSE_cl)], 'MSE_cd': [float(MSE_cd)], 'MSE': [float(MSE)],
                                    'MAPE_cl': [float(MAPE_cl)], 'MAPE_cd': [float(MAPE_cd)], 'MAPE': [float(MAPE)]})

        output_df = self.test_df.copy()
        output_df['LtD'] = ldt
        output_df['Cl_pred'] = clp
        output_df['Cd_pred'] = cdp
        output_df['LtD_pred'] = ldp

        return pred, Pmetrics_df, output_df


def run_Model(data, neurons, activation, test_df, mod='mlp_tf', EPOCHS=50, BATCH=256, lr=0.1, verbose=0):
    print('Building, training, and testing model(s)...')

    models = {}
    for act in activation:
        name = mod.upper() + '-' + act
        print(f' ====  {name}  ====')
        model = Model(data=data,
                      neurons=neurons,
                      activation=act,
                      name=name,
                      test_df=test_df,
                      mod=mod,
                      EPOCHS=EPOCHS,
                      BATCH=BATCH,
                      lr=lr,
                      verbose=verbose)

        models[name] = model

    print('-Done. Model(s) saved in "models" distionary.')
    return models


def model_predict(model, test_in, true, test_df):
    print('Predicting on testing data...')

    pred = model.model.predict(test_in)

    clp = [p[0] for p in pred]
    cdp = [p[1] for p in pred]
    ldp = [l / d for l, d in zip(clp, cdp)]
    clt = [t[0] for t in true]
    cdt = [t[1] for t in true]
    ldt = [l / d for l, d in zip(clt, cdt)]
    p = clp + cdp
    t = clt + cdt

    ACC_cl = accuracy(clt, clp)
    ACC_cd = accuracy(cdt, cdp)
    ACC = accuracy(t, p)
    R2_cl = r2_score(clt, clp)
    R2_cd = r2_score(cdt, cdp)
    MSE_cl = mean_squared_error(clt, clp)
    MSE_cd = mean_squared_error(cdt, cdp)
    MSE = mean_squared_error(t, p)
    MAE_cl = mean_absolute_error(clt, clp)
    MAE_cd = mean_absolute_error(cdt, cdp)
    MAE = mean_absolute_error(t, p)
    MAPE_cl = mean_absolute_percentage_error(clt, clp)
    MAPE_cd = mean_absolute_percentage_error(cdt, cdp)
    MAPE = mean_absolute_percentage_error(t, p)
    Pmetrics_df = pd.DataFrame({'name': [str(model.name)],
                                'ACC_cl': [float(ACC_cl)], 'ACC_cd': [float(ACC_cd)], 'ACC': [float(ACC)],
                                'MAE_cl': [float(MAE_cl)], 'MAE_cd': [float(MAE_cd)], 'MAE': [float(MAE)],
                                'R2_cl': [float(R2_cl)], 'R2_cd': [float(R2_cd)],
                                'MSE_cl': [float(MSE_cl)], 'MSE_cd': [float(MSE_cd)], 'MSE': [float(MSE)],
                                'MAPE_cl': [float(MAPE_cl)], 'MAPE_cd': [float(MAPE_cd)], 'MAPE': [float(MAPE)]})

    output_df = test_df.copy()
    output_df['LtD'] = ldt
    output_df['Cl_pred'] = clp
    output_df['Cd_pred'] = cdp
    output_df['LtD_pred'] = ldp

    print('-Done. Predictions made and metrics on predictions evaluated.')
    return pred, Pmetrics_df, output_df


def pred_metrics(Pmetrics_df=None, models=None, file='results/model-metrics.csv', df_from='current',
                 models_add=False, df_save=False, prnt=False, plot=False):
    print("Extracting metrics on the model's predictions...")

    if df_from == 'current':
        print(' From current model...')
        metrics_df = Pmetrics_df

    elif df_from == 'models':
        print(' From model(s) in "models" dictionary...')
        metrics_df = pd.DataFrame()
        for name, model in models.items():
            new_row = model.Pmetrics_df
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    elif df_from == 'file':
        print(f' From model(s) in {file}...')
        metrics_df = pd.read_csv(file, index_col=0)

    elif df_from == 'new':
        print(' Without any model...')
        metrics_df = pd.Dataframe(columns=['name', 'ACC_cl', 'ACC_cd', 'ACC', 'MAE_cl', 'MAE_cd', 'MAE',
                                           'R2_cl', 'R2_cd', 'MSE_cl', 'MSE_cd', 'MSE', 'MAPE_cl', 'MAPE_cd', 'MAPE'])

    if models_add:
        print('  Including models from "models" dictionary...')
        for name, model in models.items():
            new_row = model.Pmetrics_df
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    if df_save:
        print(f'  And saving all metrics to {file}...')
        metrics_df.to_csv(file)

    if prnt:
        print('  And printing metrics...')
        print('   === PREDICTION METRICS ===')
        for index, row in metrics_df.iterrows():
            print(f'    Model: {row[0]}.')

            print(f'    ACC:   CL: {round(row[1], 4)} and CD: {round(row[2], 4)} and ALL: {round(row[3], 4)}.')
            print(f'    MAE:   CL: {round(row[4], 4)} and CD: {round(row[5], 4)} and ALL: {round(row[6], 4)}.')
            print(f'    R2:    CL: {round(row[7], 4)} and CD: {round(row[8], 4)}.')
            print(f'    MSE:   CL: {round(row[9], 4)} and CD: {round(row[10], 4)} and ALL: {round(row[11], 4)}.')
            print(f'    MAPE:  CL: {round(row[12], 4)} and CD: {round(row[13], 4)} and ALL: {round(row[14], 4)}.')

    if plot:
        print('  And plotting metrics...')
        names = metrics_df.name.tolist()
        cols = ['blue', 'orangered', 'green']

        fig = plt.figure(0)
        fig.suptitle('PREDICTION METRICS', fontsize=16, fontname="Times New Roman", fontweight='bold')
        fig.set_figheight(7)
        fig.set_figwidth(15)
        axs = fig.subplots(2, 2)
        fig.tight_layout(pad=4, h_pad=3.5, w_pad=7)

        rels = [[list(metrics_df.ACC_cl), list(metrics_df.ACC_cd), list(metrics_df.ACC)],
                [list(metrics_df.MAE_cl), list(metrics_df.MAE_cd), list(metrics_df.MAE)],
                [list(metrics_df.MSE_cl), list(metrics_df.MSE_cd), list(metrics_df.MSE)],
                [list(metrics_df.MAPE_cl), list(metrics_df.MAPE_cd), list(metrics_df.MAPE)]]
        titles = ['Accuracy', 'Mean Average Error', 'Mean Squared Error', 'Mean Average Percentage Error']
        labelss = [['ACC_cl', 'ACC_cd', 'ACC'], ['MAE_cl', 'MAE_cd', 'MAE'],
                   ['MSE_cl', 'MSE_cd', 'MSE'], ['MAPE_cl', 'MAPE_cd', 'MAPE']]
        axindxs = [[0, 0, 1, 1], [0, 1, 0, 1]]

        for rel, title, labels, i, j in zip(rels, titles, labelss, axindxs[0], axindxs[1]):
            axs[i, j].set_title(title, fontsize=15, fontname="Times New Roman", fontweight='bold')
            axs[i, j].set_ylabel(title, fontsize=12, fontname="Times New Roman")
            x = np.arange(len(names))
            w = 0.2
            m = 0
            for lst, col, label in zip(rel, cols, labels):
                offset = w * m
                bars = axs[i, j].bar(x + offset, lst, color=col, width=w, label=label)
                axs[i, j].bar_label(bars, padding=0, fontsize=7, fontname="Times New Roman")
                m += 1
            axs[i, j].legend()
            axs[i, j].set_xticks(x + w)
            axs[i, j].set_xticklabels(names, rotation=0)

        plt.show()

    print('-Done. Prediction metrics processed as chosen.')
    return metrics_df


def train_metrics(models, mets, df_from='current', prnt=False, plot=False):
    print("Extracting metrics from the model's training...")

    if df_from == 'current':
        print(' From current model...')
        models = {list(models.keys())[0]: list(models.values())[0]}
        fitHistory = list(models.values())[0].fitHistory

    elif df_from == 'models':
        print(' From model(s) in "models" dictionary...')
        models = models
        fitHistory = [model.fitHistory for model in list(models.values())]

    if prnt:
        print('  And printing metrics...')
        print('  === TRAIN, VAL & EVAL METRICS ===')
        for name, model in models.items():
            print(f'   Model: {name}')
            min1 = min(list(model.fitHistory.history.get(f'val_{mets[0]}')))
            avg1 = sum(list(model.fitHistory.history.get(f'val_{mets[0]}'))) / len(list(model.fitHistory.history.get(f'val_{mets[0]}')))
            print(f'   VALIDATION {mets[0].upper()}:  Lowerst: {min1}, Average: {avg1}.')
            min2 = max(list(model.fitHistory.history.get(f'val_{mets[1]}')))
            avg2 = sum(list(model.fitHistory.history.get(f'val_{mets[1]}'))) / len(list(model.fitHistory.history.get(f'val_{mets[1]}')))
            print(f'   VALIDATION {mets[1].upper()}:  Highest: {min2}, Average: {avg2}.')
            print(f'   EVALUATE:        Train: {model.trainEv}, \n                    Test:  {model.testEv}.')

    if plot:
        print('  And plotting metrics...')
        fig = plt.figure(1)
        fig.suptitle('TRAINING & VALIDATION METRICS', fontsize=16, fontname="Times New Roman", fontweight='bold')
        fig.set_figheight(7)
        fig.set_figwidth(15)
        axs = fig.subplots(2, 2)
        fig.tight_layout(pad=4, h_pad=3.5, w_pad=7)

        for i, cat in zip([0, 1], ['Training', 'Validation']):
            axs[0, i].set_title(f'{cat} {mets[0].upper()} v. Epochs', fontsize=15, fontname="Times New Roman",
                                fontweight='bold')
            axs[0, i].set_ylabel(f'{mets[0].upper()}', fontsize=12, fontname="Times New Roman")
            axs[0, i].set_xlabel('Epoch', fontsize=11, fontname="Times New Roman")
            axs[0, i].set_yscale('log')
            for name, model in models.items():
                if cat == 'Training':
                    axs[0, i].plot(model.fitHistory.history.get(f'{mets[0]}'), label=f'{name} {cat} {mets[0].upper()}')
                elif cat == 'Validation':
                    axs[0, i].plot(model.fitHistory.history.get(f'val_{mets[0]}'), label=f'{name} {cat} {mets[0].upper()}')
            axs[0, i].legend()

        for i, cat in zip([0, 1], ['Training', 'Validation']):
            axs[1, i].set_title(f'{cat} {mets[1].upper()} v. Epochs', fontsize=15, fontname="Times New Roman",
                                fontweight='bold')
            axs[1, i].set_ylabel(f'{mets[1].upper()}', fontsize=12, fontname="Times New Roman")
            axs[1, i].set_xlabel('Epoch', fontsize=11, fontname="Times New Roman")
            #axs[1, i].set_yscale('log')
            for name, model in models.items():
                if cat == 'Training':
                    axs[1, i].plot(model.fitHistory.history.get(f'{mets[1]}'), label=f'{name} {cat} {mets[1].upper()}')
                elif cat == 'Validation':
                    axs[1, i].plot(model.fitHistory.history.get(f'val_{mets[1]}'), label=f'{name} {cat} {mets[1].upper()}')
            axs[1, i].legend()

        plt.show()

    print('-Done. Training metrics processed as chosen.')
    return fitHistory


def predictions(aerofoils_df, output=None, name=None, re=None, file='results/predictions.csv', df_from='current',
                model_add=False, df_save=False, plot=True, err=False):
    print("Handling model's predictions...")

    if df_from == 'current':
        print(' From current model...')
        df = output

    elif df_from == 'file':
        print(f' From model(s) in {file}...')
        df = pd.read_csv(file, index_col=0)

    if model_add:
        print('  Including new model predictions...')
        new_row = output
        df = pd.concat([df, new_row], ignore_index=True)

    if df_save:
        print(f'  And saving all predictions to {file}...')
        df.to_csv(file)

    NAMEs = list(set(df.file.tolist()))
    REs = list(set(df.Re.tolist()))
    print('   REs   - ', [re for re in REs])
    print('   Names - ', [name for name in NAMEs])

    plot_df = df[df.Re == re]
    plot_df = plot_df[plot_df.file == name]

    if plot:
        print('  And plotting predictions...')
        fig1 = plt.figure(2)
        fig1.set_figheight(7)
        fig1.set_figwidth(15)
        fig1.suptitle(plot_df.name.tolist()[0].upper() + ' || Re = {:,}'.format(int(plot_df.Re.tolist()[0])),
                      fontsize=18, fontname="Times New Roman", fontweight='bold')
        axs = fig1.subplots(2, 2)
        fig1.tight_layout(pad=4, h_pad=3.5, w_pad=7)

        axs[0, 0].set_title('Lift Coefficient v. Angle of Attack', fontsize=15, fontname="Times New Roman",
                            fontweight='bold')
        axs[0, 0].set_xlabel('Angle of Attack [deg]', fontsize=12, fontname="Times New Roman")
        axs[0, 0].set_ylabel('Lift Coefficient', fontsize=12, fontname="Times New Roman")
        a, t, p = plot_df.alpha.tolist(), plot_df.Cl.tolist(), plot_df.Cl_pred.tolist()
        axs[0, 0].plot(a, p, '--', lw=1, marker='o', markersize=2, label='Predicted')
        axs[0, 0].plot(a, t, lw=1, marker='o', markersize=2, label='True')
        axs[0, 0].legend(bbox_to_anchor=(0.25, 1))

        if err:
            axs00 = axs[0, 0].twinx()
            axs00.set_ylabel('Error', fontsize=12, fontname="Times New Roman", rotation=270, labelpad=15)
            err = [np.abs(tt - pp) for tt, pp in zip(t, p)]
            axs00.bar(a, err, width=(max(a) - min(a)) / len(a), alpha=0.1, label='Error')
            axs00.set_ylim(0, 1.5 * max(err))
            axs00.legend(bbox_to_anchor=(0.25, 0.75))

        axs[0, 1].set_title('Lift Coefficient v. Angle of Attack', fontsize=15, fontname="Times New Roman",
                            fontweight='bold')
        axs[0, 1].set_xlabel('Angle of Attack [deg]', fontsize=12, fontname="Times New Roman")
        axs[0, 1].set_ylabel('Drag Coefficient', fontsize=12, fontname="Times New Roman")
        a, t, p = plot_df.alpha.tolist(), plot_df.Cd.tolist(), plot_df.Cd_pred.tolist()
        axs[0, 1].plot(a, p, '--', lw=1, marker='o', markersize=2, label='Predicted')
        axs[0, 1].plot(a, t, lw=1, marker='o', markersize=2, label='True')
        axs[0, 1].legend(bbox_to_anchor=(0.55, 1))

        if err:
            axs01 = axs[0, 1].twinx()
            axs01.set_ylabel('Error', fontsize=12, fontname="Times New Roman", rotation=270, labelpad=15)
            err = [np.abs(tt - pp) for tt, pp in zip(t, p)]
            axs01.bar(a, err, width=(max(a) - min(a)) / len(a), alpha=0.1, label='Error')
            axs01.set_ylim(0, 1.5 * max(err))
            axs01.legend(bbox_to_anchor=(0.55, 0.75))

        axs[1, 0].set_title('Lift to Drag Ratio v. Angle of Attack', fontsize=15, fontname="Times New Roman",
                            fontweight='bold')
        axs[1, 0].set_xlabel('Angle of Attack [deg]', fontsize=12, fontname="Times New Roman")
        axs[1, 0].set_ylabel('Lift to Drag Ratio', fontsize=12, fontname="Times New Roman")
        a, t, p = plot_df.alpha.tolist(), plot_df.LtD.tolist(), plot_df.LtD_pred.tolist()
        axs[1, 0].plot(a, p, '--', lw=1, marker='o', markersize=2, label='Predicted')
        axs[1, 0].plot(a, t, lw=1, marker='o', markersize=2, label='True')
        axs[1, 0].legend(bbox_to_anchor=(0.25, 1))

        if err:
            axs10 = axs[1, 0].twinx()
            axs10.set_ylabel('Error', fontsize=12, fontname="Times New Roman", rotation=270, labelpad=15)
            err = [np.abs(tt - pp) for tt, pp in zip(t, p)]
            axs10.bar(a, err, width=(max(a) - min(a)) / len(a), alpha=0.1, label='Error')
            axs10.set_ylim(0, 1.5 * max(err))
            axs10.legend(bbox_to_anchor=(0.25, 0.75))

        aindx = aerofoils_df.loc[aerofoils_df['file'] == name].index[0]
        aerofoils.plot_profile(aerofoils_df, aindx, scatt=False, x_val=False, ax=axs[1, 1], prnt=False)

        plt.show()

    print('-Done. Predictions processed as chosen.')
    return NAMEs, REs, plot_df
