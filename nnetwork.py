import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

import tensorflow as tf
from tensorflow import keras
from keras import Sequential, optimizers
from keras.layers import Dense, BatchNormalization, Activation, InputLayer, LeakyReLU, PReLU, Dropout
from keras.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score

import aerofoils


def accuracy(y_true, y_pred):
    """Evaluates the accuracy of predictions to match target values for two equal length arrays.

    Parameters
    ----------
    y_true : array-like
        Includes target values.
    y_pred : array-like
        Includes predicted values.
    """

    indx = 0
    for t, p in zip(y_true, y_pred):
        if np.abs(t - p) < 0.01:
            indx += 1
        else:
            pass
    score = indx / len(y_pred)
    return score


class Model:
    """Represents an MLP NN model. Includes functions to build, train, evaluate, and predict with the model using the
    Tensorflow Keras API.

    Parameters
    ----------
    data : list
        Contains lists of training and testing inputs and output data.
    neurons : list
        Contains lists of neuron numbers and indexes of neuron numbers to use.
    activation : str
        Activation function to use in Activation Keras layer.
    weights: list
        Training sample_weights assigned to each individual training case.
    name : str
        Model identifier. Mainly of interest when recording metrics and predictions for many models with different
        hyperparameter settings to allow for quick identification.
    test_df : pandas.DataFrame
        DataFrame containing full range of testing data.
    EPOCHS : int, default 50
        Number of training epochs over which the model is trained.
    BATCH : int, default 256
        Number of training samples per gradient update during each epoch of training.
    lr : float, default 0.001
        Learning rate for model optimizer during training.
    verbose : int, default 0
        Keras verbosity mode. 0 = silent, 1 = progress bar, 2 = single line.
        If 1, also prints model summary.
    callbacks : bool, default False
        If True, applies callbacks as defined in train() during training. Otherwise callbacks not applied.

    Attributes
    ----------
    model : keras.Sequential
        Model with architecture and hyperparameters as defined in the build_MLP() function.
    fitHistory : keras Histroy object
        History object returned after fitting the model during training in the model.fit() function.
    trainEv : list
        List of loss and metrics for trained model on testing data.
    testEv : list
        List of loss and metrics for trained model on training data.
    pred : numpy.array
        Predictions evaluated on testing inputs.
    Pmetrics_df : pandas.DataFrame
        DataFrame of prediction metrics evaluated between preictions and targets.
    output_df : pandas.DataFrame
        Copy of output_df with additional columns to represent the target Lift-to-Drag (L/D) ratio, and predicted lift,
        drag, and L/D values.
    """

    def __init__(self, data, neurons, activation, weights, name, test_df, EPOCHS=50, BATCH=256, lr=0.001, verbose=0,
                 callbacks=None):
        self.train_in = data[0]
        self.train_out = data[1]
        self.test_in = data[2]
        self.test_out = data[3]
        self.neurons = neurons
        self.activation = activation
        self.weights = weights
        self.name = name
        self.test_df = test_df
        self.lr = lr
        self.EPOCHS = EPOCHS
        self.BATCH = BATCH
        self.verbose = verbose
        self.callbacks = callbacks

        self.model = None
        self.fitHistory = None
        self.fitHistory_df = None
        self.trainEv = None
        self.testEv = None
        self.ev_df = None
        self.pred = None
        self.Pmetrics_df = None
        self.output_df = None

        print(' Building model...')
        self.model = self.build_MLP()
        print(' -Done. Model successfully built.')
        print(' Training model...')
        self.fitHistory, self.fitHistory_df = self.train()
        print(' -Done. Model successully trained.')
        print(' Evaluating model on training and testing data...')
        self.trainEv, self.testEv, self.ev_df = self.evaluate()
        print(' -Done. Model evaluations printed above.')
        print(' Predicting on testing data...')
        self.pred, self.Pmetrics_df, self.output_df = self.predict(model=self.model, test_in=self.test_in,
                                                                   test_out=self.test_out, test_df=self.test_df)
        print(' -Done. Predictions made and metrics on predictions evaluated.')

    def build_MLP(self):
        """Builds MLP NN model to specified architecutre and hyperparameters.

        Returns
        -------
        model : keras.Sequential
            Model with architecture and hyperparameters as defined in the build_MLP() function.
        """

        model = Sequential(name=self.name)

        model.add(InputLayer(input_shape=len(self.train_in[0])))
        model.add(BatchNormalization())

        for n in self.neurons[1]:
            model.add(Dense(self.neurons[0][n]))

            if self.activation == 'leakyrelu':
                model.add(LeakyReLU(alpha=0.3))
            elif self.activation == 'prelu':
                model.add(PReLU(alpha_initializer='zeros'))
            else:
                model.add(Activation(self.activation))

            model.add(BatchNormalization())
            # model.add(Dropout(0.1))

        model.add(Dense(len(self.test_out[0])))

        OPT = optimizers.Adam(learning_rate=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        METS = ['ACC', 'MAE', 'MSE']
        model.compile(optimizer=OPT, loss='MSE', metrics=METS, weighted_metrics=METS)  # , loss_weights=[1,2])

        if self.verbose == 1:
            print(model.summary())

        return model

    def train(self):
        """Trains model with specified training hyperparameters.

        Returns
        -------
        fitHistory : keras Histroy object
            History object returned after fitting the model during training in the model.fit() function.
        """

        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=self.verbose, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, verbose=self.verbose,
                                      min_delta=1e-4, mode='min')
        if self.callbacks:
            self.callbacks = [reduce_lr]  # , early_stop]

        fitHistory = self.model.fit(self.train_in, self.train_out, epochs=self.EPOCHS, batch_size=self.BATCH,
                                    validation_split=0.1, verbose=self.verbose, callbacks=self.callbacks,
                                    sample_weight=self.weights)  # , class_weight={0:1, 1:1.5})

        fitHistory_df = pd.DataFrame(fitHistory.history)

        return fitHistory, fitHistory_df

    def evaluate(self):
        """Evaluates trained model on training and testing data in batch sizes.

        Returns
        -------
        trainEv : list
            List of loss and metrics for trained model on testing data.
        testEv : list
            List of loss and metrics for trained model on training data.
        """

        trainEv = self.model.evaluate(self.train_in, self.train_out, batch_size=self.BATCH)
        testEv = self.model.evaluate(self.test_in, self.test_out, batch_size=self.BATCH)

        ev_df = pd.DataFrame(columns=list(self.fitHistory_df.columns[:7]))
        ev_df.loc[0] = trainEv
        ev_df.loc[1] = testEv

        return trainEv, testEv, ev_df

    @classmethod
    def predict(self, model, test_in, test_out, test_df):
        """Creates predictions on testing inputs using trained model.

        Parameters
        ----------
        test_in : numpy.array
            Inputs on which model makes predictions.
        test_out : numpy.array
            Targets for predictions on test_in.
        test_df : pandas.DataFrame
            DataFrame containing full range of testing data.

        Returns
        -------
        pred : numpy.array
            Predictions evaluated on testing inputs.
        Pmetrics_df : pandas.DataFrame
            DataFrame of prediction metrics evaluated between preictions and targets.
        output_df : pandas.DataFrame
            Copy of output_df with additional columns to represent the target Lift-to-Drag (L/D) ratio, and predicted lift,
            drag, and L/D values.
        """

        pred = model.predict(test_in)

        clp = [p[0] for p in pred]
        cdp = [p[1] for p in pred]
        ldp = [l / d for l, d in zip(clp, cdp)]
        clt = [t[0] for t in test_out]
        cdt = [t[1] for t in test_out]
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
        RMSE_cl = math.sqrt(MSE_cl)
        RMSE_cd = math.sqrt(MSE_cd)
        RMSE = math.sqrt(MSE)
        Pmetrics_df = pd.DataFrame({'name': [str(model.name)],
                                    'ACC_cl': [float(ACC_cl)], 'ACC_cd': [float(ACC_cd)], 'ACC': [float(ACC)],
                                    'MAE_cl': [float(MAE_cl)], 'MAE_cd': [float(MAE_cd)], 'MAE': [float(MAE)],
                                    'R2_cl': [float(R2_cl)], 'R2_cd': [float(R2_cd)],
                                    'MSE_cl': [float(MSE_cl)], 'MSE_cd': [float(MSE_cd)], 'MSE': [float(MSE)],
                                    'RMSE_cl': [float(RMSE_cl)], 'RMSE_cd': [float(RMSE_cd)], 'RMSE': [float(RMSE)]})

        output_df = test_df.copy()
        output_df = output_df.drop(columns=['x', 'y_up', 'y_low'])
        output_df['LtD'] = ldt
        output_df['Cl_pred'] = clp
        output_df['Cd_pred'] = cdp
        output_df['LtD_pred'] = ldp

        return pred, Pmetrics_df, output_df


def run_Model(data, neurons, activation, weights, test_df, EPOCHS=50, BATCH=256, lr=0.1, verbose=0, callbacks=None):
    """Creates a dictionary of NN model names and objects.

    Parameters
    ----------
    data : list
        Contains lists of training and testing inputs and output data.
    neurons : list
        Contains lists of neuron numbers and indexes of neuron numbers to use.
    activation : list
        List of activation functions to use in Activation Keras layer for each model.
    weights: list
        Training sample_weights assigned to each individual training case.
    test_df : pandas.DataFrame
        DataFrame containing full range of testing data.
    EPOCHS : int, default 50
        Number of training epochs over which the model is trained.
    BATCH : int, default 256
        Number of training samples per gradient update during each epoch of training.
    lr : float, default 0.001
        Learning rate for model optimizer during training.
    verbose : int, default 0
        Keras verbosity mode. 0 = silent, 1 = progress bar, 2 = single line.
        If 1, also prints model summary.
    callbacks : bool, default False
        If True, applies callbacks as defined in train() during training. Otherwise callbacks not applied.

    Returns
    -------
    models : dict
        Dictionary of model names and model objects.
    """

    print('Building, training, and testing model(s)...')

    models = {}
    for act in activation:
        name = f'MLP-{act.capitalize()}'
        dup = len([i for i in os.scandir('models/') if name in str(i)])
        if dup >= 1:
            name = f'{name}-{dup}'

        print(f'  ====  {name}  ====')
        model = Model(data=data,
                      neurons=neurons,
                      activation=act,
                      weights=weights,
                      name=name,
                      test_df=test_df,
                      EPOCHS=EPOCHS,
                      BATCH=BATCH,
                      lr=lr,
                      verbose=verbose,
                      callbacks=callbacks)

        models[name] = model

    print('-Done. Model(s) stored in "models" dictionary.')
    return models


def model_predict(_model, test_in, test_out, test_df):
    """Creates predictions on testing inputs using trained model by running the model.predict() class function.

    Parameters
    ----------
    model : keras.Sequential
        Model with architecture and hyperparameters as defined in the build_MLP() function.
    test_in : numpy.array
        Inputs on which model makes predictions.
    test_out : numpy.array
        Targets for predictions on test_in.
    test_df : pandas.DataFrame
        DataFrame containing full range of testing data.

    Returns
    -------
    pred : numpy.array
        Predictions evaluated on testing inputs.
    Pmetrics_df : pandas.DataFrame
        DataFrame of prediction metrics evaluated between preictions and targets.
    output_df : pandas.DataFrame
        Copy of output_df with additional columns to represent the target Lift-to-Drag (L/D) ratio, and predicted lift,
        drag, and L/D values.
    """

    print('Predicting on testing data...')

    pred, Pmetrics_df, output_df = Model.predict(model=_model, test_in=test_in, test_out=test_out, test_df=test_df)

    print('-Done. Predictions made and metrics on predictions evaluated.')
    return pred, Pmetrics_df, output_df


def pred_metrics(Pmetrics_df, models=None, df_from='current', prnt=False, plot=False):
    """Function to handle the prediction metrics in a variety of way depending by choice of parameters."""

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

    if prnt:
        print('  And printing metrics...')
        print('   === PREDICTION METRICS ===')
        for index, row in metrics_df.iterrows():
            print(f'    MODEL: {row[0]}.')
            print(f'      ACC:  CL: {round(row[1], 6)} and CD: {round(row[2], 6)} and ALL: {round(row[3], 6)}.')
            print(f'      MAE:  CL: {round(row[4], 6)} and CD: {round(row[5], 6)} and ALL: {round(row[6], 6)}.')
            print(f'       R2:  CL: {round(row[7], 6)} and CD: {round(row[8], 6)}.')
            print(f'      MSE:  CL: {round(row[9], 6)} and CD: {round(row[10], 6)} and ALL: {round(row[11], 6)}.')
            print(f'     RMSE:  CL: {round(row[12], 6)} and CD: {round(row[13], 6)} and ALL: {round(row[14], 6)}.')

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
                [list(metrics_df.RMSE_cl), list(metrics_df.RMSE_cd), list(metrics_df.RMSE)]]
        titles = ['Accuracy', 'Mean Average Error', 'Mean Squared Error', 'Root Mean Squared Error']
        labelss = [['ACC_cl', 'ACC_cd', 'ACC'], ['MAE_cl', 'MAE_cd', 'MAE'],
                   ['MSE_cl', 'MSE_cd', 'MSE'], ['RMSE_cl', 'RMSE_cd', 'RMSE']]
        axindxs = [[0, 0, 1, 1], [0, 1, 0, 1]]

        for rel, title, labels, i, j in zip(rels, titles, labelss, axindxs[0], axindxs[1]):
            axs[i, j].set_title(title, fontsize=15, fontname="Times New Roman", fontweight='bold')
            axs[i, j].set_ylabel(title, fontsize=12, fontname="Times New Roman")
            x = np.arange(len(names))
            w = 0.2
            m = 0
            for lst, col, label in zip(rel, cols, labels):
                offset = w * m
                bars = axs[i, j].bar(x + offset, lst, color=col, width=w/1.5, label=label)
                axs[i, j].bar_label(bars, padding=0, fontsize=11, fontname="Times New Roman")
                m += 1
            axs[i, j].set_ylim(0, max(rel[0] + rel[1] + rel[2]) * 1.25)
            axs[i, j].legend()
            axs[i, j].set_xticks(x + w)
            axs[i, j].set_xticklabels(names, rotation=0)

        plt.show()

    print('-Done. Prediction metrics processed as chosen.')
    return


def train_metrics(model, models, mets=['loss', 'ACC', 'MAE'], df_from='current', prnt=False, plot=False):
    """Function to handle the training metrics in a variety of way depending by choice of parameters."""

    print("Extracting metrics from the model's training...")

    if df_from == 'current':
        print(' From current model...')
        models = {model.name: model}
        fitHistory = model.fitHistory

    elif df_from == 'models':
        print(' From model(s) in "models" dictionary...')
        models = models
        fitHistory = [model.fitHistory for model in list(models.values())]

    if prnt:
        print('  And printing metrics...')
        print('=== TRAIN, VAL & EVAL METRICS ===')
        for name, model in models.items():
            print(f'MODEL           :  {name}')
            train_loss = list(model.fitHistory.history.get(f'{mets[0]}'))[-1]
            val_loss = list(model.fitHistory.history.get(f'val_{mets[0]}'))[-1]
            print(f'{mets[0].upper()} :  Training: {round(train_loss, 6)}, Validation: {round(val_loss, 6)}.')
            train_acc = list(model.fitHistory.history.get(f'{mets[1]}'))[-1]
            val_acc = list(model.fitHistory.history.get(f'val_{mets[1]}'))[-1]
            print(f'{mets[1].upper()} :  Training: {round(train_acc, 4)}, Validation: {round(val_acc, 4)}.')
            if len(mets) == 3:
                train_mae = list(model.fitHistory.history.get(f'{mets[2]}'))[-1]
                val_mae = list(model.fitHistory.history.get(f'val_{mets[2]}'))[-1]
                print(f'{mets[2].upper()} :  Training: {round(train_mae, 4)}, Validation: {round(val_mae, 4)}.')
            print(f'EVALUATE \n Train: {model.trainEv}, \n Test:  {model.testEv}.', '\n')

    if plot:
        print('  And plotting metrics...')
        fig = plt.figure(1)
        fig.suptitle('TRAINING & VALIDATION METRICS', fontsize=20, fontname="Times New Roman", fontweight='bold')
        fig.set_figheight(7)
        fig.set_figwidth(15)
        if len(mets) == 2:
            ax1, ax2 = fig.subplots(1, 2)
        elif len(mets) == 3:
            ax1, ax2, ax3 = fig.subplots(1, 3)
        fig.tight_layout(pad=4, h_pad=3.5, w_pad=7)

        ax1.set_title('Loss (MSE) v. Epochs', fontsize=20, fontname="Times New Roman", fontweight='bold')
        ax1.set_ylabel('MSE', fontsize=18, fontname="Times New Roman")
        ax1.set_xlabel('Epoch', fontsize=18, fontname="Times New Roman")
        ax1.set_yscale('log')
        for name, model in models.items():
            ax1.plot(model.fitHistory.history.get(f'{mets[0]}'), label=f'Training {mets[0].upper()}')
            ax1.plot(model.fitHistory.history.get(f'val_{mets[0]}'), label=f'Validation {mets[0].upper()}')
        ax1.legend()

        ax2.set_title('Accuracy v. Epochs', fontsize=20, fontname="Times New Roman", fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=18, fontname="Times New Roman")
        ax2.set_xlabel('Epoch', fontsize=18, fontname="Times New Roman")
        for name, model in models.items():
            ax2.plot(model.fitHistory.history.get(f'{mets[1]}'), label=f'Training {mets[1].upper()}')
            ax2.plot(model.fitHistory.history.get(f'val_{mets[1]}'), label=f'Validation {mets[1].upper()}')
        ax2.legend()

        if len(mets) == 3:
            ax3.set_title('MAE v. Epochs', fontsize=20, fontname="Times New Roman", fontweight='bold')
            ax3.set_ylabel('MAE', fontsize=18, fontname="Times New Roman")
            ax3.set_xlabel('Epoch', fontsize=18, fontname="Times New Roman")
            ax3.set_yscale('log')
            for name, model in models.items():
                ax3.plot(model.fitHistory.history.get(f'{mets[2]}'), label=f'Training {mets[2].upper()}')
                ax3.plot(model.fitHistory.history.get(f'val_{mets[2]}'), label=f'Validation {mets[2].upper()}')
            ax3.legend()

        plt.show()

    print('-Done. Training metrics processed as chosen.')
    return


def predictions(aerofoils_df, output_df=None, name=None, re=None, plot=True, err=False):

    """Function to handle the predictions in a variety of way depending by choice of parameters."""

    print("Handling model's predictions...")

    df = output_df

    NAMEs = list(set(df.file.tolist()))
    REs = list(set(df.Re.tolist()))
    print('  REs   - ', [re for re in REs])
    print('  Names - ', [name for name in NAMEs])

    plot_df = df[df.Re == re]
    plot_df = plot_df[plot_df.file == name]

    if plot:
        print(' And plotting predictions...')
        fig1 = plt.figure(2)
        fig1.set_figheight(8)
        fig1.set_figwidth(15)
        fig1.suptitle(plot_df.name.tolist()[0].upper() + ' || Re = {:,}'.format(int(plot_df.Re.tolist()[0])),
                      fontsize=22, fontname="Times New Roman", fontweight='bold')
        axs = fig1.subplots(2, 2)
        fig1.tight_layout(pad=4, h_pad=5.5, w_pad=7)

        axs[0, 0].set_title('Lift Coefficient v. Angle of Attack', fontsize=20, fontname="Times New Roman",
                            fontweight='bold')
        axs[0, 0].set_xlabel('Angle of Attack [deg]', fontsize=18, fontname="Times New Roman")
        axs[0, 0].set_ylabel('Lift Coefficient', fontsize=18, fontname="Times New Roman")
        a, t, p = plot_df.alpha.tolist(), plot_df.Cl.tolist(), plot_df.Cl_pred.tolist()
        axs[0, 0].plot(a, p, '--', lw=1, marker='o', markersize=2, label='Predicted')
        axs[0, 0].plot(a, t, lw=1, marker='o', markersize=2, label='True')
        axs[0, 0].legend()  # bbox_to_anchor=(0.15,1))

        if err:
            axs00 = axs[0, 0].twinx()
            axs00.set_ylabel('Error', fontsize=18, fontname="Times New Roman", rotation=270, labelpad=15)
            err = [np.abs(tt - pp) for tt, pp in zip(t, p)]
            axs00.bar(a, err, width=(max(a) - min(a)) / (2 * len(a)), alpha=0.1, label='Error')
            axs00.set_ylim(0, 1.5 * max(err))
            axs00.legend(bbox_to_anchor=(0.25, 1))

        axs[0, 1].set_title('Lift Coefficient v. Angle of Attack', fontsize=20, fontname="Times New Roman",
                            fontweight='bold')
        axs[0, 1].set_xlabel('Angle of Attack [deg]', fontsize=18, fontname="Times New Roman")
        axs[0, 1].set_ylabel('Drag Coefficient', fontsize=18, fontname="Times New Roman")
        a, t, p = plot_df.alpha.tolist(), plot_df.Cd.tolist(), plot_df.Cd_pred.tolist()
        axs[0, 1].plot(a, p, '--', lw=1, marker='o', markersize=2, label='Predicted')
        axs[0, 1].plot(a, t, lw=1, marker='o', markersize=2, label='True')
        axs[0, 1].legend()  # bbox_to_anchor=(0.3,1))

        if err:
            axs01 = axs[0, 1].twinx()
            axs01.set_ylabel('Error', fontsize=18, fontname="Times New Roman", rotation=270, labelpad=15)
            err = [np.abs(tt - pp) for tt, pp in zip(t, p)]
            axs01.bar(a, err, width=(max(a) - min(a)) / (2 * len(a)), alpha=0.1, label='Error')
            axs01.set_ylim(0, 1.5 * max(err))
            axs01.legend(bbox_to_anchor=(0.4, 1))

        axs[1, 0].set_title('Lift to Drag Ratio v. Angle of Attack', fontsize=20, fontname="Times New Roman",
                            fontweight='bold')
        axs[1, 0].set_xlabel('Angle of Attack [deg]', fontsize=18, fontname="Times New Roman")
        axs[1, 0].set_ylabel('Lift to Drag Ratio', fontsize=18, fontname="Times New Roman")
        a, t, p = plot_df.alpha.tolist(), plot_df.LtD.tolist(), plot_df.LtD_pred.tolist()
        axs[1, 0].plot(a, p, '--', lw=1, marker='o', markersize=2, label='Predicted')
        axs[1, 0].plot(a, t, lw=1, marker='o', markersize=2, label='True')
        axs[1, 0].legend()  # bbox_to_anchor=(0.15,1))

        if err:
            axs10 = axs[1, 0].twinx()
            axs10.set_ylabel('Error', fontsize=18, fontname="Times New Roman", rotation=270, labelpad=15)
            err = [np.abs(tt - pp) for tt, pp in zip(t, p)]
            axs10.bar(a, err, width=(max(a) - min(a)) / (2 * len(a)), alpha=0.1, label='Error')
            axs10.set_ylim(0, 1.5 * max(err))
            axs10.legend(bbox_to_anchor=(0.25, 1))

        aindx = aerofoils_df.loc[aerofoils_df.file == name].index[0]
        aerofoils.plot_profile(aerofoils_df, aindx, scatt=False, x_val=False, ax=axs[1, 1], prnt=False)

        plt.show()

    print('-Done. Predictions processed as chosen.')
    return