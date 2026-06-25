import argparse
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

from resources import aerofoils
from resources import cases
from resources import data
from resources import nnetwork
from resources import saved


@dataclass(frozen=True)
class WorkflowConfig:
    reset: bool = False
    ren: bool = False
    points: int = 51
    n_train: int = 250
    n_test: int = 50
    neurons: list[list[int]] = field(default_factory=lambda: [[512, 256, 128, 64, 32, 16], [1, 2, 3, 4]])
    activations: list[str] = field(default_factory=lambda: ['sigmoid'])
    epochs: int = 5
    batch: int = 256
    lr: float = 0.001
    verbose: int = 0
    callbacks: bool = True
    prediction_name: str = 'n0012'
    prediction_re: float = 1000000


def set_up(_reset=False, _ren=False):
    """Set-up options for running main.py.

    Parameters
    ----------
    _reset : bool, default False
        If True checks for missing files to download from online databases to 'dat/' directories, creates case DataFrames
        from scratch, and saves them by overwriting files in 'dat-saved/' directory. Otherwise uses files already
        present in these directories.
    _ren : bool, default False
        If True runs the script for the profile coordinates and coefficients downloaded from the study conducted by
        the Universite de Rennes. Note NN training and testing will be done using this data too. Otherwise the script
        uses profile coordinates and coefficients from UIUC and Airfoil Tools.
    """

    return _reset, _ren


def run(reset=None, ren=None, config=None):
    """Run the full data preparation, model training, prediction, plotting, and saving workflow."""

    config = WorkflowConfig() if config is None else config
    if reset is not None:
        config = replace(config, reset=reset)
    if ren is not None:
        config = replace(config, ren=ren)

    reset, ren = set_up(_reset=config.reset, _ren=config.ren)

    # Download aerofoil .dat files to 'dat/aerofoil-dat' directory and case .csv files to 'dat/case-dat' directory.
    if reset is True:
        aerofoils.get_UIUC_foils(directory='dat/aerofoil-dat')
        aerofoils.get_AFT_foils(directory='dat/aerofoil-dat')
        cases.get_AFT_cases(directory='dat/case-dat')

        if ren is True:
            aerofoils.get_RENNES_foils(directory='dat/rennes-dat/aerofoil-dat')
            cases.get_RENNES_cases(directory='dat/rennes-dat/case-dat')

    # Create dictionary of Profile objects and Aerofoils DataFrame.
    profiles, aerofoils_df = aerofoils.create_profiles(
        directory='dat/aerofoil-dat',
        points=config.points,
        prnt=False,
    )

    if ren is True:
        ren_profiles, ren_aerofoils_df = aerofoils.create_profiles(
            directory='dat/rennes-dat/aerofoil-dat',
            points=config.points,
            prnt=False,
        )

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
        exp_cases_df = cases.df_from_csv(file='dat-saved/exp-cases-df.csv')

        if ren is True:
            ren_cases_df = cases.df_from_csv(file='dat-saved/ren-cases-df.csv')

    # Merge aerofoils and cases dataframes.
    data_df = data.merge(aerofoils_df, cases_df)
    exp_data_df = data.merge(aerofoils_df, exp_cases_df)

    if ren is True:
        ren_data_df = data.merge(ren_aerofoils_df, ren_cases_df)

    # Prepare & get NN inputs & outputs.
    train_df, test_df, sample_weights = data.get_data(
        df1=data_df,
        df2=None,
        nTrain=config.n_train,
        nTest=config.n_test,
    )
    train_in, train_out, test_in, test_out = data.prep_data(data=[train_df, test_df])
    dat = [train_in, train_out, test_in, test_out]
    model_test_df = test_df
    plot_aerofoils_df = aerofoils_df

    if ren is True:
        ren_train_df, ren_test_df, sample_weights = data.get_data(
            df1=ren_data_df,
            df2=exp_data_df,
            nTrain=config.n_train,
            nTest=config.n_test,
            p_foil=config.prediction_name,
            p_Re=config.prediction_re,
        )
        ren_train_in, ren_train_out, ren_test_in, ren_test_out = data.prep_data(data=[ren_train_df, ren_test_df])
        dat = [ren_train_in, ren_train_out, ren_test_in, ren_test_out]
        model_test_df = ren_test_df
        # Rennes mode tests against experimental data, which is merged with the standard aerofoil profiles.
        plot_aerofoils_df = aerofoils_df

    # Build model(s), train using training data, & predict using testing data.
    models = nnetwork.run_Model(
        data=dat,
        neurons=config.neurons,
        activation=config.activations,
        weights=sample_weights,
        test_df=model_test_df,
        EPOCHS=config.epochs,
        BATCH=config.batch,
        lr=config.lr,
        verbose=config.verbose,
        callbacks=config.callbacks,
    )
    model = list(models.values())[0]
    _model = model.model
    pred, Pmetrics_df, output_df = model.pred, model.Pmetrics_df, model.output_df
    fitHistory_df, ev_df = model.fitHistory_df, model.ev_df

    # Load model and results
    # _model = saved.load_model(directory='models/MLP-Sigmoid')
    # output_df, Pmetrics_df, fitHistory_df, ev_df = saved.load_results(
    #     directory='models/MLP-Sigmoid',
    #     pred='n0012-700000.0',
    # )

    # Predict using testing data.
    # pred, Pmetrics_df, output_df = nnetwork.model_predict(
    #     _model=_model,
    #     test_in=test_in,
    #     test_out=test_out,
    #     test_df=test_df,
    # )
    # pred, Pmetrics_df, output_df = nnetwork.model_predict(
    #     _model=_model,
    #     test_in=ren_test_in,
    #     test_out=ren_test_out,
    #     test_df=ren_test_df,
    # )

    # Training and prediction metrics.
    nnetwork.train_metrics(
        model=model,
        models=None,  # models,
        mets=['loss', 'ACC'],
        prnt=True,
        plot=True,
    )
    nnetwork.pred_metrics(
        Pmetrics_df=Pmetrics_df,
        models=None,  # models,
        prnt=True,
        plot=True,
    )

    # Predictions.
    nnetwork.predictions(
        output_df=output_df,
        name=config.prediction_name,
        re=config.prediction_re,
        aerofoils_df=plot_aerofoils_df,
        plot=True,
        err=True,
    )

    # Save model and results.
    saved.save_model(_model=_model)
    saved.save_results(
        _model=_model,
        output_df=output_df,
        Pmetrics_df=Pmetrics_df,
        fitHistory_df=fitHistory_df,
        ev_df=ev_df,
    )

    return {
        'profiles': profiles,
        'aerofoils_df': aerofoils_df,
        'cases_df': cases_df,
        'exp_cases_df': exp_cases_df,
        'data_df': data_df,
        'exp_data_df': exp_data_df,
        'train_df': train_df,
        'test_df': test_df,
        'sample_weights': sample_weights,
        'models': models,
        'model': model,
        '_model': _model,
        'pred': pred,
        'Pmetrics_df': Pmetrics_df,
        'output_df': output_df,
        'fitHistory_df': fitHistory_df,
        'ev_df': ev_df,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run the aerofoil coefficient ML prediction workflow.')
    parser.add_argument('--reset', action='store_true', help='Download/rebuild raw and saved case data before running.')
    parser.add_argument('--ren', action='store_true', help='Use Rennes profile and coefficient data for training/testing.')
    parser.add_argument('--epochs', type=int, default=WorkflowConfig.epochs, help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=WorkflowConfig.batch, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=WorkflowConfig.lr, help='Adam optimizer learning rate.')
    parser.add_argument('--n-train', type=int, default=WorkflowConfig.n_train, help='Number of training aerofoils.')
    parser.add_argument('--n-test', type=int, default=WorkflowConfig.n_test, help='Number of testing aerofoils.')
    parser.add_argument('--prediction-name', default=WorkflowConfig.prediction_name, help='Aerofoil file to plot.')
    parser.add_argument('--prediction-re', type=float, default=WorkflowConfig.prediction_re, help='Reynolds number to plot.')
    parser.add_argument('--verbose', type=int, default=WorkflowConfig.verbose, help='Keras verbosity level.')
    parser.add_argument('--no-callbacks', action='store_true', help='Disable training callbacks.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = WorkflowConfig(
        reset=args.reset,
        ren=args.ren,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        n_train=args.n_train,
        n_test=args.n_test,
        prediction_name=args.prediction_name,
        prediction_re=args.prediction_re,
        verbose=args.verbose,
        callbacks=not args.no_callbacks,
    )
    run(config=config)


if __name__ == '__main__':
    main()


#########################################################################################################
###############################################  EXTRA  #################################################
#########################################################################################################

# ----- Print DF
# pd.set_option('display.max_columns', None)
# print(data_df)

# ----- Aerofoil Plots
# aindx = aerofoils_df.loc[aerofoils_df.file == 'goe10k'].index[0]
# aerofoils.plot_profile(aerofoils_df, aindx, scatt=True, x_val=0.3, pltfig=1, prnt=True)

#########################################################################################################
#########################################################################################################
