from pathlib import Path

import keras
import pandas as pd


def save_model(_model):
    save_dir = Path('models') / _model.name
    save_dir.mkdir(parents=True, exist_ok=True)
    _model.save(str(save_dir / 'model.keras'))
    return


def _read_saved_csv(path):
    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df


def _prediction_label(output_df):
    files = output_df.file.drop_duplicates().tolist()
    reynolds = output_df.Re.drop_duplicates().tolist()

    if len(files) > 1:
        file_label = f'multi{len(files)}'
    else:
        file_label = str(files[0])

    if len(reynolds) > 1:
        re_label = f'multi{len(reynolds)}'
    else:
        re_label = str(reynolds[0])

    return f'{file_label}-{re_label}'


def _normalise_for_comparison(df):
    return df.reset_index(drop=True).astype('float64', errors='ignore').round(3)


def _results_match(left, right):
    left = _normalise_for_comparison(left)
    right = _normalise_for_comparison(right)
    return list(left.columns) == list(right.columns) and left.equals(right)


def _prediction_directory(predictions_dir, pred_on, output_df):
    pred_dir = predictions_dir / pred_on
    suffix = 0

    while pred_dir.exists():
        output_path = pred_dir / 'output.csv'
        if output_path.exists():
            output_df_check = _read_saved_csv(output_path)
            if _results_match(output_df, output_df_check):
                return pred_dir, True

        suffix += 1
        pred_dir = predictions_dir / f'{pred_on}({suffix})'

    return pred_dir, False


def save_results(_model, output_df, Pmetrics_df, fitHistory_df, ev_df):
    save_dir = Path('models') / _model.name
    predictions_dir = save_dir / 'predictions'
    training_dir = save_dir / 'training'
    evaluation_dir = save_dir / 'evaluation'

    predictions_dir.mkdir(parents=True, exist_ok=True)

    output_df = output_df.reset_index(drop=True)
    output_df = output_df.astype('float64', errors='ignore')

    pred_on = _prediction_label(output_df)
    pred_dir, already_saved = _prediction_directory(predictions_dir, pred_on, output_df)
    if already_saved:
        print(f'Results already saved in: {pred_dir}')
        return

    pred_dir.mkdir(parents=True)

    output_df.to_csv(pred_dir / 'output.csv', index=False)
    Pmetrics_df.to_csv(pred_dir / 'pred-metrics.csv', index=False)

    training_dir.mkdir(parents=True, exist_ok=True)
    fitHistory_df.to_csv(training_dir / 'fitHistory.csv', index=False)

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    ev_df.to_csv(evaluation_dir / 'evaluate.csv', index=False)

    print(f'Results saved to: {save_dir}')
    return


def load_model(directory):
    directory = Path(directory)
    model_path = directory / 'model.keras'
    legacy_path = directory / 'model'

    if model_path.exists():
        model = keras.models.load_model(str(model_path))
    elif legacy_path.is_dir():
        raise ValueError(
            f'Found legacy TensorFlow SavedModel directory at {legacy_path}. '
            'Keras 3 cannot load this format with keras.models.load_model(); '
            'retrain and save the model again to create model.keras.'
        )
    else:
        raise FileNotFoundError(
            f'No Keras model file found at {model_path}. Expected a Keras 3 .keras model file.'
        )

    return model


def load_results(directory, pred):
    directory = Path(directory)
    pred_dir = directory / 'predictions' / pred

    output_df = _read_saved_csv(pred_dir / 'output.csv')
    Pmetrics_df = _read_saved_csv(pred_dir / 'pred-metrics.csv')

    fitHistory_df = _read_saved_csv(directory / 'training' / 'fitHistory.csv')
    ev_df = _read_saved_csv(directory / 'evaluation' / 'evaluate.csv')

    return output_df, Pmetrics_df, fitHistory_df, ev_df


# TODO: Add and finish docstrings
# TODO: Add to ReadMe
