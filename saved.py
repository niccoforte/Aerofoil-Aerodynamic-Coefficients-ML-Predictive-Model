import pandas as pd
import os
from tensorflow import keras


def save_model(_model):
    save_dir = f'models/{_model.name}'
    _model.save(f'{save_dir}/model/')
    return


def save_results(_model, output_df, Pmetrics_df, fitHistory_df, ev_df):
    save_dir = f'models/{_model.name}'

    if not os.path.exists(f'{save_dir}/predictions/'):
        os.makedirs(f'{save_dir}/predictions/')

    output_df = output_df.reset_index(drop=True)
    output_df = output_df.astype('float64', errors='ignore')

    if len(set(output_df.file.tolist())) > 1:
        if len(set(output_df.Re.tolist())) > 1:
            pred_on = f'multi{len(set(output_df.file.tolist()))}-multi{len(set(output_df.Re.tolist()))}'
        else:
            pred_on = f'multi{len(set(output_df.file.tolist()))}-{output_df.Re.tolist()[0]}'
    else:
        if len(set(output_df.Re.tolist())) > 1:
            pred_on = f'{output_df.file.tolist()[0]}-multi{len(set(output_df.Re.tolist()))}'
        else:
            pred_on = f'{output_df.file.tolist()[0]}-{output_df.Re.tolist()[0]}'

    pred_dup = len([name for name in os.scandir(f'{save_dir}/predictions/') if pred_on in str(name)])
    if pred_dup >= 1:
        output_df_check = pd.read_csv(f'{save_dir}/predictions/{pred_on}/output.csv')
        output_df_check = output_df_check.drop(columns=['Unnamed: 0'])
        if (output_df.round(3) == output_df_check.round(3)).all().all():
            print(f'Results already saved in: {save_dir}/predictions/{pred_on}')
            return
        pred_dir = f'{save_dir}/predictions/{pred_on}({pred_dup})'
    else:
        pred_dir = f'{save_dir}/predictions/{pred_on}'

    os.makedirs(f'{pred_dir}')

    output_df.to_csv(f'{pred_dir}/output.csv')
    Pmetrics_df.to_csv(f'{pred_dir}/pred-metrics.csv')

    if not os.path.exists(f'{save_dir}/training/'):
        os.makedirs(f'{save_dir}/training/')
    fitHistory_df.to_csv(f'{save_dir}/training/fitHistory.csv')

    if not os.path.exists(f'{save_dir}/evaluation/'):
        os.makedirs(f'{save_dir}/evaluation/')
    ev_df.to_csv(f'{save_dir}/evaluation/evaluate.csv')

    print(f'Results saved to: {save_dir}')
    return


def load_model(directory):
    model = keras.models.load_model(f'{directory}/model')

    return model


def load_results(directory, pred):
    output_df = pd.read_csv(f'{directory}/predictions/{pred}/output.csv')
    Pmetrics_df = pd.read_csv(f'{directory}/predictions/{pred}/pred-metrics.csv')

    fitHistory_df = pd.read_csv(f'{directory}/training/fitHistory.csv')
    ev_df = pd.read_csv(f'{directory}/evaluation/evaluate.csv')

    return output_df, Pmetrics_df, fitHistory_df, ev_df


# TODO: Add and finish docstrings
# TODO: Add to ReadMe
