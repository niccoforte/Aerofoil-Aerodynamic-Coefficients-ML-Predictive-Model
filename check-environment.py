"""Offline one-epoch smoke test; temporary outputs, no changes to research data."""
import os
from pathlib import Path
import tempfile

os.environ['MPLBACKEND'] = 'Agg'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import keras
import numpy as np
from main import WorkflowConfig, run
from resources import saved

root = Path(__file__).resolve().parent
keras.utils.set_random_seed(42)
with tempfile.TemporaryDirectory(prefix='nf-beng-check-') as tmp:
    directory = Path(tmp)
    for name in ('dat', 'dat-saved'):
        (directory / name).symlink_to(root / name, target_is_directory=True)
    original_cwd = Path.cwd()
    try:
        os.chdir(directory)
        result = run(config=WorkflowConfig(n_train=2, n_test=2, epochs=1,
                                           neurons=[[8], [0]], batch=256, callbacks=False))
        assert np.isfinite(result['pred']).all()
        model = result['model']
        restored = saved.load_model(directory / 'models' / model.name)
        np.testing.assert_allclose(restored(model.test_in, training=False).numpy(),
                                   result['pred'], rtol=1e-5, atol=1e-6)
        assert list(directory.glob('models/*/predictions/*/pred-metrics.csv'))
        print('PASS: bundled data, one-epoch training, evaluation, prediction, metrics and .keras round trip.')
    finally:
        os.chdir(original_cwd)
