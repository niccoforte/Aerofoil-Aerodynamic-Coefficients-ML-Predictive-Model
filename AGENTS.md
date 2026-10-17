# Project guidance

- Read `README.md` and `requirements.txt` before changing setup or workflow behaviour.
- On this Mac use Miniforge's `nf-beng` environment (Python 3.12), not `base`, `nf-phd` or the retired Anaconda installation.
- `setup-macOS.sh` creates/refreshes the environment; running it mutates installed packages and requires an installation request.
- Keep dependency declarations in `requirements.txt`. Intel macOS requires the TensorFlow 2.16.2 platform pin; current code uses Keras 3.
- `python check-environment.py` is an offline one-epoch smoke test with temporary outputs and a `.keras` prediction round trip. It is not a scientific accuracy benchmark.
- Preserve bundled `dat/`, `dat-saved/` and ignored historical `results/`. Do not run `--reset` or overwrite archived models without explicit authorisation.
- The optional historical SavedModel uses TensorFlow's signature API for inference. Do not rename it to `.keras` or silently convert it.
- Keep setup instructions current and changes minimal. Do not commit generated models, environment inventories or test-output bulk.
- The user authorizes committing and pushing repository work by default. Pull before publishing, validate the intended changes, push to all configured destinations, and verify the remote tip and clean working tree before handoff. Report any blocker instead of leaving completed work unsynchronized.
