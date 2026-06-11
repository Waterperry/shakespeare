### Usage

To interact with the streamlit app: `uv run streamlit run scripts/run_app.py`

### Note on params / env vars

Make sure to set the env vars (or use `direnv` and a `.envrc`) for setting training params.
List of env vars to set:
 - TRAIN_DEVICE (cpu / mps / cuda:X)
 - TRAINING_RESUME_CHECKPOINT (1/0)
 - TRAINING_CHECKPOINT_PATH (optional, path to model_X.pt to restore training)
