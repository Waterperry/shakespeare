### Note on params / env vars

Make sure to set the env vars (or use `direnv` and a `.envrc`) for setting training params.
List of env vars to set:
 - TRAIN_DEVICE (cpu / mps / cuda:X)
 - TRAINING_RESUME_CHECKPOINT (1/0)
 - TRAINING_CHECKPOINT_PATH (optional, path to model_X.pt to restore training)

### Note on tokenizer

The `[SEP]` token has been repurposed as the `[EOS]` token (since the DebertaV2Tokenizer is a BERT tokenizer and does not have an `[EOS]` token).
The tokenizer produced in `outs/tokenizer` does NOT automatically add the `[SEP]` token for generation purposes.
