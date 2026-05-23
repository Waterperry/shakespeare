import json

from pathlib import Path

import polars as pl

from datasets import Dataset
from dvc import api
from fire import Fire
from rich.console import Console
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def main(input_path: str, output_path: str) -> None:
    console = Console()
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(api.params_show()["generate"]["model_name"])

    ds = Dataset.load_from_disk(input_path)
    df: pl.DataFrame = ds.to_polars()  # pyright: ignore[reportAssignmentType]

    tokens_df: pl.DataFrame = (
        df
        .explode("input_ids")
        .select("input_ids")
        .unique()
        .sort("input_ids")
    )

    unique_tokens: set[int] = set(tokens_df.to_series().to_list())

    # sort the special tokens first so they get low IDs (just for neatness)
    special_token_ids: set[int] = {tokenizer.vocab[tok] for tok in tokenizer.all_special_tokens}
    unique_tokens.update(special_token_ids)

    console.print(f"Unique tokens: {len(unique_tokens):_}", style="green")

    curr_tok: int = 0
    remapper: dict[int, int] = {}
    used_new_tokens: set[int] = set()

    for tok_id in special_token_ids:
        remapper[tok_id] = curr_tok
        used_new_tokens.add(curr_tok)
        curr_tok += 1

    for tok in tokens_df["input_ids"].to_list():
        while curr_tok in used_new_tokens:
            curr_tok += 1

        if tok in remapper:
            print(f"Have remapped {tok} already")
            continue

        remapper[tok] = curr_tok
        used_new_tokens.add(curr_tok)
        curr_tok += 1

    df = df.with_columns(
        pl.col("input_ids").list.eval(pl.element().replace_strict(remapper)).alias("input_ids")
    )

    console.print(f"Writing dataset path to {output_path} ...")
    out_ds = Dataset.from_polars(df).train_test_split(test_size=0.25)
    out_ds.save_to_disk(output_path)

    with open(Path(f"{output_path}_metadata.json").resolve(), "w+") as f:
        json.dump({"vocab_size": len(unique_tokens)}, f)

    vocab: dict[str, int] = {
        str(tokenizer.convert_ids_to_tokens(old_tok_id)): new_tok_id
        for old_tok_id, new_tok_id in remapper.items()
    }

    for idx in range(len(unique_tokens)):
        assert idx in vocab.values(), f"{idx} not in new"

    tokenizer_model = WordLevel(vocab=vocab, unk_token=tokenizer.unk_token)
    fast_tokenizer = Tokenizer(tokenizer_model)

    # just carry these fields over so we keep the pre-tokenizer/normalization scheme
    fast_tokenizer.pre_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    fast_tokenizer.normalizer   = tokenizer.backend_tokenizer.normalizer

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast_tokenizer,
        unk_token=tokenizer.unk_token,
        pad_token=tokenizer.pad_token,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        mask_token=tokenizer.mask_token,
    )

    hf_tokenizer.save_pretrained("outs/tokenizer")


if __name__ == "__main__":
    Fire(main)
