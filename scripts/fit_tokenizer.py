from datasets import Dataset, disable_caching
from dvc import api
from fire import Fire
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


def main(input_path: str, output_path: str) -> None:
    ds = Dataset.load_from_disk(input_path)
    params = api.params_show()["fit_tokenizer"]

    special_tokens: dict[str, str] = dict(
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer = Tokenizer(BPE(unk_token=special_tokens["unk_token"]))
    tokenizer.normalizer = normalizers.Sequence([NFKC(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=params["vocab_size"],
        special_tokens=sorted(special_tokens.values()),
        show_progress=True,
    )
    tokenizer.train_from_iterator(ds["sentence"], trainer)

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, **special_tokens)
    # set this now as opposed to earlier as we need to know the EOS token ID
    fast_tokenizer._tokenizer.post_processor = TemplateProcessing(  # pyright: ignore[reportOptionalMemberAccess]
        "<s> $A </s>",
        special_tokens=[
            ("<s>", fast_tokenizer.bos_token_id),
            ("</s>", fast_tokenizer.eos_token_id),
        ],
    )
    fast_tokenizer.save_pretrained("outs/tokenizer")

    def encode(batch):
        return {"input_ids": fast_tokenizer.encode(batch["sentence"])}

    disable_caching()  # tokenizer will be recognised as unique every time, causing reprocessing to occur (due to cache miss)
    ds = ds.map(encode, batched=True, batch_size=1_024)
    
    ds.train_test_split(0.2).save_to_disk(output_path)

if __name__ == "__main__":
    Fire(main)
