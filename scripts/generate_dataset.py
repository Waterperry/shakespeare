from pathlib import Path

from datasets import Dataset
from dvc import api
from fire import Fire
from rich.console import Console
from transformers import AutoTokenizer

from shakespeare.constants import INFO_STYLE


def main(input_path: str, output_path: str) -> None:
    console = Console()

    data: list[str] = []
    console.print(f"Reading raw data from {input_path} ...", style=INFO_STYLE)

    with api.open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(line)

    tokenizer = AutoTokenizer.from_pretrained(api.params_show()["generate"]["model_name"])
    tokenized = tokenizer(data)

    keys = ["input_ids", "token_type_ids", "attention_mask"]

    ds = Dataset.from_list(
        [
            {k: tokenized[k][i] for k in keys}
            for i in range(len(tokenized["input_ids"]))
        ]
    )

    ds.save_to_disk(output_path)


if __name__ == "__main__":
    Fire(main)
