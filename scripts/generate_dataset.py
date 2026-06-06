import re

from unicodedata import normalize

from datasets import Dataset
from dvc import api
from fire import Fire
from rich.console import Console

from shakespeare.constants import INFO_STYLE


CHAPTER_LINE_MATCHER = re.compile(r"^\d+$")  # \n123\n
STAGE_DIRECTION_MATCHER = re.compile(r"^\[_.*_\]$")  # \n[_STAGE DIRECTION_]\n
ACT_MATCHER = re.compile(r"^ACT [IVXCM]+$")
CHARACTER_SPEECH_INDICATOR = re.compile(r"^[A-Z]{2,}\.$")  # \nHAMLET.\n<Hamlet speech block>\n
PART_MATCHER = re.compile(r"^\d+$")
SENTENCE_ENDING_MATCHER: re.Pattern[str] = re.compile(r"[\.\?\!\:\;]")

char_remappings: dict[str, str] = {
    'À': "A",
    'Æ': "AE",
    'Ç': "C",
    'É': "E",
    'à': "a",
    'â': "a",
    'æ': "ae",
    'ç': "c",
    'è': "e",
    'é': "e",
    'ê': "e",
    'ë': "e",
    'î': "i",
    'œ': "oe",
    '—': "-",
    '‘': "'",
    '’': "'",
    '“': "\"",
    '”': "\"",
    '•': "-",
    '\ufeff': " ",
}

def split_line_to_parts(sentence: str) -> list[str]:
    global SENTENCE_ENDING_MATCHER
    parts: list[str] = []
    prev_end: int = 0
    matches = list(SENTENCE_ENDING_MATCHER.finditer(sentence))

    if len(matches) == 0:
        return [sentence]

    for match in matches:
        start, end = match.span()
        part = sentence[prev_end:start+1]
        prev_end = end
        parts.append(part)
        if end == len(sentence):
            parts.append("")

    maybe_last_part = sentence[prev_end:]
    if maybe_last_part:
        parts.append(maybe_last_part)
    return [p.strip() for p in parts]

def normalize_line(line: str) -> str:
    line = line.strip()
    line = normalize("NFKC", line)
    for src, tgt in char_remappings.items():  # TODO: make this O(n) not O(n * m)
        line = line.replace(src, tgt)
    line = line.encode("ascii", "strict").decode("ascii")
    return line


def main(input_path: str, output_path: str) -> None:
    console = Console()

    all_sentences: list[str] = []
    console.print(f"Reading raw data from {input_path} ...", style=INFO_STYLE)

    with api.open(input_path) as f:
        current_sentence: list[str] = []
        storing: bool = False
        reading_titles: bool = False
        titles: set[str] = set()

        for line in f:
            line = normalize_line(line)
            if line == "Contents":
                reading_titles = True
                storing = True
                continue
            if not (storing and line): continue
            if ACT_MATCHER.search(line): continue
            if reading_titles:
                if line in titles:
                    reading_titles = False
                else:
                    titles.add(line)
                continue

            if PART_MATCHER.match(line): continue

            # this needs to override the below condition (as we want to preface speech with `line` here, but not in general)
            if CHARACTER_SPEECH_INDICATOR.match(line):
                all_sentences.append(" ".join(current_sentence))
                current_sentence = [line]
                continue

            # handle mid-sentence punctuation
            line_parts = split_line_to_parts(line)
            if len(line_parts) == 1:
                current_sentence.append(line)
                continue
            for part in line_parts:
                if part:
                    current_sentence.append(part)

                all_sentences.append(" ".join(current_sentence))
                current_sentence = []

    ds = Dataset.from_list([{"sentence": s} for s in all_sentences if s])
    ds.save_to_disk(output_path)


if __name__ == "__main__":
    Fire(main)
