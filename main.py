from transformers import AutoModel, AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

    with open("./paragraphs.txt") as f:
        data = f.read().splitlines()

    tokenized = tokenizer(data, return_tensors="pt", truncate=False)

    print(len(tokenized))
    print(len(tokenized["input_ids"]))



if __name__ == "__main__":
    main()
