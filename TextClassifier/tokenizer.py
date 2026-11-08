import csv
from pathlib import Path

x = int(input("Enter the number of reviews to load: "))
texts = []
labels = []

texts = texts[:x]
labels = labels[:x]

data_path = Path(__file__).parent / "processed.csv"

with data_path.open("r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        texts.append(row["selected_text"].strip())
        labels.append(row["sentiment"].strip())

print(f"Loaded {len(texts)} reviews and {len(labels)} labels")
print(texts[0])
print(labels[0])

def tokenize(text):
    text = text.strip().lower()
    words = text.split()
    return words

def build_vocab(texts):
    vocab = set()

    for text in texts:
        words = tokenize(text)
        vocab.update(words)

    sorted_vocab = sorted(vocab)
    word_to_index = {
        word: index
        for index, word in enumerate(sorted_vocab)
    }
    return word_to_index

def vectorize(text, word_to_index):
    vector = [0] * len(word_to_index)

    for word in tokenize(text):
        if word in word_to_index:
            index = word_to_index[word]
            vector[index] += 1

    return vector

vector = vectorize(texts[0], word_to_index)
print("Vector length:", len(vector))
print("Words present:", sum(count > 0 for count in vector))