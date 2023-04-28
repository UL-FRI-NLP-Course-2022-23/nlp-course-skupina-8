import os
from typing import Union, List
from datasets.arrow_dataset import Dataset
import pandas as pd
""" from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse """


def remove_short_sentences_by_chars(original, translated, min_length=30):
    new1, new2 = list(), list()
    with open("removed_sentences.txt", "a") as f:
        for s1, s2 in zip(original, translated):
            if len(s1) >= min_length and len(s2) >= min_length:
                new1.append(s1)
                new2.append(s2)
            else:
                f.write("Removed sentence because it was too short: " + s1 + " " + s2 + "\n")
    return new1, new2    

    
def remove_sentences_with_too_many_numbers(original, translated, max_numbers=20):
    new1, new2 = list(), list()
    with open("removed_sentences.txt", "a") as f:
        for s1, s2 in zip(original, translated):
            if sum([s1.count(str(n)) for n in range(10)]) <= max_numbers and sum([s2.count(str(n)) for n in range(10)]) <= max_numbers:
                new1.append(s1)
                new2.append(s2)
    return new1, new2


def remove_sentences_with_too_many_special_characters(original, translated, max_special_characters=20):
    new1, new2 = list(), list()
    with open("removed_sentences.txt", "a") as f:
        for s1, s2 in zip(original, translated):
            if sum([s1.count(c) for c in "!@#$%^&*()_+-=[]{};':\"\\|,.<>/?"]) <= max_special_characters and sum([s2.count(c) for c in "!@#$%^&*()_+-=[]{};':\"\\|,.<>/?"]) <= max_special_characters:
                new1.append(s1)
                new2.append(s2)
    return new1, new2

def remove_special_characters(original, translated):
    new1, new2 = list(), list()
    with open("removed_sentences.txt", "a") as f:
        for s1, s2 in zip(original, translated):
            if not s1[0].isalpha() and not s1[-1].isalnum():
                print(s1[0], s1[-1], s1[0].isalpha(), s1[-1].isalnum())
                f.write("Removed sentence because it is not started with alphanumeric character and not ended with a special character " + s1 + " " + s2 + "\n")
                continue
            if not s2[0].isalpha() and not s2[-1].isalnum():
                f.write("Removed sentence because it is not started with alphanumeric character and not ended with a special character " + s1 + " " + s2 + "\n")
                continue
            new1.append(s1)
            new2.append(s2)
    return new1, new2


def read(path: str = "../../data/euparl600k_ensl", preprocess: Union[list[callable], None] = None) -> Dataset:
    original, translated = list(), list()
    with open(os.path.join(path, "europarl-orig-sl-all.out")) as file:
        while True:
           l = file.readline()
           if not l: break
           original.append(l.strip("\n"))
    with open(os.path.join(path, "europarl-tran-all.out")) as file:
        while True:
           l = file.readline()
           if not l: break
           translated.append(l.strip("\n"))
    if preprocess:
        for p in preprocess:
            original, translated = p(original, translated)
    df = pd.DataFrame()
    df["original"] = original
    df["translated"] = translated
    df = df.sample(frac=1, random_state=42)
    return Dataset.from_pandas(df)


def euparl(min_length: int = 30, max_numbers: int = 20, max_special_characters: int = 20, path: str = "../../data/euparl600k_ensl") -> Dataset:
    return read(path, [lambda x, y: remove_sentences_with_too_many_numbers(x, y, max_numbers), lambda x, y: remove_sentences_with_too_many_special_characters(x, y, max_special_characters), lambda x, y: remove_short_sentences_by_chars(x, y, min_length), lambda x, y: remove_special_characters(x,y)])

