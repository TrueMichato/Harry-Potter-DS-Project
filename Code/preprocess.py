import pandas as pd
import nltk.tokenize as tk
import re


data = pd.read_csv("Data/harry_potter_books.csv")

book_texts = data.groupby(["book", "chapter"])["text"].apply(lambda x: " ".join(x)).reset_index()
book_texts["chapter"] = book_texts["chapter"].apply(lambda x: int(x.split("-")[1]))
book_texts = book_texts.sort_values(by=["book", "chapter"])
book_texts = book_texts.reset_index(drop=True)
# get chapter name using a regex
book_texts["chapter name"] = book_texts["text"].str.extract(r'^([A-Z ]+)\s\s')
# make chapter name a string
book_texts["chapter name"] = book_texts["chapter name"].apply(lambda x: str(x))
book_texts["text"] = book_texts.apply(lambda row: row['text'].replace(row['chapter name'], '').replace('  ', ' '), axis=1)


# split text to sentences
book_texts["sentences"] = book_texts["text"].apply(lambda x: tk.sent_tokenize(x))
print(book_texts.head(15))

book_texts.to_csv("Data/harry_potter_books_preprocessed.csv", index=False)

# create a list of all sentences, without book or chapter affiliation
all_sentences = [sentence for sentences in book_texts["sentences"] for sentence in sentences]
all_sentences_df = pd.DataFrame(all_sentences, columns=["sentence"])

all_sentences_df.to_csv("Data/harry_potter_sentences.csv", index=False)
