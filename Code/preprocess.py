import pandas as pd
import nltk.tokenize as tk


data = pd.read_csv("Data/harry_potter_books.csv")

book_texts = data.groupby(["book", "chapter"])["text"].apply(lambda x: " ".join(x)).reset_index()
book_texts["chapter"] = book_texts["chapter"].apply(lambda x: int(x.split("-")[1]))
book_texts = book_texts.sort_values(by=["book", "chapter"])
book_texts["chapter name"] = book_texts["text"].apply(lambda x: x.split("  ")[0])
book_texts["text"] = book_texts.apply(lambda row: row['text'].replace(row['chapter name'], '').replace('  ', ' '), axis=1)

# split text to sentences
book_texts["sentences"] = book_texts["text"].apply(lambda x: tk.sent_tokenize(x))
print(book_texts.head(15))

book_texts.to_csv("Data/harry_potter_books_preprocessed.csv", index=False)