import pandas as pd


data = pd.read_csv("Data/harry_potter_books.csv")

book_texts = data.groupby("book")["text"].apply(lambda x: " ".join(x)).reset_index()

print(book_texts.head())
# split text to sentences
book_texts["sentences"] = book_texts["text"].apply(lambda x: x.split(". "))
print(book_texts.head())
