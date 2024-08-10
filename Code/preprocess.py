import pandas as pd
import nltk.tokenize as tk
import re


# data = pd.read_csv("Data/harry_potter_books.csv")

# book_texts = data.groupby(["book", "chapter"])["text"].apply(lambda x: " ".join(x)).reset_index()
# book_texts["chapter"] = book_texts["chapter"].apply(lambda x: int(x.split("-")[1]))
# book_texts = book_texts.sort_values(by=["book", "chapter"])
# book_texts = book_texts.reset_index(drop=True)
# # get chapter name using a regex
# book_texts["chapter name"] = book_texts["text"].str.extract(r'^([A-Z ]+)\s\s')
# # make chapter name a string
# book_texts["chapter name"] = book_texts["chapter name"].apply(lambda x: str(x))
# book_texts["text"] = book_texts.apply(lambda row: row['text'].replace(row['chapter name'], '').replace('  ', ' '), axis=1)


# # split text to sentences
# book_texts["sentences"] = book_texts["text"].apply(lambda x: tk.sent_tokenize(x))
# print(book_texts.head(15))

# book_texts.to_csv("Data/harry_potter_books_preprocessed.csv", index=False)

# # create a list of all sentences, without book or chapter affiliation
# all_sentences = [sentence for sentences in book_texts["sentences"] for sentence in sentences]
# all_sentences_df = pd.DataFrame(all_sentences, columns=["sentence"])

# all_sentences_df.to_csv("Data/harry_potter_sentences.csv", index=False)

import spacy
import coreferee

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("coreferee")

def resolve_coreferences(chapter_text) -> str:
    doc = nlp(chapter_text)
    chains = doc._.coref_chains

    # Create a mapping from pronouns to resolved mentions
    resolved_text = chapter_text
    for chain in chains:
        for mention in chain:
            fix = doc._.coref_chains.resolve(mention)
            resolved_text = resolved_text.replace(mention.text, fix)

    return resolved_text


def create_dict_names_id(df_characters) -> dict:
    dict_id_names = {}
    for _, row in df_characters.iterrows():
        all_names = []
        all_names.append(row["Name"])
        if pd.notna(row["Other Names"]):
            all_names.extend(row["Other Names"].split(", "))
        dict_id_names[row["Id"]] = all_names
    return dict_id_names

def reverse_dict(dict_names_id) -> dict:
    dict_id_names = {}
    for id_, names in dict_names_id.items():
        for name in names:
            if name in dict_id_names:
                dict_id_names[name].append(id_)
            else:
                dict_id_names[name] = [id_]
    return dict_id_names

def solve_couple(text: str) -> str:
    regex = re.compile(r"Mr\. and Mrs\. (\w+)")
    match = regex.search(text)
    if match:
        couple_name = match.group(1)
        name1 = "Mr._" + couple_name
        name2 = "Mrs._" + couple_name
        new_text = text.replace("Mr. and Mrs. " + couple_name, name1 + " and " + name2)
        return new_text
    return text

def solve_mr_mrs(text: str, mr: bool=True) -> str:
    regex = re.compile(r"Mr\. (\w+)") if mr else re.compile(r"Mrs\. (\w+)")
    match = regex.search(text)
    if match:
        title = "Mr." if mr else "Mrs."
        name = match.group(1)
        name = title + name
        new_text = text.replace(title + " " + name, title + "_" + name)
        return new_text
    return text

def solve_mrs(text: str) -> str:
    return solve_mr_mrs(text, False)

def solve_mrs_mr_issue(text: str) -> str:
    text = solve_mr_mrs(text)
    return solve_mrs(text)

def solve_long_name_issue(text: str, names: list) -> str:
    for name in names:
        if len(name.split(" ")) > 2:
            new_name = "__".join(name.split(" "))
            text = re.sub(rf"\b{name}\b", new_name, text)
    return text


doc = nlp("Mr.__Dursley and Mrs.__Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Mr.__Dursley was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache.")
doc._.coref_chains.print()
for chain in doc._.coref_chains:
    for mention in chain:
        if len(mention) > 1:
            for m in mention:
                fix = doc._.coref_chains.resolve(doc[m])
                print(fix)
        else:
            fix = doc._.coref_chains.resolve(doc[mention.token_indexes[0]])
            print(fix)
        # fix = doc._.coref_chains.resolve(doc[mention])
        # print(f"{mention.text} -> {fix}")
print(doc[0])
print(doc[26])
print(doc._.coref_chains.resolve(doc[0]))
print(doc._.coref_chains.resolve(doc[26]))
print(doc._.coref_chains.resolve())

# df_characters = pd.read_csv("Data/character_names.csv")
# dict_names_id = create_dict_names_id(df_characters)
# character_names = set([name for names in dict_names_id.values() for name in names])
# df_text = pd.read_csv("Data/harry_potter_books_preprocessed.csv")

# df_text["text"] = df_text["text"].apply(lambda x: solve_couple(x))
# df_text["text"] = df_text["text"].apply(lambda x: solve_mrs_mr_issue(x))
# df_text["text"] = df_text["text"].apply(lambda x: solve_long_name_issue(x, character_names))


# df_text["sentences"] = df_text["text"].apply(lambda x: tk.sent_tokenize(x))
# print(df_text["text"][0])
# print(df_text["sentences"][0])
