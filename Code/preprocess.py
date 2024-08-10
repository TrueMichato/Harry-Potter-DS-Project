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

# def replace_pronouns_with_names(chapter_text: str, character_names) -> str:
#     doc = nlp(chapter_text)

#     # Dictionary to store replacements
#     replacements = {}

#     # Iterate over coreference chains
#     for chain in doc._.coref_chains:
#         # Find the main entity that is a character's name
#         for mention in chain:
#             mention_text = doc[mention.start:mention.end].text
#             if mention_text in character_names:
#                 character_name = mention_text
#                 break
#         else:
#             # If no character name is found in the chain, skip to the next one
#             continue

#         # Replace pronouns in the chain with the character's name
#         for mention in chain:
#             mention_text = doc[mention.start:mention.end].text
#             if mention_text.lower() in ["he", "she", "him", "her", "his", "hers", "they", "them"]:
#                 replacements[(mention.start, mention.end)] = character_name

#     # Replace text in the original document
#     new_text = list(doc)
#     for (start, end), name in replacements.items():
#         new_text[start:end] = [name]

#     return " ".join([token.text for token in new_text])


def resolve_coreferences(chapter_text) -> str:
    doc = nlp(chapter_text)

    # Get the coreference clusters
    clusters = doc._.coref_clusters

    # Create a mapping from pronouns to resolved mentions
    resolved_text = chapter_text
    for cluster in clusters:
        main_entity = cluster.main.text
        for mention in cluster.mentions:
            if mention != cluster.main:
                resolved_text = resolved_text.replace(mention.text, main_entity)

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

def solve_couple(text: str, names_dict: dict) -> str:
    regex = re.compile(r"Mr\. and Mrs\. (\w+)")
    match = regex.search(text)
    if match:
        couple_name = match.group(1)
        name1 = "Mr. " + couple_name
        name2 = "Mrs. " + couple_name
        if name1 in names_dict and name2 in names_dict:
            name1_id = "id_" + str(names_dict[name1][0])
            name2_id = "id_" + str(names_dict[name2][0])
            new_text = text.replace("Mr. and Mrs. " + couple_name, name1_id + " and " + name2_id)
        else:
            new_text = text.replace("Mr. and Mrs. " + couple_name, name1 + " and " + name2)
        return new_text
    return text

def solve_mr_mrs(text: str, names_dict: dict, mr: bool=True) -> str:
    regex = re.compile(r"Mr\. (\w+)") if mr else re.compile(r"Mrs\. (\w+)")
    match = regex.search(text)
    if match:
        title = "Mr." if mr else "Mrs."
        name = match.group(1)
        name = title + name
        if name in names_dict:
            name_id = "id_" + str(names_dict[name][0])
            new_text = text.replace(title + name, name_id)
        else:
            new_text = text
        return new_text
    return text


# doc = nlp("213 and 214, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense. Harry Potter was the director of a firm called Grunnings, which made drills. He was a big, beefy man with hardly any neck, although he did have a very large mustache.")
# doc._.coref_chains.print()
# print(doc[0])
# print(doc[26])
# print(doc._.coref_chains.resolve(doc[0]))
# print(doc._.coref_chains.resolve(doc[26]))
# print(doc._.coref_chains.resolve())

df_characters = pd.read_csv("Data/character_names.csv")
dict_names_id = create_dict_names_id(df_characters)
print(dict_names_id)
dict_id_names = reverse_dict(dict_names_id)
print(dict_id_names)
# character_names = set([name for names in dict_names_id.values() for name in names])
# df_text = pd.read_csv("Data/harry_potter_books_preprocessed.csv")
# df_text["text"] = df_text["text"].apply(resolve_coreferences)
# )
# df_text["sentences"] = df_text["text"].apply(lambda x: tk.sent_tokenize(x))
# print(df_text["text"][0])
# print(df_text["sentences"][0])
