# import pandas as pd
# import nltk.tokenize as tk
# import re


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

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('coreferee')

def replace_pronouns_with_names(chapter_text, character_names):
    doc = nlp(chapter_text)
    
    # Dictionary to store replacements
    replacements = {}

    # Iterate over coreference chains
    for chain in doc._.coref_chains:
        # Find the main entity that is a character's name
        for mention in chain:
            mention_text = doc[mention.start:mention.end].text
            if mention_text in character_names:
                character_name = mention_text
                break
        else:
            # If no character name is found in the chain, skip to the next one
            continue
        
        # Replace pronouns in the chain with the character's name
        for mention in chain:
            mention_text = doc[mention.start:mention.end].text
            if mention_text.lower() in ["he", "she", "him", "her", "his", "hers", "they", "them"]:
                replacements[(mention.start, mention.end)] = character_name
    
    # Replace text in the original document
    new_text = list(doc)
    for (start, end), name in replacements.items():
        new_text[start:end] = [name]
    
    return " ".join([token.text for token in new_text])