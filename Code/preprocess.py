import pandas as pd
import nltk.tokenize as tk
import re
import spacy
import coreferee
import tqdm

# nlp = spacy.load("en_core_web_trf")
# nlp.add_pipe("coreferee")


def resolve_coreferences(chapter_text) -> str:
    doc = nlp(chapter_text)
    text_broken = [token.text for token in doc]

    # Create a mapping from pronouns to resolved mentions
    for chain in doc._.coref_chains:
        for mention in chain:
            if len(mention) > 1:
                for m in mention:
                    fix = doc._.coref_chains.resolve(doc[m])
                    if fix:
                        fix = [f.text for f in fix]
                        text_broken[m] = " and ".join(fix)
            else:
                fix = doc._.coref_chains.resolve(doc[mention.token_indexes[0]])
                if fix:
                    fix = [f.text for f in fix]
                    text_broken[mention.token_indexes[0]] = " and ".join(fix)
    resolved_text = put_together(text_broken)

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
        name1 = "Mr.__" + couple_name
        name2 = "Mrs.__" + couple_name
        new_text = text.replace("Mr. and Mrs. " + couple_name, name1 + " and " + name2)
        return new_text
    return text


def solve_mr_mrs(text: str, mr: bool = True) -> str:
    regex = re.compile(r"Mr\. (\w+)") if mr else re.compile(r"Mrs\. (\w+)")
    match = regex.search(text)
    if match:
        title = "Mr." if mr else "Mrs."
        name = match.group(1)
        new_text = text.replace(title + " " + name, title + "__" + name)
        return new_text
    return text


def solve_mrs_mr_issue(text: str) -> str:
    text = solve_mr_mrs(text)
    return solve_mr_mrs(text, False)


def solve_long_name_issue(text: str, names: list) -> str:
    for name in names:
        if len(name.split(" ")) >= 2:
            new_name = "__".join(name.split(" "))
            # print(f"{name=} -> {new_name=}")
            text = re.sub(rf"\b{name}\b", new_name, text)
    return text


def family_fix(text: str) -> str:
    family_dict = {
        "Dursleys": ["Mr. Dursley", "Mrs. Dursley", "Dudley Dursley"],
        "Potters": ["James Potter", "Lily Potter", "Harry Potter"],
        "Weasleys": [
            "Arthur Weasley",
            "Molly Weasley",
            "Bill Weasley",
            "Charlie Weasley",
            "Percy Weasley",
            "Fred Weasley",
            "George Weasley",
            "Ron Weasley",
            "Ginny Weasley",
        ],
        "Malfoys": ["Lucius Malfoy", "Narcissa Malfoy", "Draco Malfoy"],
    }
    for family in ["Dursleys", "Potters", "Weasleys", "Malfoys"]:
        family_names = " and ".join(family_dict[family])
        text = text.replace(family, family_names)
    return text


def put_together(words: list) -> str:
    sentence = words[0]

    for word in words[1:]:
        if word.isalnum() or (word in ["'", '"']) or ("." in word or "_" in word) or re.match(r"^\w+\ and .*", word):
            sentence += " " + word
        else:
            sentence += word
    return (
        sentence.replace(" .", ".")
        .replace(" ,", ",")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" '", "'")
        .replace(' "', '"')
    )


def preprocess_text(text: str, chapter: int, book: str) -> str:
    print(f"preprocessing {book=}, {chapter=}")
    text = solve_couple(text)
    text = solve_mrs_mr_issue(text)
    text = solve_long_name_issue(text, character_names)
    text = resolve_coreferences(text)
    text = text.replace("__", " ")
    text = family_fix(text)
    return text

def fix_spacing(text: str, names: list) -> str:
    global i
    i += 1
    print(f"{i}/{length}")
    for name in tqdm.tqdm(names):
        text = re.sub(fr'((?<=[a-z]){re.escape(name)})|({re.escape(name)}(?<=[a-z]))', r' ' + name, text)
    return text

def fix_unique_names() -> None:
    df_characters = pd.read_csv("Data/character_names.csv")
    p_name = df_characters["Name"].apply(lambda x: x.split(" ")[0])

    unique_mask = p_name.duplicated(keep=False) == False

    df_characters.loc[unique_mask, 'Other Names'] = df_characters.loc[unique_mask, 'Other Names'].fillna('') + \
                                        df_characters.loc[unique_mask, 'Other Names'].apply(lambda x: ', ' if x else '') + \
                                        p_name[unique_mask]

    # Remove leading/trailing whitespace and commas
    df_characters['Other Names'] = df_characters['Other Names'].str.strip(', ')
    df_characters['Other Names'] = df_characters['Other Names'].str.replace(' , ', ', ')
    df_characters['Other Names'] = df_characters['Other Names'].fillna('')
    df_characters['Other Names'] = df_characters['Other Names'].str.split(', ').apply(lambda x: ", ".join(set(x)))

    df_characters.to_csv("Data/character_names.csv", index=False)

# fix_unique_names()

# data = pd.read_csv("Data/harry_potter_books.csv")
df_characters = pd.read_csv("Data/character_names.csv")

dict_names_id = create_dict_names_id(df_characters)
character_names = set([name for names in dict_names_id.values() for name in names])

# book_texts = (
#     data.groupby(["book", "chapter"])["text"].apply(lambda x: " ".join(x)).reset_index()
# )
# book_texts["chapter"] = book_texts["chapter"].apply(lambda x: int(x.split("-")[1]))
# book_texts = book_texts.sort_values(by=["book", "chapter"])
# book_texts = book_texts.reset_index(drop=True)
# print(book_texts.head(15))
# book_texts.to_csv("Data/harry_potter_books_test.csv", index=False)
# book_texts["text"] = book_texts.apply(lambda row: preprocess_text(row["text"], row["chapter"], row["book"]), axis=1)
# # get chapter name using a regex
# #  book_texts["chapter name"] = book_texts["text"].str.extract(r'^([A-Z ]+)\s\s')
# # # make chapter name a string
# # book_texts["chapter name"] = book_texts["chapter name"].apply(lambda x: str(x))
# # book_texts["text"] = book_texts.apply(lambda row: row['text'].replace(row['chapter name'], '').replace('  ', ' '), axis=1)


# split text to sentences
# book_texts["sentences"] = book_texts["text"].apply(lambda x: tk.sent_tokenize(x))
# print(book_texts.head(15))

# book_texts.to_csv("Data/harry_potter_books_preprocessed.csv", index=False)

# # create a list of all sentences, without book or chapter affiliation
# all_sentences = [
#     sentence for sentences in book_texts["sentences"] for sentence in sentences
# ]
# all_sentences_df = pd.DataFrame(all_sentences, columns=["sentence"])

# all_sentences_df.to_csv("Data/harry_potter_sentences.csv", index=False)
preprocessed_data = pd.read_csv("Data/harry_potter_books_preprocessed.csv")
length = len(preprocessed_data.index)
i = 0

preprocessed_data['text'] = preprocessed_data['text'].apply(lambda x: fix_spacing(x, character_names))
preprocessed_data["sentences"] = preprocessed_data["text"].apply(lambda x: tk.sent_tokenize(x))
# # print(book_texts.head(15))

preprocessed_data.to_csv("Data/harry_potter_books_preprocessed.csv", index=False)
# create a list of all sentences, without book or chapter affiliation
# all_sentences = [
#     sentence for sentences in preprocessed_data["sentences"] for sentence in sentences
# ]
# all_sentences_df = pd.DataFrame(all_sentences, columns=["sentence"])

# all_sentences_df.to_csv("Data/harry_potter_sentences.csv", index=False)
