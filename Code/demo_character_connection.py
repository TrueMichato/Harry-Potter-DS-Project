import pandas as pd
from itertools import combinations

# Example data for sentences
data_sentences = {
    'sentence': [
        "Character1 and Character2 went to the market.",
        "Character3 saw Character1 at the park.",
        "Character3 and Character2 had a conversation.",
        "Character1 and Character4 were at the library.",
        "Character5 joined Character2 and Character3 at the cafe.",
        "Character4 and Character5 were discussing a book.",
    ]
}

# Example data for character names
data_characters = {
    'name': [
        "Character1",
        "Character2",
        "Character3",
        "Character4",
        "Character5"
    ]
}




# Convert the example data into DataFrames
df_sentences = pd.DataFrame(data_sentences)
df_characters = pd.DataFrame(data_characters)

# Extract the list of character names from the 'name' column
character_names = df_characters['name'].tolist()


def find_character_pairs(sentence, character_names):
    present_characters = []
    for name in character_names:
        if name in sentence:
            present_characters.append(name)
    return list(combinations(present_characters, 2))


df_sentences['character_pairs'] = df_sentences['sentence'].apply(lambda x: find_character_pairs(x, character_names))
df_exploded = df_sentences.explode('character_pairs').dropna(subset=['character_pairs'])

# Count the occurrences of each character pair
pair_counts = df_exploded['character_pairs'].value_counts().to_dict()

# Output the result
print(pair_counts)
