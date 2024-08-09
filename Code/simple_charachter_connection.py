import pandas as pd
from itertools import combinations

def create_dict_connections(df_sentences, dict_names_id):
    def find_character_id_pairs(sentence, dict_names_id):
        present_ids = []
        for char_id, names in dict_names_id.items():
            for name in names:
                if name in sentence:
                    present_ids.append(char_id)
                    break  # Break after the first match to avoid double counting
        return list(combinations(sorted(present_ids), 2))

    df_sentences['character_pairs'] = df_sentences['sentence'].apply(lambda x: find_character_id_pairs(x, dict_names_id))
    df_exploded = df_sentences.explode('character_pairs').dropna(subset=['character_pairs'])

    pair_counts = df_exploded['character_pairs'].value_counts().to_dict()
    return pair_counts

def create_dict_names_id(df_characters):
    dict_id_names = {}
    for _, row in df_characters.iterrows():
        all_names = []
        all_names.append(row['Name'])
        if pd.notna(row['Other Names']):
            all_names.extend(row['Other Names'].split(", "))
        dict_id_names[row['Id']] = all_names
    return dict_id_names

def main():
    df_sentences = pd.read_csv(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\harry_potter_sentences.csv")
    df_characters = pd.read_csv(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\character_names.csv")
    dict_names_id = create_dict_names_id(df_characters)
    dict_connections = create_dict_connections(df_sentences, dict_names_id)
    print(dict_connections)

if __name__ == "__main__":
    main()
