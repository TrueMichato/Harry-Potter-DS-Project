import pandas as pd
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import re

def create_dict_connections(df_sentences, dict_names_id):
    def find_character_id_pairs(sentence, dict_names_id):
        present_ids = []
        for char_id, names in dict_names_id.items():
            for name in names:
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, sentence):
                    present_ids.append(char_id)
                    break  # Break after the first match to avoid double counting
        return list(set(combinations(sorted(present_ids), 2)))

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

def check_add_node(G, name):
    if name not in G.nodes:
        G.add_node(name)
    return G

def plot_simple_connections(pair_counts, dict_names_id, threshold_count=3):
    G = nx.Graph()
    for pair, count in pair_counts.items():
        if count < threshold_count:
            continue
        name1 = dict_names_id[pair[0]][0]
        name2 = dict_names_id[pair[1]][0]
        G = check_add_node(G, name1)
        G = check_add_node(G, name2)
        if not G.has_edge(name1, name2):
            G.add_edge(name1, name2, weight=count)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(20, 20))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray", font_color='black',
            edgecolors="black", linewidths=1, alpha=0.7, width=2, connectionstyle='arc3, rad = 0.1', arrows=True)
    plt.show()


def remove_characters_below_threshold(dict_names_id, df_sentences, threshold=2):
    # Create a new dictionary to store filtered characters
    filtered_dict = {}

    for id, names in dict_names_id.items():
        total_count = sum(df_sentences['sentence'].str.contains(name, regex=False).sum() for name in names)
        if total_count >= threshold:
            filtered_dict[id] = names

    return filtered_dict

def main():
    df_sentences = pd.read_csv(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\harry_potter_sentences.csv")
    df_characters = pd.read_csv(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\character_names.csv")
    dict_names_id = create_dict_names_id(df_characters)
    dict_names_id = remove_characters_below_threshold(dict_names_id, df_sentences, threshold=14)
    pair_counts = create_dict_connections(df_sentences, dict_names_id)
    # print(pair_counts)
    plot_simple_connections(pair_counts, dict_names_id, threshold_count=10)

if __name__ == "__main__":
    main()

