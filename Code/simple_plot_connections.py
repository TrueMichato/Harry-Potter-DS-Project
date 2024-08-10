import pandas as pd
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

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

    pos = nx.spring_layout(G, k=0.3, iterations=50)  # Increase 'k' for more spacing, and more iterations for stability
    plt.figure(figsize=(12, 12))  # Increase the figure size to allow more space
    nx.draw(G, pos, with_labels=True, node_size=5000, node_color="skyblue", font_size=12, font_weight="bold", edge_color="gray", font_color='black')
    plt.show()


def main():
    df_sentences = pd.read_csv(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\harry_potter_sentences.csv")
    df_characters = pd.read_csv(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\character_names.csv")
    dict_names_id = create_dict_names_id(df_characters)
    pair_counts = create_dict_connections(df_sentences, dict_names_id)
    print(pair_counts)
    plot_simple_connections(pair_counts, dict_names_id, threshold_count=1)

if __name__ == "__main__":
    main()

