import pandas as pd
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import re
import pickle
import numpy as np
import matplotlib.colors as mcolors


def check_special_family_names(name, sentence):
    # making sure it won't catch the wrong family member cause this family names are associated
    # with the main characters
    pattern = r'\b(\w+)\b\s+' + re.escape(name) + r'\b'
    match = re.search(pattern, sentence)
    if match:
        prev_word = match.group(1)
        if name == "Potter":
            return prev_word in ["Mr.", "Harry"]
        if name == "Weasley":
            return prev_word in ["Ron", "Ronald"]
        if name == "Malfoy":
            return prev_word in ["Mr.", "Draco"]
    return True


def create_dict_connections(df_sentences, dict_names_id):
    def find_character_id_pairs(sentence, dict_names_id):
        present_ids = []
        for char_id, names in dict_names_id.items():
            for name in names:
                pattern = r'\b' + re.escape(name) + r'\b'
                match = re.search(pattern, sentence)
                if match:
                    if name in ["Potter", "Weasley", "Malfoy"]:
                        if not check_special_family_names(name, sentence):
                            break
                    present_ids.append(char_id)
                    break  # Break after the first match to avoid double counting
        return list(set(combinations(sorted(present_ids), 2)))

    df_sentences['character_pairs'] = df_sentences['sentence'].apply(
        lambda x: find_character_id_pairs(x, dict_names_id))
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
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold",
            edge_color="gray", font_color='black',
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


# def plot_weighted_connections(pair_counts, dict_names_id, threshold_count=3):
#     G = nx.Graph()
#
#     # Add nodes and weighted edges to the graph
#     for pair, count in pair_counts.items():
#         if count < threshold_count:
#             continue
#         name1 = dict_names_id[pair[0]][0]
#         name2 = dict_names_id[pair[1]][0]
#         G = check_add_node(G, name1)
#         G = check_add_node(G, name2)
#         if not G.has_edge(name1, name2):
#             G.add_edge(name1, name2, weight=count)
#
#     # Sort nodes alphabetically
#     sorted_nodes = sorted(G.nodes())
#
#     # Apply circular layout with sorted nodes
#     pos = nx.circular_layout(G)
#     pos = {node: pos[node] for node in sorted_nodes}  # Maintain circular layout but with sorted order
#
#     plt.figure(figsize=(20, 20))
#
#     # Extract edge weights
#     edges = G.edges(data=True)
#     weights = [edge[2]['weight'] for edge in edges]
#
#     # Draw the graph with weighted edges
#     nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue",
#             font_size=10, font_weight="bold", edge_color="gray",
#             font_color='black', edgecolors="black", linewidths=1, alpha=0.7,
#             width=[weight * 0.1 for weight in weights])  # Edge thickness proportional to weight
#
#     plt.show()


def plot_weighted_connections(pair_counts, dict_names_id, threshold_count=3):
    G = nx.Graph()

    # Add nodes and weighted edges to the graph
    for pair, count in pair_counts.items():
        if count < threshold_count:
            continue
        name1 = dict_names_id[pair[0]][0]
        name2 = dict_names_id[pair[1]][0]
        G = check_add_node(G, name1)
        G = check_add_node(G, name2)
        if not G.has_edge(name1, name2):
            G.add_edge(name1, name2, weight=count)

    # Sort nodes alphabetically
    sorted_nodes = sorted(G.nodes())

    # Calculate circular layout positions for nodes in alphabetical order
    num_nodes = len(sorted_nodes)
    angle_step = 2 * np.pi / num_nodes
    pos = {node: (np.cos(i * angle_step), np.sin(i * angle_step)) for i, node in enumerate(sorted_nodes)}

    plt.figure(figsize=(20, 20))

    # Extract edge weights
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]

    # Normalize the weights for color mapping and edge width
    max_weight = max(weights) if weights else 1
    min_weight = min(weights) if weights else 1
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)

    # Use the updated method to get the colormap
    cmap = plt.colormaps.get_cmap('Greys')
    edge_colors = [cmap(norm(weight) * 0.7 + 0.3) for weight in weights]  # Avoid very light colors
    edge_widths = [norm(weight) * 5 for weight in weights]  # Scale edge width by weight

    # Draw the graph with weighted edges
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue",
            font_size=10, font_weight="bold", edge_color=edge_colors,
            font_color='black', edgecolors="black", linewidths=1, alpha=0.7,
            width=edge_widths)  # Edge width proportional to weight

    plt.show()





def save_pair_counts(pair_counts):
    with open(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\pair_counts.pkl", "wb") as f:
        pickle.dump(pair_counts, f)


def get_pair_counts_from_pickle():
    with open(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\pair_counts.pkl", "rb") as f:
        pair_counts = pickle.load(f)
    return pair_counts


def save_dict_names_id(dict_names_id):
    with open(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\dict_names_id.pkl", "wb") as f:
        pickle.dump(dict_names_id, f)


def get_dict_names_id_from_pickle():
    with open(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\dict_names_id.pkl", "rb") as f:
        dict_names_id = pickle.load(f)
    return dict_names_id


def main():
    # todo: remove the pickle usage in the future
    # df_sentences = pd.read_csv(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\harry_potter_sentences.csv")
    # df_characters = pd.read_csv(r"C:\Users\mikal\Documents\CSMSE\needle\final_project\Data\character_names.csv")
    # dict_names_id = create_dict_names_id(df_characters)
    # dict_names_id = remove_characters_below_threshold(dict_names_id, df_sentences, threshold=14)
    # save_dict_names_id(dict_names_id)
    # pair_counts = create_dict_connections(df_sentences, dict_names_id)
    # save_pair_counts(pair_counts)

    dict_names_id = get_dict_names_id_from_pickle()
    pair_counts = get_pair_counts_from_pickle()
    # plot_simple_connections(pair_counts, dict_names_id, threshold_count=10)
    # todo: fix the plotting of the weight by edge color
    plot_weighted_connections(pair_counts, dict_names_id, threshold_count=10)

    # todo: Louvain for community detection for the plot
    # pagerank


if __name__ == "__main__":
    main()
