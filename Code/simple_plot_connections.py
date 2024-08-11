import pandas as pd
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import re
import pickle
import numpy as np
import matplotlib.colors as mcolors
import random
import matplotlib.cm as cm


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
                            continue
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


# def make_graph_sparse(G, fraction=0.5):
#     edges = list(G.edges())
#     num_edges_to_keep = int(len(edges) * fraction)
#     edges_to_keep = random.sample(edges, num_edges_to_keep)
#     G_sparse = nx.Graph()
#     G_sparse.add_edges_from(edges_to_keep)
#     return G_sparse

def make_graph_sparse(G, fraction=0.5):
    # Get all edges with their attributes
    edges = list(G.edges(data=True))
    num_edges_to_keep = int(len(edges) * fraction)

    # Sample edges to keep
    edges_to_keep = random.sample(edges, num_edges_to_keep)

    # Create a new graph
    G_sparse = nx.Graph()

    # Add nodes and edges with attributes to the sparse graph
    for edge in edges_to_keep:
        u, v, attr = edge
        G_sparse.add_edge(u, v, **attr)

    return G_sparse


def generate_unique_positions(nodes, width=1, height=1, min_dist=0.1):
    pos = {}
    while len(pos) < len(nodes):
        node = len(pos)
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        overlap = False
        for (ox, oy) in pos.values():
            if np.sqrt((x - ox) ** 2 + (y - oy) ** 2) < min_dist:
                overlap = True
                break
        if not overlap:
            pos[node] = (x, y)
    return pos


def plot_try(pair_counts, dict_names_id, threshold_count):
    G = nx.Graph()
    for pair, count in pair_counts.items():
        if count < threshold_count:
            continue
        name1 = dict_names_id[pair[0]][0]
        name2 = dict_names_id[pair[1]][0]
        G.add_node(name1)
        G.add_node(name2)
        if not G.has_edge(name1, name2):
            G.add_edge(name1, name2, weight=count)

    # Make the graph sparse
    G = make_graph_sparse(G, fraction=0.2)

    # Compute PageRank
    pagerank_scores = nx.pagerank(G, weight='weight')

    nodes = list(G.nodes())
    pos_dict = generate_unique_positions(nodes, width=1, height=1, min_dist=0.1)

    # Create a position mapping for node names
    pos = {node: (x, y) for node, (x, y) in zip(nodes, pos_dict.values())}

    # Define a minimum node size and a scaling factor for PageRank
    min_node_size = 100  # Minimum size for nodes
    scaling_factor = 5000  # Increase this factor to make size differences more noticeable
    # Normalize PageRank scores for node size
    node_sizes = [max(pagerank_scores[node] * scaling_factor, min_node_size) for node in
                  nodes]  # Scale PageRank scores and apply minimum size

    # # Normalize sizes for color mapping
    min_value = min(node_sizes)
    max_value = max(node_sizes)
    norm = plt.Normalize(vmin=min_value, vmax=max_value)
    colormap = cm.get_cmap('winter_r')  # Choose a colormap
    colors = colormap(norm(node_sizes))  # Use the colormap for color shading

    # Increase the figure size
    plt.figure(figsize=(20, 20))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes)

    # Draw edges with width proportional to weight
    edges = G.edges(data=True)
    weights = [edge[2]["weight"] for edge in edges]
    # weights get_edge_attributes
    min_width = 0.5  # Set a minimum edge width
    max_width = 2  # Set a maximum edge width
    scaling_factor_edges = 10  # Adjust this scaling factor as needed
    widths = [min(max(w / scaling_factor_edges, min_width), max_width) for w in weights]
    # widths = [max(w / 10, min_width) for w in weights]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, edge_color='gray', alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=5, font_color='black', font_weight='bold')

    plt.title("Character Relationship Network")
    plt.axis('off')
    plt.show()


# def plot_weighted_connections(pair_counts, dict_names_id, threshold_count=3, min_color=0.3, colormap_name="Oranges",
#                               power_factor=2):
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
#     # # Sort nodes alphabetically
#     # sorted_nodes = sorted(G.nodes())
#
#     # Sort nodes by last name
#     def get_last_name(full_name):
#         return full_name.split()[-1]
#
#     sorted_nodes = sorted(G.nodes(), key=get_last_name)
#
#     # Calculate circular layout positions for nodes in alphabetical order
#     num_nodes = len(sorted_nodes)
#     angle_step = 2 * np.pi / num_nodes
#     pos = {node: (np.cos(i * angle_step), np.sin(i * angle_step)) for i, node in enumerate(sorted_nodes)}
#
#     fig, ax = plt.subplots(figsize=(25, 25))
#
#     # Extract edge weights
#     edges = G.edges(data=True)
#     weights = [edge[2]['weight'] for edge in edges]
#
#     # Normalize the weights for color mapping
#     max_weight = max(weights) if weights else 1
#     min_weight = min(weights) if weights else 1
#     norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
#
#     # Apply a power transformation to increase contrast between colors
#     cmap = plt.colormaps.get_cmap(colormap_name)
#     edge_colors = [cmap(min_color + (norm(weight) ** power_factor) * (1 - min_color)) for weight in weights]
#
#     # Draw the graph with uniform edge width and higher contrast colors
#     nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue",
#             font_size=10, font_weight="bold", edge_color=edge_colors,
#             font_color='black', edgecolors="black", linewidths=1, alpha=0.7,
#             width=3, ax=ax)
#
#     # Create a scalar mappable for the color bar legend
#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#     sm.set_array([])  # You need to set an array for the color bar, but it's not used here
#     cbar = plt.colorbar(sm, ax=ax, shrink=0.8)  # Add the color bar and pass the ax object
#     cbar.set_label('Edge Weight', rotation=270, labelpad=20)  # Label for the color bar
#     cbar.ax.invert_yaxis()  # Optional: invert color bar to have lighter colors at the bottom
#
#     plt.show()


def save_pair_counts(pair_counts):
    with open(r"..\Data\pair_counts.pkl", "wb") as f:
        pickle.dump(pair_counts, f)


def get_pair_counts_from_pickle():
    with open(r"..\Data\pair_counts.pkl", "rb") as f:
        pair_counts = pickle.load(f)
    return pair_counts


def save_dict_names_id(dict_names_id):
    with open(r"..\Data\dict_names_id.pkl", "wb") as f:
        pickle.dump(dict_names_id, f)


def get_dict_names_id_from_pickle():
    with open(r"..\Data\dict_names_id.pkl", "rb") as f:
        dict_names_id = pickle.load(f)
    return dict_names_id


def main():
    # todo: remove the pickle usage in the future
    # df_sentences = pd.read_csv(r"..\Data\harry_potter_sentences.csv")
    # df_characters = pd.read_csv(r"..\Data\character_names.csv")
    # dict_names_id = create_dict_names_id(df_characters)
    # dict_names_id = remove_characters_below_threshold(dict_names_id, df_sentences, threshold=14)
    # save_dict_names_id(dict_names_id)
    # pair_counts = create_dict_connections(df_sentences, dict_names_id)
    # save_pair_counts(pair_counts)

    dict_names_id = get_dict_names_id_from_pickle()
    pair_counts = get_pair_counts_from_pickle()
    # plot_simple_connections(pair_counts, dict_names_id, threshold_count=10)
    # todo: fix the plotting of the weight by edge color
    # plot_weighted_connections(pair_counts, dict_names_id, threshold_count=10)
    plot_try(pair_counts, dict_names_id, threshold_count=5)
    # todo: Louvain for community detection for the plot
    # pagerank


if __name__ == "__main__":
    main()
