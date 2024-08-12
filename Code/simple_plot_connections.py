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
import community as community_louvain
import igraph as ig
import leidenalg as la
from cdlib import algorithms
from matplotlib.colors import LinearSegmentedColormap, Normalize


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
    G = create_weighted_graph(pair_counts, dict_names_id, threshold_count=10)

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
        total_count = sum(df_sentences['sentence'].str.contains(r"\b" + re.escape(name) + r"\b", regex=True).sum() for name in names)
        if total_count >= threshold:
            filtered_dict[id] = names

    return filtered_dict


def make_graph_sparse(G, fraction=0.5):
    # Get all edges with their attributes
    edges = list(G.edges(data=True))
    num_edges_to_keep = int(len(edges) * fraction)

    # Sample edges to keep
    edges_to_keep = random.sample(edges, num_edges_to_keep)

    # Create a new graph
    G_sparse = nx.Graph()

    # Add nodes and edges with attributes to the sparse graph
    for edge in edges:
        u, v, attr = edge
        G_sparse.add_edge(u, v, **attr)

    return G_sparse


def generate_unique_positions(nodes, width=1, height=1, min_dist=0.1):
    pos = {}
    num_tries = 0
    while len(pos) < len(nodes):
        node = len(pos)
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        overlap = False
        for (ox, oy) in pos.values():
            if np.sqrt((x - ox) ** 2 + (y - oy) ** 2) < min_dist and num_tries < 1000:
                num_tries+=1
                overlap = True
                break
        if not overlap:
            num_tries = 0
            pos[node] = (x, y)
    return pos


def plot_page_rank(pair_counts, dict_names_id, threshold_count):
    G = create_weighted_graph(pair_counts, dict_names_id, threshold_count)

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
    colormap = plt.colormaps['winter_r']  # Choose a colormap
    colors = colormap(norm(node_sizes))  # Use the colormap for color shading

    # Increase the figure size
    plt.figure(figsize=(20, 20))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes)

    # Draw edges with width proportional to weight
    edges = G.edges(data=True)
    weights = [edge[2]["weight"] for edge in edges]

    custom_cmap = LinearSegmentedColormap.from_list("custom_grey", ['#B2BEB5', '#000000'])
    # # Normalize the edge weights to map them to a darker range of greys
    norm = Normalize(vmin=min(weights), vmax=max(weights))

    # Map weights to colors using the custom colormap
    edge_colors = [custom_cmap(norm(w)) for w in weights]

    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, edge_color=edge_colors, alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=5, font_color='black', font_weight='bold')

    plt.title("Character Relationship Network")
    plt.axis('off')
    plt.show()

    return G, pos



def plot_weighted_connections(pair_counts, dict_names_id, threshold_count=3, min_color=0.3, colormap_name="Oranges", power_factor=2):
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

    pos = nx.random_layout(G)

    fig, ax = plt.subplots(figsize=(25, 25))

    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]

    max_weight = max(weights) if weights else 1
    min_weight = min(weights) if weights else 1
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)

    cmap = plt.colormaps[colormap_name]
    edge_colors = [cmap(min_color + (norm(weight) ** power_factor) * (1 - min_color)) for weight in weights]

    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue",
            font_size=10, font_weight="bold", edge_color=edge_colors,
            font_color='black', edgecolors="black", linewidths=1, alpha=0.7,
            width=3, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Edge Weight', rotation=270, labelpad=20)
    cbar.ax.invert_yaxis()

    plt.show()


def plot_louvain_communities(G, pos, colormap_name='tab10', resolution=1.0):
    partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)

    cmap = plt.colormaps[colormap_name]
    num_communities = len(set(partition.values()))

    fig, ax = plt.subplots(figsize=(20, 20))
    for spine in ax.spines.values():
        spine.set_visible(False)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=300,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=4, font_weight="bold", font_color='black')

    plt.title(f"Louvain Community Detection - {num_communities} Communities Detected (Resolution={resolution})")
    plt.show()


# def plot_louvain_communities(G, pos, colormap_name='tab10'):
#     partition = community_louvain.best_partition(G, weight='weight')
#
#     cmap = plt.colormaps[colormap_name]
#     num_communities = len(set(partition.values()))
#
#     fig, ax = plt.subplots(figsize=(20, 20))
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#     nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=3000,
#                            cmap=cmap, node_color=list(partition.values()))
#     nx.draw_networkx_edges(G, pos, alpha=0.5)
#     nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_color='black')
#
#     plt.title(f"Louvain Community Detection - {num_communities} Communities Detected")
#     plt.show()


def create_weighted_graph(pair_counts, dict_names_id, threshold_count=3):
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
    return G


# def plot_leiden_communities_with_weights(G, pos, colormap_name='tab10'):
#     # Convert the NetworkX graph to an iGraph graph with edge weights
#     # Ensure that the weights are floating point values
#     edges = [(u, v, float(data['weight'])) for u, v, data in G.edges(data=True)]
#     G_ig = ig.Graph.TupleList(edges, directed=False, weights=True)
#
#     # Perform Leiden community detection using edge weights
#     partition = la.find_partition(G_ig, la.ModularityVertexPartition, weights=G_ig.es['weight'])
#
#     # Get the community membership of each node
#     membership = partition.membership
#
#     # Map the memberships back to the NetworkX nodes
#     node_communities = {node: membership[idx] for idx, node in enumerate(G.nodes())}
#
#     # Get the unique communities
#     num_communities = len(set(node_communities.values()))
#
#     # Prepare the colormap
#     cmap = plt.colormaps[colormap_name]
#
#     fig, ax = plt.subplots(figsize=(20, 20))
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#
#     # Draw nodes with colors based on community membership
#     nx.draw_networkx_nodes(G, pos, node_color=[node_communities[node] for node in G.nodes()],
#                            node_size=3000, cmap=cmap, alpha=0.8)
#
#     # Draw edges
#     nx.draw_networkx_edges(G, pos, alpha=0.5)
#
#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_color='black')
#
#     plt.title(f"Leiden Community Detection with Weights - {num_communities} Communities Detected")
#     plt.show()
#
#     return node_communities


def plot_leiden_communities_with_weights(G, pos, colormap_name='tab10', resolution=1.0):
    # Convert the NetworkX graph to an iGraph graph with edge weights
    edges = [(u, v, float(data['weight'])) for u, v, data in G.edges(data=True)]
    G_ig = ig.Graph.TupleList(edges, directed=False, weights=True)

    # Perform Leiden community detection using edge weights
    partition = la.find_partition(G_ig, la.RBConfigurationVertexPartition, weights=G_ig.es['weight'], resolution_parameter=resolution)

    # Get the community membership of each node
    membership = partition.membership

    # Map the memberships back to the NetworkX nodes
    node_communities = {node: membership[idx] for idx, node in enumerate(G.nodes())}

    # Get the unique communities
    num_communities = len(set(node_communities.values()))

    # Prepare the colormap
    cmap = plt.colormaps[colormap_name]

    fig, ax = plt.subplots(figsize=(20, 20))
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw nodes with colors based on community membership
    nx.draw_networkx_nodes(G, pos, node_color=[node_communities[node] for node in G.nodes()],
                           node_size=300, cmap=cmap, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=4, font_weight="bold", font_color='black')

    plt.title(f"Leiden Community Detection with Weights - {num_communities} Communities Detected (Resolution={resolution})")
    plt.show()

    return node_communities


def plot_surprise_communities(G, pos=None, colormap_name='spring'):
    # Perform Surprise community detection
    communities = algorithms.surprise_communities(G)

    # Extract communities and the number of communities detected
    community_map = {node: i for i, community in enumerate(communities.communities) for node in community}
    num_communities = len(communities.communities)

    # If no position is provided, use spring layout
    if pos is None:
        pos = nx.spring_layout(G)

    # Prepare the colormap
    cmap = plt.colormaps[colormap_name]

    # Plot the graph with node colors based on their community
    plt.figure(figsize=(20, 20))
    for i, community in enumerate(communities.communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_size=3000,
                               node_color=[cmap(i / num_communities)], alpha=0.8)

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_color='black')

    plt.title(f"Surprise Community Detection - {num_communities} Communities Detected")
    plt.show()

def calc_semantic(indices, indices_to_semantics):
    sum_semantic = 0
    for i in indices:
        sum_semantic+= indices_to_semantics[i]
    return sum_semantic

def plot_semantic_relations(pair_counts, dict_names_id, pairs_to_indices, indices_to_semantics, threshold_count=30):
    G = nx.Graph()
    for pair, count in pair_counts.items():
        if count < threshold_count:
            continue
        name1 = dict_names_id[pair[0]][0]
        name2 = dict_names_id[pair[1]][0]
        G = check_add_node(G, name1)
        G = check_add_node(G, name2)
        if not G.has_edge(name1, name2):
            sum_semantic = calc_semantic(pairs_to_indices[pair], indices_to_semantics)
            G.add_edge(name1, name2, semantic=sum_semantic)

    # Make the graph sparse
    G = make_graph_sparse(G, fraction=0.2)

    nodes = list(G.nodes())
    pos_dict = generate_unique_positions(nodes, width=1, height=1, min_dist=0.1)

    # Create a position mapping for node names
    pos = {node: (x, y) for node, (x, y) in zip(nodes, pos_dict.values())}

    # Increase the figure size
    plt.figure(figsize=(20, 20))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="blue", node_size=300)

    # Draw edges with width proportional to weight
    edges = G.edges(data=True)
    semantics = [edge[2]["semantic"] for edge in edges]

    custom_cmap = LinearSegmentedColormap.from_list("custom_blue_red", ['#0000FF', '#FF0000'])
    # # Normalize the edge weights to map them to a darker range of greys
    norm = Normalize(vmin=min(semantics), vmax=max(semantics))

    # Map semantics to colors using the custom colormap
    edge_colors = [custom_cmap(norm(s)) for s in semantics]

    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, edge_color=edge_colors, alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=5, font_color='black', font_weight='bold')

    # # Create the colorbar
    # fig = plt.gcf()
    # ax = fig.add_axes([0.9, 0.1, 0.03, 0.8])  # Position the colorbar on the right side
    # sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    # sm.set_array([])  # We don't actually need an array here
    #
    # cbar = plt.colorbar(sm, cax=ax)
    # cbar.set_label('Character Relationship Sentiment', fontsize=12)
    # cbar.set_ticks([0, 0.5, 1])
    # cbar.set_ticklabels(['Negative (Blue)', 'Neutral (Mixed)', 'Positive (Red)'])
    # Create the colorbar
    fig = plt.gcf()
    ax = fig.add_axes([0.1, 0.05, 0.6, 0.02])  # Adjust size and position

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # We don't actually need an array here

    cbar = plt.colorbar(sm, cax=ax)
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Negative Relationship', 'Positive Relationship'])

    # Add labels next to the colorbar
    cbar.ax.text(1, 2.7, 'Negative Relationship', va='top', ha='left', fontsize=10, color='#0000FF')
    cbar.ax.text(3, 2.7, 'Positive Relationship', va='top', ha='right', fontsize=10, color='#FF0000')

    # plt.title("Character Relationship Network")
    plt.axis('off')
    plt.show()



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
    # dict_names_id = remove_characters_below_threshold(dict_names_id, df_sentences, threshold=16)
    # save_dict_names_id(dict_names_id)
    # pair_counts = create_dict_connections(df_sentences, dict_names_id)
    # save_pair_counts(pair_counts)

    dict_names_id = get_dict_names_id_from_pickle()
    pair_counts = get_pair_counts_from_pickle()

    # plot_simple_connections(pair_counts, dict_names_id, threshold_count=10)
    # plot_weighted_connections(pair_counts, dict_names_id, threshold_count=10)
    # G, pos = plot_page_rank(pair_counts, dict_names_id, threshold_count=15)

    # For Louvain Community Detection with adjusted resolution
    # plot_louvain_communities(G, pos, resolution=1.7)

    # For Leiden Community Detection with adjusted resolution
    # plot_leiden_communities_with_weights(G, pos, resolution=1.7)

    pair_counts = {(189, 42): 3, (32, 11): 2}
    dict_names_id = {42: ["Harry", "Daniel"], 189: ["Albus", "Brian"], 32: ["Severus", "Alan"], 11: ["Hermione", "Emma"]}
    pairs_to_indices = {(189, 42): [0, 1, 2], (32, 11): [3, 4, 5]}
    indices_to_semantics = {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1}
    plot_semantic_relations(pair_counts, dict_names_id, pairs_to_indices, indices_to_semantics, threshold_count=2)
    # plot_surprise_communities(G, pos)

if __name__ == "__main__":
    main()
