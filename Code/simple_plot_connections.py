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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


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


def remove_characters_below_threshold(dict_names_id, df_sentences, threshold=2):
    filtered_dict = {}

    for id, names in dict_names_id.items():
        total_count = sum(
            df_sentences['sentence'].str.contains(r"\b" + re.escape(name) + r"\b", regex=True).sum() for name in names)
        if total_count >= threshold:
            filtered_dict[id] = names

    return filtered_dict


def create_dict_connections(df_sentences, dict_names_id):
    df_sentences['character_pairs'] = df_sentences['sentence'].apply(
        lambda x: find_character_id_pairs(x, dict_names_id))
    df_exploded = df_sentences.explode('character_pairs').dropna(subset=['character_pairs'])

    pair_counts = df_exploded['character_pairs'].value_counts().to_dict()
    return pair_counts


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
    return [tuple(sorted(pair)) for pair in combinations(set(present_ids), 2)]


def create_dict_names_id(df_characters):
    dict_id_names = {}
    for _, row in df_characters.iterrows():
        all_names = []
        all_names.append(row['Name'])
        if pd.notna(row['Other Names']):
            all_names.extend(row['Other Names'].split(", "))
        dict_id_names[row['Id']] = all_names
    return dict_id_names


def create_pair_sentences(df_sentences, dict_names_id):
    # Apply the function to get character pairs for each sentence
    df_sentences['character_pairs'] = df_sentences['sentence'].apply(
        lambda x: find_character_id_pairs(x, dict_names_id))

    pair_dict = {}
    non_empty_indexes = set()

    # Iterate through the DataFrame and populate the dictionary and set
    for index, pairs in df_sentences[['character_pairs']].iterrows():
        if pairs['character_pairs']:  # Check if the list is not empty
            non_empty_indexes.add(index)  # Add to the set of non-empty indexes
            for pair in pairs['character_pairs']:
                if pair not in pair_dict:
                    pair_dict[pair] = [index]
                else:
                    pair_dict[pair].append(index)

    return pair_dict, non_empty_indexes


def check_add_node(G, name):
    if name not in G.nodes:
        G.add_node(name)
    return G


def plot_simple_connections(pair_counts, dict_names_id, threshold_count=3):
    G = create_weighted_graph(pair_counts, dict_names_id, threshold_count=threshold_count)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(20, 20))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold",
            edge_color="gray", font_color='black',
            edgecolors="black", linewidths=1, alpha=0.7, width=2, connectionstyle='arc3, rad = 0.1', arrows=True)
    plt.show()


def make_graph_sparse(G, fraction=0.5):
    edges = list(G.edges(data=True))
    num_edges_to_keep = int(len(edges) * fraction)

    G_sparse = nx.Graph()

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
                num_tries += 1
                overlap = True
                break
        if not overlap:
            num_tries = 0
            pos[node] = (x, y)
    return pos


def plot_page_rank(pair_count, dict_names_id, threshold_count):
    G = create_weighted_graph(pair_count, dict_names_id, threshold_count)
    G = make_graph_sparse(G, fraction=0.2)
    pagerank_scores = nx.pagerank(G, weight='weight')

    nodes = list(G.nodes())
    pos_dict = generate_unique_positions(nodes, width=1, height=1, min_dist=0.1)

    pos = {node: (x, y) for node, (x, y) in zip(nodes, pos_dict.values())}

    min_node_size = 100  # Minimum size for nodes
    scaling_factor = 5000  # Increase this factor to make size differences more noticeable
    # Normalize PageRank scores for node size
    node_sizes = [max(pagerank_scores[node] * scaling_factor, min_node_size) for node in
                  nodes]  # Scale PageRank scores and apply minimum size

    min_value = min(node_sizes)
    max_value = max(node_sizes)
    norm = plt.Normalize(vmin=min_value, vmax=max_value)
    colormap = plt.colormaps['winter_r']  # Choose a colormap
    colors = colormap(norm(node_sizes))  # Use the colormap for color shading

    plt.figure(figsize=(20, 20))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes)
    edges = G.edges(data=True)
    weights = [edge[2]["weight"] for edge in edges]
    custom_cmap = LinearSegmentedColormap.from_list("custom_grey", ['#B2BEB5', '#000000'])
    # # Normalize the edge weights to map them to a darker range of greys
    norm = Normalize(vmin=min(weights), vmax=max(weights))

    # Map weights to colors using the custom colormap
    edge_colors = [custom_cmap(norm(w)) for w in weights]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, edge_color=edge_colors, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=5, font_color='black', font_weight='bold')

    plt.title("Character Relationship Network")
    plt.axis('off')
    plt.show()

    return G, pos


def plot_louvain_communities(G, pos, colormap_name='tab10', resolution=1.0):
    partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)

    cmap = plt.colormaps[colormap_name]
    num_communities = len(set(partition.values()))

    fig, ax = plt.subplots(figsize=(20, 20))
    for spine in ax.spines.values():
        spine.set_visible(False)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=3000,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_color='black')

    plt.title(f"Louvain Community Detection - {num_communities} Communities Detected (Resolution={resolution})")
    plt.show()


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

def plot_leiden_communities(G, pos, colormap_name='tab10', resolution=1.0):
    # Convert the NetworkX graph to an iGraph graph with edge weights
    edges = [(u, v, float(data['weight'])) for u, v, data in G.edges(data=True)]
    G_ig = ig.Graph.TupleList(edges, directed=False, weights=True)

    partition = la.find_partition(G_ig, la.RBConfigurationVertexPartition, weights=G_ig.es['weight'],
                                  resolution_parameter=resolution)
    membership = partition.membership
    node_communities = {node: membership[idx] for idx, node in enumerate(G.nodes())}
    num_communities = len(set(node_communities.values()))

    cmap = plt.colormaps[colormap_name]

    fig, ax = plt.subplots(figsize=(20, 20))
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw nodes with colors based on community membership
    nx.draw_networkx_nodes(G, pos, node_color=[node_communities[node] for node in G.nodes()],
                           node_size=3000, cmap=cmap, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_color='black')

    plt.title(f"Leiden Community Detection with Weights - {num_communities} Communities Detected (Resolution={resolution})")
    plt.show()

    return node_communities


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
            sum_semantic = calc_semantic(pairs_to_indices[pair], indices_to_semantics) / count
            G.add_edge(name1, name2, semantic=sum_semantic)

    # Make the graph sparse
    G = make_graph_sparse(G, fraction=0.2)

    nodes = list(G.nodes())
    pos_dict = generate_unique_positions(nodes, width=1, height=1, min_dist=0.1)

    # Create a position mapping for node names
    pos = {node: (x, y) for node, (x, y) in zip(nodes, pos_dict.values())}

    # Increase the figure size and create axes
    fig, ax = plt.subplots(figsize=(20, 20))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="blue", node_size=300)

    # Draw edges with width proportional to weight
    edges = G.edges(data=True)
    semantics = [edge[2]["semantic"] for edge in edges]
    custom_cmap = LinearSegmentedColormap.from_list("custom_blue_red", ['#0000FF', '#FF0000'], N=6)

    # Normalize the edge weights to map them to a darker range of greys
    norm = Normalize(vmin=min(semantics), vmax=max(semantics))

    # Map semantics to colors using the custom colormap
    edge_colors = [custom_cmap(norm(s)) for s in semantics]

    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color=edge_colors, alpha=0.7)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=5, font_color='black', font_weight='bold')

    # Create a new axes for the colorbar
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.02])  # [left, bottom, width, height]

    # Create the colorbar
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # We don't actually need an array here
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')

    # Remove ticks from the colorbar
    cbar.set_ticks([])

    # Add labels at the ends of the colorbar
    cbar.ax.text(0, 1.5, 'Negative Relationship', va='bottom', ha='left', fontsize=10, color='#0000FF', transform=cbar.ax.transAxes)
    cbar.ax.text(1, 1.5, 'Positive Relationship', va='bottom', ha='right', fontsize=10, color='#FF0000', transform=cbar.ax.transAxes)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    

def analyze_sentiment_advanced(set_sentences, df_sentences):
    # Check if GPU is available and set the device accordingly
    device = 0 if torch.cuda.is_available() else -1

    # Load the sentiment analysis pipeline with the correct device
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=device
    )

    sentiment_dict = {}

    for index in set_sentences:
        sentence = df_sentences.loc[index, 'sentence']
        result = sentiment_pipeline(sentence)[0]

        # The result contains 'label' and 'score', e.g., {'label': 'POSITIVE', 'score': 0.99}
        if result['label'] == 'LABEL_2':  # Positive sentiment
            sentiment_dict[index] = 1
        elif result['label'] == 'LABEL_0':  # Negative sentiment
            sentiment_dict[index] = -1
        else:  # Neutral sentiment (depends on the model; may be labeled differently)
            sentiment_dict[index] = 0
    return sentiment_dict




def analyze_sentiment_vader(set_sentences, df_sentences):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = {}

    for index in set_sentences:
        sentence = df_sentences.loc[index, 'sentence']
        result = analyzer.polarity_scores(sentence)

        # VADER provides a compound score which is a normalized score between -1 and 1
        if result['compound'] >= 0.05:
            sentiment_dict[index] = 1
        elif result['compound'] <= -0.05:
            sentiment_dict[index] = -1
        else:
            sentiment_dict[index] = 0

    return sentiment_dict


def analyze_sentiment_textblob(set_sentences, df_sentences):
    sentiment_dict = {}

    for index in set_sentences:
        sentence = df_sentences.loc[index, 'sentence']
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity

        # TextBlob provides a polarity score between -1 and 1
        if polarity >= 0.05:
            sentiment_dict[index] = 1  # Positive sentiment
        elif polarity <= -0.05:
            sentiment_dict[index] = -1  # Negative sentiment
        else:
            sentiment_dict[index] = 0  # Neutral sentiment

    return sentiment_dict


def save_pair_counts(pair_counts, path_pair_counts) -> None:
    with open(path_pair_counts, "wb") as f:
        pickle.dump(pair_counts, f)


def get_pair_counts_from_pickle(path_pair_counts) -> dict:
    with open(path_pair_counts, "rb") as f:
        pair_counts = pickle.load(f)
    return pair_counts


def save_dict_names_id(dict_names_id, path_names_id) -> None:
    with open(path_names_id, "wb") as f:
        pickle.dump(dict_names_id, f)


def get_dict_names_id_from_pickle(path_names_id) -> dict:
    with open(path_names_id, "rb") as f:
        dict_names_id = pickle.load(f)
    return dict_names_id


def save_pair_sentences(pair_sentences, set_sentences, path_pair_sentences, path_set_sentences) -> None:
    with open(path_pair_sentences, "wb") as f:
        pickle.dump(pair_sentences, f)
    with open(path_set_sentences, "wb") as f:
        pickle.dump(set_sentences, f)


def get_pair_sentences_from_pickle(path_pair_sentences, path_set_sentences):
    with open(path_pair_sentences, "rb") as f:
        pair_sentences = pickle.load(f)
    with open(path_set_sentences, "rb") as f:
        set_sentences = pickle.load(f)
    return pair_sentences, set_sentences


def main(paths) -> None:
    # todo: remove the pickle usage in the future
    df_sentences = pd.read_csv(paths["sentences"])
    df_characters = pd.read_csv(paths["characters"])
    dict_names_id = create_dict_names_id(df_characters)
    dict_names_id = remove_characters_below_threshold(dict_names_id, df_sentences, threshold=16)
    save_dict_names_id(dict_names_id, paths["names_id"])
    pair_sentences, set_sentences = create_pair_sentences(df_sentences, dict_names_id)
    save_pair_sentences(pair_sentences, set_sentences, paths["pair_sentences"], paths["set_sentences"])
    pair_counts = create_dict_connections(df_sentences, dict_names_id)
    save_pair_counts(pair_counts)

    # dict_names_id = get_dict_names_id_from_pickle(paths["names_id"])
    # pair_counts = get_pair_counts_from_pickle(paths["pair_counts"])
    # pair_sentences, set_sentences = get_pair_sentences_from_pickle(paths["pair_sentences"], paths["set_sentences"])

    indices_to_semantics = analyze_sentiment_advanced(set_sentences, df_sentences)

    # plot_simple_connections(pair_counts, dict_names_id, threshold_count=10)
    # G, pos = plot_page_rank(pair_counts, dict_names_id, threshold_count=15)
    # plot_louvain_communities(G, pos, resolution=1.7)
    # plot_leiden_communities(G, pos, resolution=1.7)
    # pair_counts = {(189, 42): 3, (32, 11): 2, (189, 11): 2}
    # dict_names_id = {42: ["Harry", "Daniel"], 189: ["Albus", "Brian"], 32: ["Severus", "Alan"], 11: ["Hermione", "Emma"]}
    # pairs_to_indices = {(189, 42): [0, 1, 2], (32, 11): [3, 4, 5], (189, 11): [1, 2]}
    # indices_to_semantics = {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1}
    plot_semantic_relations(pair_counts, dict_names_id, pair_sentences, indices_to_semantics, threshold_count=300)

if __name__ == "__main__":
    PATH_SENTENCES =  "Data/harry_potter_sentences.csv" # r"..\Data\harry_potter_sentences.csv"
    PATH_CHARACTERS = "Data/character_names.csv" # r"..\Data\character_names.csv"
    PATH_NAMES_ID = "Data/dict_names_id.pkl" # r"..\Data\dict_names_id.pkl"
    PATH_PAIR_COUNTS = "Data/pair_counts.pkl" # r"..\Data\pair_counts.pkl"
    PATH_PAIR_SENTENCES = "Data/pair_sentences.pkl" # r"..\Data\pair_sentences.pkl"
    PATH_SET_SENTENCES = "Data/set_sentences.pkl" # r"..\Data\set_sentences.pkl"
    PATHS = {
        "sentences": PATH_SENTENCES,
        "characters": PATH_CHARACTERS,
        "names_id": PATH_NAMES_ID,
        "pair_counts": PATH_PAIR_COUNTS,
        "pair_sentences": PATH_PAIR_SENTENCES,
        "set_sentences": PATH_SET_SENTENCES,
    }
    main(PATHS)
