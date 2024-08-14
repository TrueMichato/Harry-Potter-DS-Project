import pandas as pd
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import re
import pickle
import numpy as np
import community as community_louvain
import igraph as ig
import leidenalg as la
import cdlib
from matplotlib.colors import LinearSegmentedColormap, Normalize
from textblob import TextBlob
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from networkx.algorithms.community import partition_quality


def check_special_family_names(name: str, sentence: str) -> bool:
    """
    Checks if a family name in a sentence is associated with a specific main character.
    This makes sure it won't catch the wrong family member because this family names are associated
    with the main characters
    Args:
        name (str): The family name to check (e.g., "Potter").
        sentence (str): The sentence to search for the name.

    Returns:
        bool: True if the name is correctly associated with the main character, False otherwise.
    """
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


def check_special_family_names(name: str, sentence: str) -> bool:
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


def remove_characters_below_threshold(dict_names_id: dict, df_sentences: pd.DataFrame, threshold: int = 2) -> dict:
    """
    Removes characters from the dictionary if they appear in fewer sentences than the threshold.
    This is done to better control the network size and focus on the most relevant characters.

    Args:
        dict_names_id (dict): Dictionary mapping character IDs to their names.
        df_sentences (pd.DataFrame): DataFrame containing all book sentences to check for character mentions.
        threshold (int): Minimum number of occurrences required to keep the character. Default is 2.

    Returns:
        dict: Filtered dictionary with characters meeting the threshold.
    """
    filtered_dict = {}

    for id, names in dict_names_id.items():
        total_count = sum(
            df_sentences['sentence'].str.contains(r"\b" + re.escape(name) + r"\b", regex=True).sum() for name in names)
        if total_count >= threshold:
            filtered_dict[id] = names

    return filtered_dict


def create_dict_connections(df_sentences: pd.DataFrame, dict_names_id: dict) -> dict:
    """
    Creates a dictionary of character pairs - if both characters appear in the same sentence,
    we increase their dictionary value.

    Args:
        df_sentences (pd.DataFrame): DataFrame containing all book sentences.
        dict_names_id (dict): Dictionary mapping character IDs to their names.

    Returns:
        dict: Dictionary with character pair counts.
    """
    df_sentences['character_pairs'] = df_sentences['sentence'].apply(
        lambda x: find_character_id_pairs(x, dict_names_id))
    df_exploded = df_sentences.explode('character_pairs').dropna(subset=['character_pairs'])

    pair_counts = df_exploded['character_pairs'].value_counts().to_dict()
    return pair_counts


def find_character_id_pairs(sentence: str, dict_names_id: dict) -> list[tuple]:
    """
    Finds pairs of character IDs mentioned in the same sentence.

    Args:
        sentence (str): The sentence to analyze.
        dict_names_id (dict): Dictionary mapping character IDs to their names.

    Returns:
        list[tuple]: List of tuples containing character ID pairs.
    """
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
                break
    return [tuple(sorted(pair)) for pair in combinations(set(present_ids), 2)]


def create_dict_names_id(df_characters: pd.DataFrame) -> dict:
    """
    Creates a dictionary mapping character IDs to their names.

    Args:
        df_characters (pd.DataFrame): DataFrame containing character information.

    Returns:
        dict: Dictionary mapping character IDs to their names.
    """
    dict_id_names = {}
    for _, row in df_characters.iterrows():
        all_names = []
        all_names.append(row['Name'])
        if pd.notna(row['Other Names']):
            all_names.extend(row['Other Names'].split(", "))
        dict_id_names[row['Id']] = all_names
    return dict_id_names


def create_pair_sentences(df_sentences: pd.DataFrame, dict_names_id: dict) -> tuple[dict, set]:
    """
    Creates a dictionary of character pairs with their sentence indices and a set of non-empty indexes.

    Args:
        df_sentences (pd.DataFrame): DataFrame containing all book sentences.
        dict_names_id (dict): Dictionary mapping character IDs to their names.

    Returns:
        tuple: A dictionary of character pairs with sentence indices and a set of non-empty indexes.
    """
    df_sentences['character_pairs'] = df_sentences['sentence'].apply(
        lambda x: find_character_id_pairs(x, dict_names_id))

    pair_dict = {}
    non_empty_indexes = set()

    for index, pairs in df_sentences[['character_pairs']].iterrows():
        if pairs['character_pairs']:
            non_empty_indexes.add(index)
            for pair in pairs['character_pairs']:
                if pair not in pair_dict:
                    pair_dict[pair] = [index]
                else:
                    pair_dict[pair].append(index)

    return pair_dict, non_empty_indexes


def check_add_node(G: nx.Graph, name: str) -> nx.Graph:
    """
    Checks if a node exists in the graph, and adds it if it doesn't.

    Args:
        G (nx.Graph): The graph to check.
        name (str): The node name to check and potentially add.

    Returns:
        nx.Graph: The updated graph.
    """
    if name not in G.nodes:
        G.add_node(name)
    return G


def plot_simple_connections(pair_counts: dict, dict_names_id: dict, threshold_count: int = 3) -> None:
    """
    Plots a simple connection graph of character relationships based on pair counts.

    Args:
        pair_counts (dict): Dictionary of character pair counts.
        dict_names_id (dict): Dictionary mapping character IDs to their names.
        threshold_count (int): Minimum count to include a pair. Default is 3.
    """
    G = create_weighted_graph(pair_counts, dict_names_id, threshold_count=threshold_count)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(20, 20))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold",
            edge_color="gray", font_color='black',
            edgecolors="black", linewidths=1, alpha=0.7, width=2, connectionstyle='arc3, rad = 0.1', arrows=True)
    plt.show()


def generate_unique_positions(nodes: list, width: int = 1, height: int = 1, min_dist: float = 0.1) -> dict:
    """
    Generates unique positions for nodes within a specified area, ensuring minimum distance between them.

    Args:
        nodes (list): List of nodes to position.
        width (int): Width of the area. Default is 1.
        height (int): Height of the area. Default is 1.
        min_dist (float): Minimum distance between nodes. Default is 0.1.

    Returns:
        dict: Dictionary of node positions.
    """
    pos = {}
    len_pos = 0
    num_tries = 0
    len_nodes = len(nodes)
    while len_pos < len_nodes:
        node = len_pos
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
            len_pos += 1
    return pos


def plot_page_rank(pair_count: dict, dict_names_id: dict, threshold_count: int) -> tuple[nx.Graph, dict]:
    """
    Plots a PageRank-based graph of character relationships. 
    The pagerank controls the size and color of the nodes, 
    while the weight of the edges controls their color intensity.

    Args:
        pair_count (dict): Dictionary of character pair counts.
        dict_names_id (dict): Dictionary mapping character IDs to their names.
        threshold_count (int): Minimum count to include a pair.

    Returns:
        tuple: The graph and node positions.
    """
    G = create_weighted_graph(pair_count, dict_names_id, threshold_count)
    pagerank_scores = nx.pagerank(G, weight='weight')

    nodes = list(G.nodes())
    pos_dict = generate_unique_positions(nodes, width=1, height=1, min_dist=0.1)

    pos = {node: (x, y) for node, (x, y) in zip(nodes, pos_dict.values())}

    min_node_size = 100
    scaling_factor = 5000
    node_sizes = [max(pagerank_scores[node] * scaling_factor, min_node_size) for node in
                  nodes]
    min_value = min(node_sizes)
    max_value = max(node_sizes)
    norm = plt.Normalize(vmin=min_value, vmax=max_value)
    colormap = plt.colormaps['winter_r']
    colors = colormap(norm(node_sizes))

    plt.figure(figsize=(20, 20))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes)
    edges = G.edges(data=True)
    weights = [edge[2]["weight"] for edge in edges]
    custom_cmap = LinearSegmentedColormap.from_list("custom_grey", ['#B2BEB5', '#000000'])
    norm = Normalize(vmin=min(weights), vmax=max(weights))

    edge_colors = [custom_cmap(norm(w)) for w in weights]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, edge_color=edge_colors, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=5, font_color='black', font_weight='bold')

    plt.title("Character Relationship Network")
    plt.axis('off')
    plt.show()

    return G, pos


def plot_louvain_communities(G: nx.Graph, pos: dict, colormap_name: str = 'tab10', resolution: float = 1.0):
    """
    Detects communities within a graph using the Louvain method and plots the resulting communities.
    
    Args:
        G (nx.Graph): The graph to analyze.
        pos (dict): A dictionary of node positions.
        colormap_name (str): The name of the colormap to use for community colors.
        resolution (float): Resolution parameter for Louvain community detection.
        
    Returns:
        dict: A dictionary mapping nodes to their corresponding communities.
    """
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

    return partition


def create_weighted_graph(pair_counts: dict, dict_names_id: dict, threshold_count: int = 3) -> nx.Graph:
    """
    Creates a weighted graph based on character pair counts, excluding pairs below a threshold count.
    
    Args:
        pair_counts (dict): A dictionary of character pairs and their interaction counts.
        dict_names_id (dict): A dictionary mapping character names to IDs.
        threshold_count (int): The minimum count required for a pair to be included in the graph. 
                                default value = 3.
        
    Returns:
        nx.Graph: A NetworkX graph with characters as nodes and interaction counts as edge weights.
    """
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


def plot_leiden_communities(G: nx.Graph, pos: dict, colormap_name: str = 'tab10', resolution: float = 1.0) -> dict:
    """
    Detects communities within a graph using the Leiden method and plots the resulting communities.
    
    Args:
        G (nx.Graph): The graph to analyze.
        pos (dict): A dictionary of node positions.
        colormap_name (str): The name of the colormap to use for community colors.
        resolution (float): Resolution parameter for Leiden community detection.
        
    Returns:
        dict: A dictionary mapping nodes to their corresponding communities.
    """
    G_ig = cdlib.nx_to_igraph(G)

    partition = la.find_partition(G_ig, la.RBConfigurationVertexPartition, weights=G_ig.es['weight'],
                                  resolution_parameter=resolution)
    membership = partition.membership
    node_communities = {node: membership[idx] for idx, node in enumerate(G.nodes())}
    num_communities = len(set(node_communities.values()))

    cmap = plt.colormaps[colormap_name]

    fig, ax = plt.subplots(figsize=(20, 20))
    for spine in ax.spines.values():
        spine.set_visible(False)

    nx.draw_networkx_nodes(G, pos, node_color=[node_communities[node] for node in G.nodes()],
                           node_size=3000, cmap=cmap, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", font_color='black')

    plt.title(
        f"Leiden Community Detection with Weights - {num_communities} Communities Detected (Resolution={resolution})")
    plt.show()

    return node_communities


def calc_semantic(indices: list, indices_to_semantics: list) -> float:
    """
    Calculates the sum of semantic values for given indices.
    
    Args:
        indices (list): A list of indices.
        indices_to_semantics (list): A list mapping indices to their semantic values.
        
    Returns:
        float: The sum of semantic values.
    """
    sum_semantic = 0
    for i in indices:
        sum_semantic += indices_to_semantics[i]
    return sum_semantic


def plot_sentiment_relations(pair_counts: dict, dict_names_id: dict, pairs_to_indices: dict, indices_to_semantics: dict,
                             threshold_count: int = 30,
                             model: str = "cardiffnlp/twitter-roberta-base-sentiment") -> dict:
    """
    Plots a graph of character relationships with edges colored based on sentiment analysis.
    
    Args:
        pair_counts (dict): A dictionary of character pairs and their interaction counts.
        dict_names_id (dict): A dictionary mapping character names to IDs.
        pairs_to_indices (dict): A dictionary mapping pairs to their corresponding indices.
        indices_to_semantics (dict): A dictionary mapping indices to their semantic values.
        threshold_count (int): The minimum count required for a pair to be included in the graph.
        model (str): The name of the sentiment analysis model to use.
        
    Returns:
        dict: A dictionary mapping character pairs to their sentiment values.
    """
    G = nx.Graph()
    pairs_model = {}

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
            pairs_model[(name1, name2)] = sum_semantic

    nodes = list(G.nodes())
    pos_dict = generate_unique_positions(nodes, width=1, height=1, min_dist=0.1)
    pos = {node: (x, y) for node, (x, y) in zip(nodes, pos_dict.values())}
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=300)
    edges = G.edges(data=True)
    semantics = [edge[2]["semantic"] for edge in edges]
    custom_cmap = LinearSegmentedColormap.from_list("custom_blue_red", ['#0000FF', '#FF0000'], N=6)
    norm = Normalize(vmin=min(semantics), vmax=max(semantics))
    edge_colors = [custom_cmap(norm(s)) for s in semantics]

    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color=edge_colors, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=5, font_color='black', font_weight='bold')

    cax = fig.add_axes([0.2, 0.05, 0.6, 0.02])  # [left, bottom, width, height]

    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks([])
    cbar.ax.text(0, 1.5, 'Negative Relationship', va='bottom', ha='left', fontsize=10, color='#0000FF',
                 transform=cbar.ax.transAxes)
    cbar.ax.text(1, 1.5, 'Positive Relationship', va='bottom', ha='right', fontsize=10, color='#FF0000',
                 transform=cbar.ax.transAxes)

    ax.set_title(f"Sentiment Relations Plot - Model: {model}", fontsize=12, fontweight='bold')

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    return pairs_model




def analyze_sentiment_advanced(set_sentences: set, df_sentences: pd.DataFrame, model: str) -> dict:
    """
    Performs sentiment analysis on a set of sentences using a specified model and returns a dictionary of sentiment scores.

    Args:
        set_sentences (set): A set of sentence indices.
        df_sentences (pd.DataFrame): A DataFrame containing sentences.
        model (str): The name of the sentiment analysis model to use.

    Returns:
        dict: A dictionary mapping sentence indices to sentiment scores.
    """
    device = 0 if torch.cuda.is_available() else -1

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        device=device
    )

    sentences = df_sentences.loc[list(set_sentences), 'sentence'].tolist()
    dataset = Dataset.from_dict({"sentence": sentences})

    def sentiment_analysis_batch(examples: dict) -> dict:
        results = sentiment_pipeline(examples["sentence"])
        return {"label": [result['label'] for result in results]}

    results = dataset.map(sentiment_analysis_batch, batched=True, batch_size=64)

    sentiment_dict = {}
    for idx, label in zip(set_sentences, results['label']):
        if label == 'LABEL_2':  # Positive sentiment
            sentiment_dict[idx] = 1
        elif label == 'LABEL_0':  # Negative sentiment
            sentiment_dict[idx] = -1
        else:  # Neutral sentiment
            sentiment_dict[idx] = 0

    return sentiment_dict


def analyze_sentiment_textblob(set_sentences: set, df_sentences: pd.DataFrame) -> dict:
    """
    Performs sentiment analysis using TextBlob and returns a dictionary of sentiment scores.
    
    Args:
        set_sentences (set): A set of sentence indices.
        df_sentences (pd.DataFrame): A DataFrame containing all book sentences.
        
    Returns:
        dict: A dictionary mapping sentence indices to sentiment scores.
    """
    sentiment_dict = {}

    for index in set_sentences:
        sentence = df_sentences.loc[index, 'sentence']
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity

        if polarity >= 0.05:
            sentiment_dict[index] = 1  # Positive sentiment
        elif polarity <= -0.05:
            sentiment_dict[index] = -1  # Negative sentiment
        else:
            sentiment_dict[index] = 0  # Neutral sentiment

    return sentiment_dict


def calc_sentiment_accuracy(experts_tagging: dict[tuple[str, str], int], model_results, tolerance=0.25) -> float:
    """
    Calculates the accuracy of a sentiment model by comparing its results to expert tagging within a specified tolerance.
    
    Args:
        experts_tagging (dict): A dictionary of expert-tagged character pairs and their sentiment scores.
        model_results (dict): A dictionary of model-generated sentiment scores for character pairs.
        tolerance (float): The tolerance range within which a model's result is considered accurate.
        
    Returns:
        float: The accuracy of the model.
    """
    max_model_value = max(model_results.values())
    min_model_value = min(model_results.values())
    mid_model_value = (max_model_value + min_model_value) / 2

    tolerance_range = (max_model_value - min_model_value) * tolerance

    expert_adjusted = {}
    for pair, value in experts_tagging.items():
        if value == 1:
            expert_adjusted[pair] = max_model_value
        elif value == -1:
            expert_adjusted[pair] = min_model_value
        elif value == 0:
            expert_adjusted[pair] = mid_model_value
        elif value == 0.5:
            expert_adjusted[pair] = (3 * max_model_value + min_model_value) / 4
        elif value == -0.5:
            expert_adjusted[pair] = (max_model_value + 3 * min_model_value) / 4

    correct_predictions = 0

    for pair in expert_adjusted:
        if pair in model_results:
            expert_value = expert_adjusted[pair]
            model_value = model_results[pair]

            if expert_value - tolerance_range <= model_value <= expert_value + tolerance_range:
                correct_predictions += 1

    accuracy = correct_predictions / len(expert_adjusted)
    return accuracy


def evaluate_model_against_experts(experts_tagging: dict, model_results, tolerance=0.25) -> float:
    """
    Evaluates the performance of a sentiment model by comparing its results to expert tagging and calculates accuracy.
    
    Args:
        experts_tagging (dict): A dictionary of expert-tagged character pairs and their sentiment scores.
        model_results (dict): A dictionary of model-generated sentiment scores for character pairs.
        tolerance (float): The tolerance range within which a model's result is considered accurate.
        
    Returns:
        float: The accuracy of the model.
    """
    correct_predictions = 0

    for pair in experts_tagging:
        if pair in model_results:
            expert_value = experts_tagging[pair]
            model_value = model_results[pair]

            if expert_value - tolerance <= model_value <= expert_value + tolerance:
                correct_predictions += 1

    accuracy = correct_predictions / len(experts_tagging)
    return accuracy


def convert_dict_to_communities(partition: dict) -> list:
    """
    Converts a partition dictionary into a list of communities.
    
    Args:
        partition (dict): A dictionary mapping nodes to their respective communities.
        
    Returns:
        list: A list of sets, where each set represents a community.
    """
    community_dict = {}

    for node, community in partition.items():
        if community not in community_dict:
            community_dict[community] = set()
        community_dict[community].add(node)

    communities = list(community_dict.values())

    return communities


def eval_community_detection(G: nx.Graph, partition: dict):
    """
    Evaluates the quality of community detection on a graph.
    
    Args:
        G (nx.Graph): The graph on which community detection was performed.
        partition (dict): A dictionary mapping nodes to their respective communities.
        
    Returns:
        float: The quality score of the partition.
    """
    communities = convert_dict_to_communities(partition)
    return partition_quality(G, communities)


def evaluate_model_precision_recall_f1(experts_tagging: dict, model_results: dict) -> dict:
    """
    Calculates precision, recall, and F1 score for a sentiment model based on expert tagging.
    
    Args:
        experts_tagging (dict): A dictionary of expert-tagged character pairs and their sentiment scores.
        model_results (dict): A dictionary of model-generated sentiment scores for character pairs.
        
    Returns:
        dict: A dictionary containing precision, recall, and F1 score.
    """
    y_true = []
    y_pred = []

    for pair, expert_value in experts_tagging.items():
        model_value = model_results[pair]

        if expert_value > 0:
            y_true.append(1)
        elif expert_value < 0:
            y_true.append(0)

        if model_value > 0:
            y_pred.append(1)  # Model predicts positive
        else:
            y_pred.append(0)  # Model predicts negative

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {"precision": precision, "recall": recall, "f1": f1}


def save_pair_counts(pair_counts: dict, path_pair_counts: str) -> None:
    """
    Saves pair counts to a pickle file.
    
    Args:
        pair_counts (dict): A dictionary of character pairs and their interaction counts.
        path_pair_counts (str): The file path to save the pair counts.
    """
    with open(path_pair_counts, "wb") as f:
        pickle.dump(pair_counts, f)


def print_model_evaluation(model: str, model_results: dict) -> None:
    """
    Prints the evaluation metrics for a sentiment model based on expert tagging.
    
    Args:
        model (str): The name of the sentiment analysis model.
        model_results (dict): A dictionary of model-generated sentiment scores for character pairs.
    """
    experts_tagging = {
        ('Harry Potter', 'Ron Weasley'): 1,
        ('Hermione Granger', 'Harry Potter'): 1,
        ('Hermione Granger', 'Ron Weasley'): 0.5,
        ('Albus Dumbledore', 'Harry Potter'): 1,
        ('Harry Potter', 'Rubeus Hagrid'): 1,
        ('Harry Potter', 'Severus Snape'): -1,
        ('Fred Weasley', 'George Weasley'): 1,
        ('Sirius Black', 'Harry Potter'): 1,
        ('Ginny Weasley', 'Harry Potter'): 1,
        ('Ginny Weasley', 'Ron Weasley'): 0.5,
        ('Fred Weasley', 'Harry Potter'): 1,
        ('Vernon Dursley', 'Harry Potter'): -1,
        ('George Weasley', 'Harry Potter'): 1,
        ('Fred Weasley', 'Ron Weasley'): 1,
        ('Harry Potter', 'Remus Lupin'): 1,
        ('Dudley Dursley', 'Harry Potter'): -1,
        ('Petunia Dursley', 'Vernon Dursley'): 1,
        ('Harry Potter', 'Neville Longbottom'): 1,
        ('George Weasley', 'Ron Weasley'): 1,
        ('Harry Potter', 'Minerva McGonagall'): 0.5,
        ('Dudley Dursley', 'Vernon Dursley'): 1,
        ('Hermione Granger', 'Rubeus Hagrid'): 1,
        ('Dudley Dursley', 'Petunia Dursley'): 1,
        ('Albus Dumbledore', 'Ron Weasley'): 0.5,
        ('Petunia Dursley', 'Harry Potter'): -1,
        ('Ron Weasley', 'Rubeus Hagrid'): 1,
        ('Albus Dumbledore', 'Hermione Granger'): 0.5,
        ('Dolores Umbridge', 'Harry Potter'): -1,
        ('Percy Weasley', 'Ron Weasley'): -0.5,
        ('Bill Weasley', 'Ron Weasley'): 1,
        ('Draco Malfoy', 'Harry Potter'): -1
    }

    accuracy = calc_sentiment_accuracy(experts_tagging, model_results)
    print(f"Accuracy for model {model}:\n", accuracy)

    metrics = evaluate_model_precision_recall_f1(experts_tagging, model_results)
    print(f"Metrics for model {model}:")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1']}")


def get_pair_counts_from_pickle(path_pair_counts: str) -> dict:
    """
    Loads pair counts from a pickle file.
    
    Args:
        path_pair_counts (str): The file path to load the pair counts from.
        
    Returns:
        dict: A dictionary of character pairs and their interaction counts.
    """
    with open(path_pair_counts, "rb") as f:
        pair_counts = pickle.load(f)
    return pair_counts


def save_dict_names_id(dict_names_id: dict, path_names_id: str) -> None:
    """
    Saves a dictionary of character names and IDs to a pickle file.
    
    Args:
        dict_names_id (dict): A dictionary mapping character names to IDs.
        path_names_id (str): The file path to save the dictionary.
    """
    with open(path_names_id, "wb") as f:
        pickle.dump(dict_names_id, f)


def get_dict_names_id_from_pickle(path_names_id: str) -> dict:
    """
    Loads a dictionary of character names and IDs from a pickle file.
    
    Args:
        path_names_id (str): The file path to load the dictionary from.
        
    Returns:
        dict: A dictionary mapping character names to IDs.
    """
    with open(path_names_id, "rb") as f:
        dict_names_id = pickle.load(f)
    return dict_names_id


def save_pair_sentences(pair_sentences: dict, set_sentences: set, path_pair_sentences: str,
                        path_set_sentences: str) -> None:
    """
    Saves pair sentences and set sentences to pickle files.
    
    Args:
        pair_sentences (dict): A dictionary mapping character pairs to sentence indices.
        set_sentences (set): A set of sentence indices.
        path_pair_sentences (str): The file path to save the pair sentences.
        path_set_sentences (str): The file path to save the set sentences.
    """
    with open(path_pair_sentences, "wb") as f:
        pickle.dump(pair_sentences, f)
    with open(path_set_sentences, "wb") as f:
        pickle.dump(set_sentences, f)


def get_pair_sentences_from_pickle(path_pair_sentences: str, path_set_sentences: str) -> tuple[dict, set]:
    """
    Loads pair sentences and set sentences from pickle files.
    
    Args:
        path_pair_sentences (str): The file path to load the pair sentences from.
        path_set_sentences (str): The file path to load the set sentences from.
        
    Returns:
        tuple: A tuple containing the pair sentences and set sentences.
    """
    with open(path_pair_sentences, "rb") as f:
        pair_sentences = pickle.load(f)
    with open(path_set_sentences, "rb") as f:
        set_sentences = pickle.load(f)
    return pair_sentences, set_sentences


def main(paths) -> None:
    df_sentences = pd.read_csv(paths["sentences"])
    # df_characters = pd.read_csv(paths["characters"])
    # dict_names_id = create_dict_names_id(df_characters)
    # dict_names_id = remove_characters_below_threshold(dict_names_id, df_sentences, threshold=16)
    # save_dict_names_id(dict_names_id, paths["names_id"])
    # pair_sentences, set_sentences = create_pair_sentences(df_sentences, dict_names_id)
    # save_pair_sentences(pair_sentences, set_sentences, paths["pair_sentences"], paths["set_sentences"])
    # pair_counts = create_dict_connections(df_sentences, dict_names_id)
    # save_pair_counts(pair_counts, paths["pair_counts"])

    # get data from the pickle files:
    dict_names_id = get_dict_names_id_from_pickle(paths["names_id"])
    pair_counts = get_pair_counts_from_pickle(paths["pair_counts"])
    pair_sentences, set_sentences = get_pair_sentences_from_pickle(paths["pair_sentences"], paths["set_sentences"])

    # plots that represent the character relationships:
    # plot_simple_connections(pair_counts, dict_names_id, threshold_count=10)
    G, pos = plot_page_rank(pair_counts, dict_names_id, threshold_count=30)
    partition = plot_louvain_communities(G, pos, resolution=1.7)
    node_communities = plot_leiden_communities(G, pos, resolution=1.7)

    # run sentiment analysis on the sentences:
    # model = "cardiffnlp/twitter-roberta-base-sentiment"
    # indices_to_semantics = analyze_sentiment_advanced(set_sentences, df_sentences, model)
    # model_results = plot_sentiment_relations(pair_counts, dict_names_id, pair_sentences, indices_to_semantics,
    #                                          threshold_count=250,
    #                                          model=model)
    # print_model_evaluation(model, model_results)
    #
    # model = "TextBlob"
    # indices_to_semantics = analyze_sentiment_textblob(set_sentences, df_sentences)
    # model_results = plot_sentiment_relations(pair_counts, dict_names_id, pair_sentences, indices_to_semantics,
    #                                          threshold_count=250,
    #                                          model=model)
    # print_model_evaluation(model, model_results)
    coverage_louvain, performance_louvain = eval_community_detection(G, partition)
    coverage_leiden, performance_leiden = eval_community_detection(G, node_communities)
    print(
        "For Louvain's partition the coverage is " + coverage_louvain + " and the performance is " + performance_louvain)
    print("For Leiden's partition the coverage is " + coverage_leiden + " and the performance is " + performance_leiden)


if __name__ == "__main__":
    main(PATHS)
