from plot_connections import *
from dynamic_utils import *
from utils import *
import tqdm


def reverse_dict(dict_pair_sentences: dict) -> dict:
    """
    Reverse the keys and values of a dictionary where the values are lists of sentences.

    Args:
        dict_pair_sentences (dict): A dictionary where the keys are IDs and the values are lists of sentences.

    Returns:
        dict: A dictionary where the keys are sentences and the values are lists of IDs.

    """
    dict_sentence_pairs = {}
    for ids, sentences in dict_pair_sentences.items():
        for sentence in sentences:
            if sentence in dict_sentence_pairs:
                dict_sentence_pairs[sentence].append(ids)
            else:
                dict_sentence_pairs[sentence] = [ids]
    return dict_sentence_pairs


def create_dynamic_connections_db(pair_sentences: dict) -> list:
    """
    Create dynamic connections database based on pair_sentences. 
    This database represents the changes in connections between characters as the books progress.

    Args:
        pair_sentences (dict): A dictionary containing pairs of sentences and their corresponding IDs.

    Returns:
        list: A list of tuples representing the dynamic connections database. Each tuple contains a sentence and a dictionary of connections.

    """
    dict_pair_sentences = reverse_dict(pair_sentences)
    dict_pair_sentences_list = sorted(
        [(sentence, ids) for sentence, ids in dict_pair_sentences.items()],
        key=lambda x: x[0],
    )
    temp_connections = {}
    dynamic_connections = []
    for sentence, ids in dict_pair_sentences_list:
        for id in ids:
            if id in temp_connections:
                temp_connections[id] += 1
            else:
                temp_connections[id] = 1
        dynamic_connections.append((sentence, temp_connections.copy()))
    return dynamic_connections

# def get_edges(G: nx.Graph, positions: dict, pagerank_scores: dict, node_trace: go.Scatter) -> go.Scatter:
#     edge_traces = []
    
#     for edge in G.edges(data=True):
#         x0, y0 = positions[edge[0]]
#         x1, y1 = positions[edge[1]]
        
#         # Determine which node has higher PageRank
#         if pagerank_scores[edge[0]] > pagerank_scores[edge[1]]:
#             color = node_trace.marker.color[list(G.nodes()).index(edge[0])]
#         else:
#             color = node_trace.marker.color[list(G.nodes()).index(edge[1])]
        
#         # Get edge weight
#         weight = edge[2].get('weight', 1)
        
#         edge_trace = go.Scatter(
#             x=[x0, x1, weight],
#             y=[y0, y1, weight],
#             line=dict(width=0.5, color="#888"),
#             hoverinfo='text',
#             mode='lines',
#             text=f'Weight: {weight}',
#             opacity=0.7
#         )
#         # edge_traces.marker.line.color = color
#         edge_traces.append(edge_trace)
    
    
#     return edge_traces

def get_edges(G: nx.Graph, positions: dict) -> go.Scatter:
    """
    Generate a scatter plot trace for the edges of a graph.

    Parameters:
        G (nx.Graph): The input graph.
        positions (dict): A dictionary mapping nodes to their positions.

    Returns:
        go.Scatter: The scatter plot trace for the edges.
    """

    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    return edge_trace


def get_nodes(G: nx.Graph, positions: dict) -> go.Scatter:
    """
    Generate a scatter plot of nodes in a graph.

    Parameters:
        G (nx.Graph): The graph object.
        positions (dict): A dictionary mapping nodes to their positions.

    Returns:
        go.Scatter: The scatter plot of nodes.
    """
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        text=list(G.nodes()),
        textposition="top center",
        textfont=dict(family="sans serif", size=11, color="black"),
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale="Jet_r",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15, title="Importance", xanchor="left", titleside="right"
            ),
            line_width=2,
        ),
    )
    return node_trace


def plot_network(edge_trace: go.Scatter, node_trace: go.Scatter, progress: str) -> go.Frame:
    """
    Creates a frame for a character relationship network plot.

    Parameters:
    - edge_trace (go.Scatter): The scatter trace for edges in the network.
    - node_trace (go.Scatter): The scatter trace for nodes in the network.
    - progress (str): A string indicating the progress (meaning point in the story) of the network plot.

    Returns:
    - go.Frame: The frame object for the character relationship network plot.
    """
    return go.Frame(
        data=[edge_trace, node_trace],
        name=progress,
        layout=go.Layout(title=f'Character Relationship Network - {progress}<br><sup>Node color and size correspond to "Importance", which relates to both pagerank and connectivity</sup>')
    )


def dynamic_pagerank_plot(pair_sentences: dict, dict_names_id: dict) -> None:
    """
    Generates a dynamic pagerank plot based on the and dictionary of names and IDs.
    Args:
        pair_sentences (dict): A dictionary mapping each pair of characters to a list containing all sentences they appear in.
        dict_names_id (dict): A dictionary mapping names to IDs.
    Returns:
        None
    """
    dynamic_connections = create_dynamic_connections_db(pair_sentences)
    df_sentences = pd.read_csv(PATHS_TOMER["sentences"])
    dict_sentences = df_sentences.to_dict('index')
    sentences_per_chapter = df_sentences.groupby(["book", "chapter"])["sentence"].count()
    per_chapter_dict = sentences_per_chapter.to_dict()
    sentences_per_book = df_sentences.groupby(["book"])["sentence"].count()
    per_book_dict = sentences_per_book.to_dict()
    per_book_dict = {int(key.split(" ")[1][0]): value for key, value in per_book_dict.items()}

    pos_dict_meta = {}
    frames = []
    sentences = []
    last_sentence = 0
    last_chapter = 0
    last_book = ""
    last_sentence_in_book = {}
    for i, (sentence, connections) in tqdm.tqdm(enumerate(dynamic_connections)):
        book = dict_sentences[sentence]["book"]
        chapter = dict_sentences[sentence]["chapter"]
        if last_chapter != chapter:
            last_chapter = chapter
            last_sentence = sentence - 1 
        chapter_progress = (sentence - last_sentence + 1) / per_chapter_dict[(book, chapter)] * 100
        sentences.append(f"{round(chapter_progress, 1)}% of Chapter {chapter} of {book}")
        if last_book != book:
            last_book = book
            pos_dict_meta = {}
            last_sentence_in_book[int(book.split(" ")[1][0])] = last_sentence

        G = create_weighted_graph(connections, dict_names_id, threshold_count=4*int(book.split(" ")[1][0]))
        pagerank_scores = nx.pagerank(G, weight="weight")
        nodes_loc = [node for node in G.nodes() if node not in pos_dict_meta.keys()]
        pos_dict = generate_unique_positions(nodes_loc, width=1, height=1, min_dist=0.1)
        pos = {node: (x, y) for node, (x, y) in zip(nodes_loc, pos_dict.values())}
        for node, pos in pos.items():
            if node in pos_dict_meta:
                continue
            else:
                pos_dict_meta[node] = pos
        if i % 20 == 0:
            edge_trace = get_edges(G, pos_dict_meta)
            node_trace = get_nodes(G, pos_dict_meta)

            min_node_size = 10  
            scaling_factor = (
                250  
            )
            node_sizes = [
                max(pagerank_scores[node] * scaling_factor, min_node_size)
                for node in G.nodes
            ]  
            node_adjacencies = []
            node_text = []
            for node, adjacencies in enumerate(G.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
                node_text.append(adjacencies[0])

            node_trace.marker.color = [ 
                node_sizes[i] + node_adjacencies[i] for i in range(len(node_sizes))
            ] 
            node_trace.text = node_text
            node_trace.marker.size = [ 
                node_sizes[i] + node_adjacencies[i] for i in range(len(node_sizes))
            ] 
            

            fig = plot_network(edge_trace, node_trace, sentences[-1])
            frames.append(fig)
        

    x_coords, y_coords = zip(*pos_dict_meta.values())
    x_min, x_max = min(x_coords),  max(x_coords)
    y_min, y_max =  min(y_coords), max(y_coords)
    padding = 0.1
    x_range = [x_min - padding, x_max + padding]
    y_range = [y_min - padding, y_max + padding]
 
    final_fig = go.Figure(
                data= frames[0].data,
                layout=go.Layout(
                    title='Character Relationship Network<br><sup>Node color and size correspond to "Importance", which relates to both pagerank and connectivity</sup>',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[dict(text="To speed up or slow down, press the <b>Pause</b> button before pressing another button", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.035)],
                    xaxis=dict(range=x_range, showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(range=y_range, showgrid=False, zeroline=False, showticklabels=False),
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[AnimationButtons.play(frame_duration=500), AnimationButtons.pause(),AnimationButtons.regular_speed(frame_duration=500), AnimationButtons.speed_up(250), AnimationButtons.speed_up2(100),  AnimationButtons.speed_up3(20), AnimationButtons.slow_down(1000)], 
                    )],
                ),
                frames=frames
            )
    final_fig.show()
    frames_to_gif(frames, f"images/dynamic_relationship_graph_all_books - 19 sample.gif")
    


if __name__ == "__main__":
    dict_names_id = get_dict_names_id_from_pickle(PATHS_TOMER["names_id"])
    pair_counts = get_pair_counts_from_pickle(PATHS_TOMER["pair_counts"])
    pair_sentences, set_sentences = get_pair_sentences_from_pickle(
        PATHS_TOMER["pair_sentences"], PATHS_TOMER["set_sentences"]
    )
    dynamic_pagerank_plot(pair_sentences, dict_names_id)
    # n = 1000
    # test = [i for i in range(n)]
    # import time
    # print("Testing regular func")
    # start = time.time()
    # generate_unique_positions(test, width=1, height=1, min_dist=0.1)
    # end = time.time()
    # print(f"Running regular func on {n} samples took {end - start}")
    # print("Testing my func")
    # start = time.time()
    # my_generate_unique_positions(test, width=1, height=1, min_dist=0.1)
    # end = time.time()
    # print(f"Running my func on {n} samples took {end - start}")

