from simple_plot_connections import *
from dynamic_utils import *
from utils import *


def reverse_dict(dict_pair_sentences: dict) -> dict:
    dict_sentence_pairs = {}
    for ids, sentences in dict_pair_sentences.items():
        for sentence in sentences:
            if sentence in dict_sentence_pairs:
                dict_sentence_pairs[sentence].append(ids)
            else:
                dict_sentence_pairs[sentence] = [ids]
    return dict_sentence_pairs


def create_dynamic_connections_db(pair_sentences: dict) -> list:
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


def get_edges(G: nx.Graph, positions: dict) -> go.Scatter:
    edge_x = []
    edge_y = []
    for edge in G.edges():
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
                thickness=15, title="Node Rank", xanchor="left", titleside="right"
            ),
            line_width=2,
        ),
    )
    return node_trace


def plot_network(edge_trace: go.Scatter, node_trace: go.Scatter, sentence: str) -> go.Frame:
    # fig = go.Figure(
    #     data=[edge_trace, node_trace],
    #     layout=go.Layout(
    #         title="<br>Character Relationship Network",
    #         titlefont_size=16,
    #         showlegend=False,
    #         hovermode="closest",
    #         margin=dict(b=20, l=5, r=5, t=40),
    #         annotations=[
    #             dict(
    #                 text="",
    #                 showarrow=False,
    #                 xref="paper",
    #                 yref="paper",
    #                 x=0.005,
    #                 y=-0.002,
    #             )
    #         ],
    #         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #     ),
    # )
    # fig.show()
    # return fig
    return go.Frame(
        data=[edge_trace, node_trace],
        name=sentence,
        layout=go.Layout(title=f"<br>Character Relationship Network - {sentence}")
    )


def dynamic_pagerank_plot(pair_counts, pair_sentences, dict_names_id, threshold_count) -> None:
    dynamic_connections = create_dynamic_connections_db(pair_sentences)
    df_sentences = pd.read_csv(PATHS["sentences"])
    dict_sentences = df_sentences.to_dict('index')
    sentences_per_chapter = df_sentences.groupby(["book", "chapter"])["sentence"].count()
    per_chapter_dict = sentences_per_chapter.to_dict()
    # print(f"{per_chapter_dict=}")

    # max_book_chapter = max([dict_sentences[i]["book_chapter"] for i in dict_sentences.keys()])
    pos_dict_meta = {}
    frames = []
    sentences = []
    last_sentence = 0
    last_chapter = 0
    for sentence, connections in dynamic_connections[:300]:
        # print(f"{sentence=}")
        # print("Connections are: ", connections)
        # print("pair_counts: ", pair_counts)
        G = create_weighted_graph(connections, dict_names_id, threshold_count)
        G = make_graph_sparse(G, fraction=0.2)
        pagerank_scores = nx.pagerank(G, weight="weight")
        ### Might create a problem with the positions, so should be checked if issues arise ########
        nodes_loc = [node for node in G.nodes() if node not in pos_dict_meta.keys()]
        pos_dict = generate_unique_positions(nodes_loc, width=1, height=1, min_dist=0.1)
        pos = {node: (x, y) for node, (x, y) in zip(nodes_loc, pos_dict.values())}
        for node, pos in pos.items():
            if node in pos_dict_meta:
                continue
            else:
                pos_dict_meta[node] = pos
        #######################################################################################
        edge_trace = get_edges(G, pos_dict_meta)
        node_trace = get_nodes(G, pos_dict_meta)

        min_node_size = 10  # Minimum size for nodes
        scaling_factor = (
            350  # Increase this factor to make size differences more noticeable
        )
        # Normalize PageRank scores for node size
        node_sizes = [
            max(pagerank_scores[node] * scaling_factor, min_node_size)
            for node in G.nodes
        ]  # Scale PageRank scores and apply minimum size

        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(G.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            # print(f"{node=}, {adjacencies[0]=}")
            node_text.append(adjacencies[0])

        node_trace.marker.color = node_sizes 
        node_trace.text = node_text
        node_trace.marker.size = [ 
            node_sizes[i] + node_adjacencies[i] for i in range(len(node_sizes))
        ]
        book = dict_sentences[sentence]["book"]
        chapter = dict_sentences[sentence]["chapter"]
        if last_chapter != chapter:
            last_chapter = chapter
            last_sentence = sentence - 1 
        chapter_progress = (sentence - last_sentence + 1) / per_chapter_dict[(book, chapter)] * 100
        sentences.append(f"{round(chapter_progress, 1)}% of Chapter {chapter} of {book}")
        fig = plot_network(edge_trace, node_trace, sentences[-1])
        frames.append(fig)
        

    x_coords, y_coords = zip(*pos_dict_meta.values())
    x_min, x_max = min(x_coords),  max(x_coords)
    y_min, y_max =  min(y_coords), max(y_coords)
    padding = 0.1
    x_range = [x_min - padding, x_max + padding]
    y_range = [y_min - padding, y_max + padding]
    
    final_fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Character Relationship Network",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(text="To speed up or slow down, press the <b>Pause</b> button before pressing another button", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                xaxis=dict(range=x_range, showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=y_range, showgrid=False, zeroline=False, showticklabels=False),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[AnimationButtons.play(frame_duration=500), AnimationButtons.pause(),AnimationButtons.regular_speed(frame_duration=500), AnimationButtons.speed_up(250), AnimationButtons.slow_down(1000),],  
                # direction="left",
                # pad={"r": 10, "t": 87},
                # showactive=False,
                # x=0.1,
                # xanchor="right",
                # y=0,
                # yanchor="top"
                )],
            # sliders=AnimationButtons.slider(sentences)
            ),
            frames=frames
        )
    final_fig.show()
  

    # animation_to_gif(
    #     fig, "dynamic_relationship_graph.gif", 1000, width=400, height=400
    # )


if __name__ == "__main__":
    dict_names_id = get_dict_names_id_from_pickle(PATHS["names_id"])
    pair_counts = get_pair_counts_from_pickle(PATHS["pair_counts"])
    pair_sentences, set_sentences = get_pair_sentences_from_pickle(
        PATHS["pair_sentences"], PATHS["set_sentences"]
    )
    dynamic_pagerank_plot(pair_counts, pair_sentences, dict_names_id, 0)
