import os
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize
import pickle
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_GRAPH = None

def get_current_graph():
    global CURRENT_GRAPH
    return CURRENT_GRAPH

def draw_junctions(image, junctions):

    output = image.copy()

    print("Junctions Found:", len(junctions))

    for j in junctions:

        x = int(j["x"])
        y = int(j["y"])

        print(x, y)

        cv2.circle(
            output,
            (x, y),
            25,
            (0, 0, 255),
            -1
        )
        cv2.putText(
            output,
            "J",
            (x+10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

    return output


def load_graph():

    graph_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "graph.pkl")
    )

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    return G


def load_critical_junctions():

    critical_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "critical_junctions.pkl")
    )

    with open(critical_path, "rb") as f:
        data = pickle.load(f)

    return data
def simulate_blockage(G, x, y):

    if G is None:
        return None

    target = min(
        G.nodes(),
        key=lambda n:
        (n[0] - x)**2 +
        (n[1] - y)**2
    )

    before = nx.number_connected_components(G)

    temp = G.copy()

    if target in temp:
        temp.remove_node(target)

    after = nx.number_connected_components(temp)

    resilience = before / after
    return {

        "blocked_node": {
            "x": int(target[0]),
            "y": int(target[1])
        },

        "before_components": int(before),

        "after_components": int(after),

        "resilience_score": float(resilience)
    }
def analyze_network(pred_binary):

    kernel = np.ones((15,15), np.uint8)

    pred_binary = cv2.dilate(
        pred_binary.astype(np.uint8),
        kernel,
        iterations=1
    )

    pred_binary = cv2.erode(
        pred_binary,
        kernel,
        iterations=1
    )

    skeleton = skeletonize(
        pred_binary.astype(bool)
    )

    G = nx.Graph()

    rows, cols = skeleton.shape

    for y in range(rows):
        for x in range(cols):

            if not skeleton[y, x]:
                continue

            G.add_node((x, y))

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:

                    if dx == 0 and dy == 0:
                        continue

                    nx_ = x + dx
                    ny_ = y + dy

                    if (
                        0 <= nx_ < cols
                        and
                        0 <= ny_ < rows
                        and
                        skeleton[ny_, nx_]
                    ):
                        G.add_edge(
                            (x, y),
                            (nx_, ny_)
                        )
    if G.number_of_nodes() == 0:

        return {
            "graph_nodes": 0,
            "graph_edges": 0,
            "junction_count": 0,
            "resilience_score": 0,
            "skeleton_pixels": 0,
            "critical_junctions": []
        }
    
    
    largest = max(
        nx.connected_components(G),
        key=len
    )

    G_main = G.subgraph(
        largest
    ).copy()

    critical_junctions = []

    score = nx.betweenness_centrality(
        G_main,
        k=50,
        normalized=True
    )

    for node in G_main.nodes():

        if G_main.degree(node) >= 3:

            critical_junctions.append(
                (
                    node,
                    score.get(node, 0)
                )
            )

    critical_junctions.sort(
        key=lambda x: x[1],
        reverse=True
    )
    filtered = []

    for node, score in critical_junctions:

        keep = True

        for old_node, _ in filtered:

            dist = np.sqrt(
                (node[0]-old_node[0])**2 +
                (node[1]-old_node[1])**2
            )

            if dist < 60:
                keep = False
                break

        if keep:
            filtered.append(
                (node, score)
            )
    top_junctions = []

    for node, score in filtered[:10]:

        top_junctions.append({
            "x": int(node[0]),
            "y": int(node[1]),
            "score": float(score)
        })

    before = nx.number_connected_components(
        G_main
    )

    if len(critical_junctions) > 0:

        temp = G_main.copy()

        temp.remove_node(
            critical_junctions[0][0]
        )

        after = nx.number_connected_components(
            temp
        )

        resilience = before / after

    else:

        after = before
        resilience = 1.0

    global CURRENT_GRAPH
    CURRENT_GRAPH = G_main

    return {

    "graph_nodes":
        int(G_main.number_of_nodes()),

    "graph_edges":
        int(G_main.number_of_edges()),

    "junction_count":
        int(len(filtered)),

    "resilience_score":
        float(resilience),

    "skeleton_pixels":
        int(skeleton.sum()),

    "critical_junctions":
        top_junctions,
}
