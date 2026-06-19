from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import os
import networkx as nx
from pydantic import BaseModel
from .graph_engine import (
    load_graph,
    load_critical_junctions,
    analyze_network,
    draw_junctions,
    get_current_graph,
    simulate_blockage,
)
from .model import predict_mask

class BlockageRequest(BaseModel):

    junction_x: int
    junction_y: int


class RouteRequest(BaseModel):

    source_x: int
    source_y: int

    target_x: int
    target_y: int


os.makedirs(
    "outputs",
    exist_ok=True
)

app = FastAPI(
    title="Route Resilience API"
)

@app.get("/")
def home():

    return {
        "message":
        "Route Resilience Running"
    }


@app.get("/health")
def health():

    return {
        "status":"ok"
    }


@app.get("/graph-stats")
def graph_stats():

    G = load_graph()

    return {

        "nodes":
        G.number_of_nodes(),

        "edges":
        G.number_of_edges(),

        "components":
        1
    }

@app.get("/critical-junctions")
def critical_junctions():

    data = load_critical_junctions()

    result = []

    for node, score in data[:10]:

        result.append({
            "x": int(node[0]),
            "y": int(node[1]),
            "score": float(score)
        })

    return {
        "count": int(len(data)),
        "top_10": result
    }

@app.get("/resilience")
def resilience():

    return {
        "before_components": 1,
        "after_components": 3,
        "resilience_score": 0.333
    }
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    nparr = np.frombuffer(
        contents,
        np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    pred_binary = predict_mask(img)

    road_pixels = int(
        pred_binary.sum()
    )

    road_percent = float(
        road_pixels /
        pred_binary.size * 100
    )

    graph_data = analyze_network(
        pred_binary
    )
    overlay = draw_junctions(
        img,
        graph_data["critical_junctions"]
    )
    cv2.imwrite(
        "outputs/result.jpg",
        overlay
    )
    saved = cv2.imwrite(
        "../outputs/result.jpg",
        overlay
    )
    cv2.imwrite(
        "../outputs/debug_mask.png",
        pred_binary * 255
    )
    cv2.imwrite(
        "../outputs/original.png",
        img
    )
    cv2.imwrite(
        "../outputs/morphology.jpg",
        pred_binary * 255
    )

    print("Image Saved:", saved)
    print("Shape:", overlay.shape)

    return {

        "road_pixels":
            road_pixels,

        "road_percent":
            road_percent,

        **graph_data,

        "overlay":
            "outputs/result.jpg"
    }

@app.post("/route")
def route(req: RouteRequest):

    G = get_current_graph()

    if G is None:

        return {
            "error":
            "Run /predict first"
        }

    source = (
        req.source_x,
        req.source_y
    )

    target = (
        req.target_x,
        req.target_y
    )

    source_node = min(
        G.nodes(),
        key=lambda n:
        (n[0]-source[0])**2 +
        (n[1]-source[1])**2
    )

    target_node = min(
        G.nodes(),
        key=lambda n:
        (n[0]-target[0])**2 +
        (n[1]-target[1])**2
    )

    try:

        path = nx.shortest_path(
            G,
            source_node,
            target_node
        )

        return {

            "route_found": True,

            "path_length": len(path),

            "source_node": source_node,

            "target_node": target_node,

            "path": path[:50]
        }

    except:

        return {
            "route_found": False
        }


@app.post("/simulate-blockage")
def blockage(req: BlockageRequest):

    print("REQUEST RECEIVED")

    G = load_graph()

    print("Nodes:", G.number_of_nodes())

    result = simulate_blockage(
        G,
        req.junction_x,
        req.junction_y
    )

    print(result)

    return result