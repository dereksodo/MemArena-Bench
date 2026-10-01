"""Social network construction with Dunbar-layered graphs.

Supports Watts-Strogatz (default), Erdős-Rényi, and Barabási-Albert models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from MASim.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DunbarLayerConfig:
    """Configuration for a single Dunbar layer."""
    size: int = 5             # target number of nodes in this layer
    beta: float = 0.3         # WS rewiring probability
    k: int = 4                # WS nearest-neighbor degree
    weight: float = 1.0       # base edge weight for this layer


@dataclass
class GraphConfig:
    """Full social graph configuration."""
    n_agents: int = 50
    model: str = "watts_strogatz"  # watts_strogatz | erdos_renyi | barabasi_albert
    seed: int = 42
    alpha: float = 0.33           # weight decay factor across layers
    layers: List[DunbarLayerConfig] = field(default_factory=lambda: [
        DunbarLayerConfig(size=5,   beta=0.30, k=4, weight=1.0),
        DunbarLayerConfig(size=15,  beta=0.15, k=4, weight=0.33),
        DunbarLayerConfig(size=50,  beta=0.08, k=6, weight=0.109),
        DunbarLayerConfig(size=150, beta=0.05, k=8, weight=0.036),
    ])
    # ER-specific
    er_p: float = 0.1
    # BA-specific
    ba_m: int = 3

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphConfig":
        layers = d.pop("layers", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if layers:
            cfg.layers = [DunbarLayerConfig(**lc) for lc in layers]
        return cfg


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_dunbar_graph(cfg: GraphConfig) -> nx.Graph:
    """Build a Dunbar-layered social graph.

    Each ego node gets layered neighborhoods with decreasing edge weights.
    The underlying topology is Watts-Strogatz within each layer.
    """
    rng = np.random.default_rng(cfg.seed)
    G = nx.Graph()

    # Add all agent nodes
    for i in range(cfg.n_agents):
        G.add_node(f"agent_{i:04d}", layer_memberships={})

    agent_ids = list(G.nodes)

    # For each agent, assign layered neighborhoods
    for ego_idx, ego in enumerate(agent_ids):
        others = [a for a in agent_ids if a != ego]
        rng.shuffle(others)

        assigned = 0
        for layer_idx, layer_cfg in enumerate(cfg.layers):
            layer_size = min(layer_cfg.size, len(others) - assigned)
            if layer_size <= 0:
                break

            layer_members = others[assigned : assigned + layer_size]
            assigned += layer_size

            weight = cfg.alpha ** layer_idx
            for member in layer_members:
                if G.has_edge(ego, member):
                    # Take the stronger weight
                    existing = G[ego][member]["weight"]
                    G[ego][member]["weight"] = max(existing, weight)
                    # Keep the closer layer
                    existing_layer = G[ego][member]["layer"]
                    G[ego][member]["layer"] = min(existing_layer, layer_idx + 1)
                else:
                    G.add_edge(ego, member, weight=weight, layer=layer_idx + 1)

    # Apply WS-like rewiring per layer
    _apply_ws_rewiring(G, cfg, rng)

    log.info(
        "Built Dunbar graph: %d nodes, %d edges, CC=%.3f, avg_path=%.3f",
        G.number_of_nodes(),
        G.number_of_edges(),
        nx.average_clustering(G) if G.number_of_edges() > 0 else 0,
        _safe_avg_path(G),
    )
    return G


def _apply_ws_rewiring(G: nx.Graph, cfg: GraphConfig, rng: np.random.Generator) -> None:
    """Rewire edges with layer-specific beta probabilities (Watts-Strogatz style)."""
    edges = list(G.edges(data=True))
    nodes = list(G.nodes)

    for u, v, data in edges:
        layer = data.get("layer", 1)
        if layer - 1 >= len(cfg.layers):
            continue
        beta = cfg.layers[layer - 1].beta

        if rng.random() < beta:
            # Rewire: keep u, replace v with a random node
            candidates = [n for n in nodes if n != u and not G.has_edge(u, n)]
            if not candidates:
                continue
            new_v = rng.choice(candidates)
            weight = data["weight"]
            G.remove_edge(u, v)
            G.add_edge(u, new_v, weight=weight, layer=layer)


def build_er_graph(cfg: GraphConfig) -> nx.Graph:
    """Build an Erdős-Rényi random graph (baseline for ablation)."""
    G = nx.erdos_renyi_graph(cfg.n_agents, cfg.er_p, seed=cfg.seed)
    # Relabel nodes
    mapping = {i: f"agent_{i:04d}" for i in range(cfg.n_agents)}
    G = nx.relabel_nodes(G, mapping)
    # Assign uniform weights
    for u, v in G.edges:
        G[u][v]["weight"] = 1.0
        G[u][v]["layer"] = 0

    log.info("Built ER graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def build_ba_graph(cfg: GraphConfig) -> nx.Graph:
    """Build a Barabási-Albert preferential attachment graph (baseline)."""
    G = nx.barabasi_albert_graph(cfg.n_agents, cfg.ba_m, seed=cfg.seed)
    mapping = {i: f"agent_{i:04d}" for i in range(cfg.n_agents)}
    G = nx.relabel_nodes(G, mapping)
    for u, v in G.edges:
        G[u][v]["weight"] = 1.0
        G[u][v]["layer"] = 0

    log.info("Built BA graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def build_graph(cfg: GraphConfig) -> nx.Graph:
    """Build a graph according to the model specified in config."""
    builders = {
        "watts_strogatz": build_dunbar_graph,
        "dunbar": build_dunbar_graph,
        "erdos_renyi": build_er_graph,
        "barabasi_albert": build_ba_graph,
    }
    builder = builders.get(cfg.model)
    if builder is None:
        raise ValueError(f"Unknown graph model: {cfg.model}. Choose from {list(builders.keys())}")
    return builder(cfg)


def relabel_graph(G: nx.Graph, names: List[str]) -> nx.Graph:
    """Relabel graph nodes using human-readable names.

    Args:
        G: Graph with placeholder node IDs (e.g. agent_0000).
        names: List of new names, same length as G.number_of_nodes().

    Returns:
        New graph with relabeled nodes.
    """
    old_ids = list(G.nodes)
    if len(names) != len(old_ids):
        raise ValueError(f"Expected {len(old_ids)} names, got {len(names)}")
    mapping = dict(zip(old_ids, names))
    return nx.relabel_nodes(G, mapping)


# ---------------------------------------------------------------------------
# Graph analytics
# ---------------------------------------------------------------------------

def graph_properties(G: nx.Graph) -> Dict[str, Any]:
    """Compute summary statistics for a graph."""
    degrees = [d for _, d in G.degree()]
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "clustering_coefficient": nx.average_clustering(G) if G.number_of_edges() > 0 else 0,
        "avg_shortest_path": _safe_avg_path(G),
        "degree_mean": float(np.mean(degrees)) if degrees else 0,
        "degree_std": float(np.std(degrees)) if degrees else 0,
        "degree_max": max(degrees) if degrees else 0,
        "is_connected": nx.is_connected(G),
        "n_components": nx.number_connected_components(G),
    }


def _safe_avg_path(G: nx.Graph) -> float:
    """Compute average shortest path length, handling disconnected graphs."""
    if G.number_of_nodes() <= 1 or G.number_of_edges() == 0:
        return 0.0
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    # Average over largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(largest_cc)
    if subG.number_of_nodes() <= 1:
        return 0.0
    return nx.average_shortest_path_length(subG)
