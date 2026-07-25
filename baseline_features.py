"""
baseline_features.py

Builds a tabular, non-temporal dataset for Baseline A:
"given a disruption starting at seed node S, will node V eventually be disrupted?"

This deliberately throws away the graph's message-passing structure and reduces
it to hand-engineered features -- that's the point. If a GNN can't beat this,
the graph structure isn't adding value and you need to know that before you
write your results section.

Features per (sim_id, node) row:
  - shortest_path_len_from_seed : hops from seed to this node (capped if unreachable)
  - degree_centrality           : node's in+out degree centrality in the static graph
  - betweenness_centrality      : node's betweenness (proxy for "structural chokepoint")
  - capacity_score, financial_health, resilience, baseline_risk : node's own static features
  - seed_tier_*                 : one-hot of which tier the shock originated in
  - seed_base_transmission      : how "severe" this particular shock was

Label:
  - disrupted : 1 if this node was ever in state==1 (DISRUPTED) during this simulation
"""

import networkx as nx
import numpy as np
import pandas as pd


def load_graph_from_csv(output_dir="output"):
    nodes = pd.read_csv(f"{output_dir}/nodes.csv")
    edges = pd.read_csv(f"{output_dir}/edges.csv")
    G = nx.DiGraph()
    for _, row in nodes.iterrows():
        G.add_node(int(row.node_id), **row.to_dict())
    for _, row in edges.iterrows():
        G.add_edge(int(row.source), int(row.target), criticality=row.criticality)
    return G, nodes, edges


def compute_static_centrality(G):
    """Compute once -- these don't change across simulations."""
    # use undirected for betweenness since we care about structural position generally
    und = G.to_undirected()
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(und, k=min(100, len(und)), seed=42)  # approx for speed
    return degree_centrality, betweenness


def build_baseline_dataset(output_dir="output", max_hops_cap=10, sample_frac=1.0, seed=42):
    G, nodes_df, edges_df = load_graph_from_csv(output_dir)
    log = pd.read_csv(f"{output_dir}/cascade_log.csv.gz")
    meta = pd.read_csv(f"{output_dir}/cascade_meta.csv")

    print("Computing static centrality (once)...")
    degree_centrality, betweenness = compute_static_centrality(G)

    node_static = nodes_df.set_index("node_id")

    # ever-disrupted set per sim_id (label source)
    print("Computing ever-disrupted sets per simulation...")
    disrupted_mask = log["state"] == 1
    ever_disrupted = (
        log[disrupted_mask].groupby("sim_id")["node_id"].apply(set).to_dict()
    )

    rng = np.random.default_rng(seed)
    rows = []

    print(f"Building feature rows for {len(meta)} simulations...")
    # cache shortest-path lengths per seed node (many sims may share seeds, but even if not,
    # this is only computed once per unique seed encountered)
    sp_cache = {}

    all_nodes = list(G.nodes())
    for _, sim_row in meta.iterrows():
        sim_id = int(sim_row.sim_id)
        seed_node = int(sim_row.seed_node)
        seed_tier = sim_row.seed_tier
        base_transmission = sim_row.base_transmission
        disrupted_set = ever_disrupted.get(sim_id, set())

        if seed_node not in sp_cache:
            lengths = nx.single_source_shortest_path_length(G, seed_node)
            sp_cache[seed_node] = lengths
        lengths = sp_cache[seed_node]

        candidate_nodes = all_nodes
        if sample_frac < 1.0:
            candidate_nodes = rng.choice(all_nodes, size=int(len(all_nodes) * sample_frac), replace=False)

        for node_id in candidate_nodes:
            hop_dist = lengths.get(node_id, max_hops_cap)
            hop_dist = min(hop_dist, max_hops_cap)
            static = node_static.loc[node_id]
            rows.append({
                "sim_id": sim_id,
                "node_id": node_id,
                "shortest_path_len_from_seed": hop_dist,
                "degree_centrality": degree_centrality[node_id],
                "betweenness_centrality": betweenness[node_id],
                "capacity_score": static.capacity_score,
                "financial_health": static.financial_health,
                "resilience": static.resilience,
                "baseline_risk": static.baseline_risk,
                "seed_tier": seed_tier,
                "seed_base_transmission": base_transmission,
                "disrupted": int(node_id in disrupted_set),
            })

    df = pd.DataFrame(rows)
    df = pd.get_dummies(df, columns=["seed_tier"], prefix="seed_tier")
    return df


if __name__ == "__main__":
    df = build_baseline_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"Positive rate (disrupted=1): {df.disrupted.mean():.3f}")
    df.to_csv("output/baseline_dataset.csv", index=False)
    print("Saved to output/baseline_dataset.csv")
