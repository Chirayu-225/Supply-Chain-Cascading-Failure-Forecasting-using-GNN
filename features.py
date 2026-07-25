"""
features.py
Assigns node and edge attributes that a GNN can actually learn from.

Node features:
- capacity_score       : production capacity, 0-1 (log-normal -> few giants, many small firms)
- financial_health      : 0-1, beta-distributed (skewed toward healthy, since most firms are stable)
- region                : categorical, affects correlated regional shocks (e.g. one flood hits many)
- resilience            : 0-1, buffer stock / backup supplier availability -- KEY variable, since
                           this is what determines whether a node absorbs or passes on a shock
- baseline_risk         : 0-1, static risk score (e.g. political/geographic risk of its region)

Edge features:
- criticality           : 0-1, how hard it is to substitute this supplier (sole-source -> high)
- volume_share          : fraction of downstream node's input volume coming through this edge
- lead_time_days        : replenishment lead time
"""

import numpy as np
import pandas as pd

REGIONS = ["N_America", "W_Europe", "E_Asia", "SE_Asia", "S_Asia", "Latin_America"]
REGION_BASE_RISK = {  # illustrative, not literal geopolitical claims -- tune freely
    "N_America": 0.15, "W_Europe": 0.15, "E_Asia": 0.30,
    "SE_Asia": 0.35, "S_Asia": 0.30, "Latin_America": 0.25,
}


def assign_node_features(G, seed=42):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = len(nodes)

    capacity_raw = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    capacity_score = (capacity_raw - capacity_raw.min()) / (capacity_raw.max() - capacity_raw.min())

    financial_health = rng.beta(a=5, b=2, size=n)  # skewed healthy
    resilience = rng.beta(a=2, b=3, size=n)          # skewed toward LOW resilience (realistic -- most firms run lean)
    region = rng.choice(REGIONS, size=n)
    baseline_risk = np.array([REGION_BASE_RISK[r] for r in region]) + rng.normal(0, 0.05, size=n)
    baseline_risk = np.clip(baseline_risk, 0.01, 0.99)

    for i, node in enumerate(nodes):
        G.nodes[node]["capacity_score"] = float(capacity_score[i])
        G.nodes[node]["financial_health"] = float(financial_health[i])
        G.nodes[node]["resilience"] = float(resilience[i])
        G.nodes[node]["region"] = region[i]
        G.nodes[node]["baseline_risk"] = float(baseline_risk[i])

    return G


def assign_edge_features(G, seed=43):
    rng = np.random.default_rng(seed)
    # volume_share: for each downstream node, split 1.0 across its incoming edges (Dirichlet)
    for node in G.nodes():
        in_edges = list(G.in_edges(node))
        if not in_edges:
            continue
        shares = rng.dirichlet(np.ones(len(in_edges)) * 0.8)  # alpha<1 -> some edges dominate (sole-source-like)
        for (u, v), share in zip(in_edges, shares):
            criticality = float(np.clip(share * rng.uniform(0.7, 1.3), 0, 1))
            lead_time = float(rng.gamma(shape=4, scale=5))  # days, mean ~20
            G.edges[u, v]["volume_share"] = float(share)
            G.edges[u, v]["criticality"] = criticality
            G.edges[u, v]["lead_time_days"] = lead_time
    return G


def nodes_to_dataframe(G):
    rows = []
    for node, data in G.nodes(data=True):
        row = {"node_id": node, **data}
        rows.append(row)
    return pd.DataFrame(rows)


def edges_to_dataframe(G):
    rows = []
    for u, v, data in G.edges(data=True):
        row = {"source": u, "target": v, **data}
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from graph_builder import generate_supply_chain_graph
    G, tiers = generate_supply_chain_graph()
    G = assign_node_features(G)
    G = assign_edge_features(G)
    ndf = nodes_to_dataframe(G)
    edf = edges_to_dataframe(G)
    print(ndf.select_dtypes(include=[np.number]).describe())
    print(edf.select_dtypes(include=[np.number]).describe())
