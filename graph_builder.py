"""
graph_builder.py
Builds a synthetic but structurally realistic supply chain network.

Design choices (and why):
- Tiered structure (raw material -> component -> subassembly -> OEM -> retail)
  because real supply chains flow in one direction with distinct roles at each tier.
- Preferential attachment within tier-to-tier connections, because real supplier
  networks are NOT random graphs -- a small number of hub suppliers (e.g. a single
  rare-earth processor, a single advanced foundry) end up supplying a large fraction
  of downstream firms. This is what makes cascading failures interesting: hub nodes
  are systemic risk points.
- Some cross-tier "shortcut" edges are added (e.g. a Tier 1 selling directly to
  retail) because real chains aren't perfectly layered.
"""

import networkx as nx
import numpy as np


def _connect_preferential(G, upstream, downstream, avg_suppliers_per_downstream, rng):
    """Connect each downstream node to a random number of upstream nodes,
    biased toward upstream nodes that already have more downstream connections
    (preferential attachment -> creates hub suppliers)."""
    degree = {u: 1.0 for u in upstream}  # start at 1 so everyone has a chance
    for d in downstream:
        k = rng.poisson(avg_suppliers_per_downstream) + 1
        k = min(k, len(upstream))
        weights = np.array([degree[u] for u in upstream])
        probs = weights / weights.sum()
        chosen = rng.choice(upstream, size=k, replace=False, p=probs)
        for u in chosen:
            G.add_edge(u, d)
            degree[u] += 1.0


def generate_supply_chain_graph(
    n_tier3=250,   # raw material / mining / basic chemicals
    n_tier2=120,   # components
    n_tier1=50,    # subassemblies
    n_oem=15,      # brand-name manufacturers
    n_retail=25,   # retail / distribution
    shortcut_edge_frac=0.03,  # fraction of extra cross-tier edges
    seed=42,
):
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    tiers = {}
    node_id = 0

    def add_nodes(n, tier_name):
        nonlocal node_id
        ids = []
        for _ in range(n):
            G.add_node(node_id, tier=tier_name)
            ids.append(node_id)
            node_id += 1
        return ids

    tier3 = add_nodes(n_tier3, "tier3_raw_material")
    tier2 = add_nodes(n_tier2, "tier2_component")
    tier1 = add_nodes(n_tier1, "tier1_subassembly")
    oem = add_nodes(n_oem, "oem")
    retail = add_nodes(n_retail, "retail")

    tiers = {"tier3": tier3, "tier2": tier2, "tier1": tier1, "oem": oem, "retail": retail}

    _connect_preferential(G, tier3, tier2, avg_suppliers_per_downstream=3, rng=rng)
    _connect_preferential(G, tier2, tier1, avg_suppliers_per_downstream=4, rng=rng)
    _connect_preferential(G, tier1, oem, avg_suppliers_per_downstream=6, rng=rng)
    _connect_preferential(G, oem, retail, avg_suppliers_per_downstream=2, rng=rng)

    # Shortcut edges: some Tier1/Tier2 firms sell directly further downstream
    all_upstream_shortcut_pool = tier2 + tier1
    all_downstream_shortcut_pool = oem + retail
    n_shortcuts = int(shortcut_edge_frac * G.number_of_edges())
    for _ in range(n_shortcuts):
        u = rng.choice(all_upstream_shortcut_pool)
        v = rng.choice(all_downstream_shortcut_pool)
        if u != v:
            G.add_edge(u, v)

    # Ensure no isolated nodes (every node should matter for cascade purposes)
    isolated = list(nx.isolates(G))
    for node in isolated:
        tier = G.nodes[node]["tier"]
        # connect it to a random node in the next tier down, or previous tier if it's retail
        if tier == "retail":
            partner = rng.choice(oem)
            G.add_edge(partner, node)
        else:
            downstream_map = {"tier3_raw_material": tier2, "tier2_component": tier1,
                               "tier1_subassembly": oem, "oem": retail}
            partner = rng.choice(downstream_map[tier])
            G.add_edge(node, partner)

    return G, tiers


if __name__ == "__main__":
    G, tiers = generate_supply_chain_graph()
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    for t, ids in tiers.items():
        print(f"  {t}: {len(ids)} nodes")
    degrees = [d for _, d in G.out_degree()]
    print(f"Max out-degree (hub check): {max(degrees)}")
