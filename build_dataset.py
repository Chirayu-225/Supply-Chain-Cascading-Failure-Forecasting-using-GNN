"""
build_dataset.py
Orchestrates the full pipeline and exports a GNN-training-ready dataset:

Outputs (written to ./output/):
  nodes.csv           - static node features (one row per node)
  edges.csv           - static edge list + edge features (one row per edge)
  cascade_log.csv.gz - per-simulation, per-timestep, per-node state
                         (this is your temporal training signal)
  cascade_meta.csv    - per-simulation summary (seed node, severity, depth)
  dataset_card.md     - documentation of generation parameters (for your report)

Suggested ML framing once loaded:
  - Node classification per timestep: state_t -> state_{t+1} (SUSCEPTIBLE/DISRUPTED/RECOVERED)
  - Graph-level regression: predict total cascade size / duration given seed node + graph state
  - Use cascade_log pivoted to (sim_id, step) snapshots as your temporal graph sequence
    for PyTorch Geometric Temporal (DynamicGraphTemporalSignal or similar).
"""

import os
import json
import numpy as np
import pandas as pd

from graph_builder import generate_supply_chain_graph
from features import assign_node_features, assign_edge_features, nodes_to_dataframe, edges_to_dataframe
from cascade_sim import run_many_cascades


def build(
    output_dir="output",
    n_tier3=250, n_tier2=120, n_tier1=50, n_oem=15, n_retail=25,
    n_simulations=2000, max_steps=26, graph_seed=42, sim_seed=123,
):
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1/4: generating graph topology...")
    G, tiers = generate_supply_chain_graph(
        n_tier3=n_tier3, n_tier2=n_tier2, n_tier1=n_tier1,
        n_oem=n_oem, n_retail=n_retail, seed=graph_seed,
    )
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Step 2/4: assigning node/edge features...")
    G = assign_node_features(G, seed=graph_seed)
    G = assign_edge_features(G, seed=graph_seed + 1)

    node_df = nodes_to_dataframe(G)
    edge_df = edges_to_dataframe(G)

    print(f"Step 3/4: running {n_simulations} cascade simulations "
          f"(this is where 'abundant and complex' comes from)...")
    log_df, meta_df = run_many_cascades(G, n_simulations=n_simulations, seed=sim_seed, max_steps=max_steps)

    print("Step 4/4: writing output files...")
    node_path = os.path.join(output_dir, "nodes.csv")
    edge_path = os.path.join(output_dir, "edges.csv")
    log_path = os.path.join(output_dir, "cascade_log.csv.gz")
    meta_path = os.path.join(output_dir, "cascade_meta.csv")
    card_path = os.path.join(output_dir, "dataset_card.md")

    node_df.to_csv(node_path, index=False)
    edge_df.to_csv(edge_path, index=False)
    log_df.to_csv(log_path, index=False, compression="gzip")
    meta_df.to_csv(meta_path, index=False)

    card = f"""# Synthetic Supply Chain Cascade Dataset

## Generation parameters
- Nodes: {G.number_of_nodes()} total
  - tier3_raw_material: {n_tier3}
  - tier2_component: {n_tier2}
  - tier1_subassembly: {n_tier1}
  - oem: {n_oem}
  - retail: {n_retail}
- Edges: {G.number_of_edges()}
- Simulations: {n_simulations}, max {max_steps} timesteps each
- Graph random seed: {graph_seed}, simulation random seed: {sim_seed}

## Files
- `nodes.csv`: node_id, tier, capacity_score, financial_health, resilience, region, baseline_risk
- `edges.csv`: source, target, volume_share, criticality, lead_time_days
- `cascade_log.csv.gz`: sim_id, step, node_id, state (0=susceptible, 1=disrupted, 2=recovered)
- `cascade_meta.csv`: sim_id, seed_node, seed_tier, base_transmission, n_nodes_affected, cascade_depth_steps

## Cascade outcome summary
- Mean nodes affected per simulation: {meta_df.n_nodes_affected.mean():.2f}
- Median nodes affected: {meta_df.n_nodes_affected.median():.1f}
- Max nodes affected in a single cascade: {meta_df.n_nodes_affected.max()}
- Mean cascade depth (steps until fully resolved or capped): {meta_df.cascade_depth_steps.mean():.2f}

## Known limitations (state these explicitly in your report)
- Network topology and propagation dynamics are synthetic, calibrated by structural
  assumptions (tiering, preferential attachment, criticality-weighted transmission)
  rather than fitted to a specific real company's disclosed supplier network.
- Propagation model is a simplified independent-cascade/SIR process; it does not
  model inventory buffers, contractual penalties, or multi-supplier substitution
  logic explicitly (resilience score is a proxy for this).
- Recommended: validate qualitative cascade behavior (speed, depth) against
  published case studies (e.g. 2011 Thailand floods HDD shortage, 2021 chip
  shortage propagation timelines) and report the comparison in your methodology.
"""
    with open(card_path, "w") as f:
        f.write(card)

    print(f"\nDone. Files written to {output_dir}/:")
    for p in [node_path, edge_path, log_path, meta_path, card_path]:
        size_kb = os.path.getsize(p) / 1024
        print(f"  {p}  ({size_kb:.1f} KB)")

    return node_df, edge_df, log_df, meta_df


if __name__ == "__main__":
    build()
