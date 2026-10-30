"""
predict_cascade.py

The "product" layer: given a trained GNN (from train_gnn.py) and a chosen seed
node, autoregressively rolls the model forward to forecast how a disruption
would spread -- this is what a supply chain manager would actually click on.

Limitation to state plainly in your report: the trained model predicts a single
"probability of being disrupted next step" channel (not the full 3-state
susceptible/disrupted/recovered dynamics), so this rollout treats disruption as
monotonic (a node predicted disrupted stays flagged as disrupted for
visualization purposes) rather than modeling recovery. That's a reasonable
simplification for a "how far could this spread" forecasting tool, but it is a
simplification -- extending the model to predict all 3 state channels (instead
of 1) is a natural "future work" item if you have time.

Usage:
    python3 predict_cascade.py --seed_node 12 --steps 15
    python3 predict_cascade.py --list_nodes        # to see valid node ids + tiers
"""

import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx
from train_gnn import DCRNNPredictor, DEVICE
from pyg_loader_example import (
    load_raw, build_static_edge_index, build_static_node_features, build_model_ready_edges,
)


def load_trained_model(output_dir="output"):
    checkpoint = torch.load(f"{output_dir}/gnn_model.pt", map_location=DEVICE, weights_only=False)
    model = DCRNNPredictor(checkpoint["node_feature_dim"], hidden_dim=checkpoint["hidden_dim"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_graph_objects(output_dir="output"):
    nodes, edges, log, meta = load_raw(output_dir)
    node_id_to_idx = {nid: i for i, nid in enumerate(sorted(nodes.node_id.unique()))}
    idx_to_node_id = {i: nid for nid, i in node_id_to_idx.items()}
    edge_index_np, edge_attr_np = build_static_edge_index(edges, node_id_to_idx)
    static_features, feature_names = build_static_node_features(nodes, node_id_to_idx)

    edge_index, edge_weight = build_model_ready_edges(
        edge_index_np, edge_attr_np, static_features.shape[0], DEVICE
    )
    return nodes, edges, node_id_to_idx, idx_to_node_id, edge_index, edge_weight, static_features


def rollout(model, seed_node_id, node_id_to_idx, static_features, edge_index, edge_weight,
            steps=15, disruption_threshold=0.5):
    """Autoregressively roll the model forward `steps` timesteps starting from a
    single disrupted seed node. Returns a [steps+1, n_nodes] array of predicted
    disruption probabilities and a [steps+1, n_nodes] array of binary flagged states."""
    n_nodes = static_features.shape[0]
    seed_idx = node_id_to_idx[seed_node_id]

    state_onehot = np.zeros((n_nodes, 3), dtype=np.float32)
    state_onehot[:, 0] = 1.0  # everyone starts SUSCEPTIBLE
    state_onehot[seed_idx] = [0.0, 1.0, 0.0]  # seed node starts DISRUPTED

    prob_history = [state_onehot[:, 1].copy()]  # step 0: seed is prob=1, everyone else 0
    flagged = state_onehot[:, 1].copy() > 0.5

    h = None
    with torch.no_grad():
        for t in range(steps):
            x = np.concatenate([static_features, state_onehot], axis=1)
            x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            logit, h = model(x_t, edge_index, edge_weight, h)
            probs = torch.sigmoid(logit).cpu().numpy()

            prob_history.append(probs.copy())
            newly_flagged = probs > disruption_threshold
            flagged = flagged | newly_flagged  # monotonic -- see module docstring caveat

            # feed predicted state back in as next input (autoregressive)
            state_onehot = np.zeros((n_nodes, 3), dtype=np.float32)
            state_onehot[~flagged, 0] = 1.0
            state_onehot[flagged, 1] = 1.0

    return np.array(prob_history), flagged


def visualize_cascade(G, idx_to_node_id, node_id_to_idx, prob_history, seed_node_id,
                       output_path="output/cascade_forecast.png", snapshot_steps=None):
    """Static multi-panel figure: network colored by predicted disruption
    probability at a few chosen timesteps, laid out once (fixed layout) so
    panels are visually comparable."""
    if snapshot_steps is None:
        max_step = prob_history.shape[0] - 1
        snapshot_steps = sorted(set([0, max_step // 3, 2 * max_step // 3, max_step]))

    pos = nx.spring_layout(G, seed=42, k=0.25)  # fixed layout, computed once

    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(6 * len(snapshot_steps), 6))
    if len(snapshot_steps) == 1:
        axes = [axes]

    for ax, step in zip(axes, snapshot_steps):
        probs_by_idx = prob_history[step]
        node_colors = [probs_by_idx[node_id_to_idx[n]] for n in G.nodes()]
        node_sizes = [500 if n == seed_node_id else 60 for n in G.nodes()]

        nodes_drawn = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, cmap="Reds", vmin=0, vmax=1,
            node_size=node_sizes, linewidths=0.5, edgecolors="gray",
        )
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, arrows=False, width=0.5)
        ax.set_title(f"Step {step}")
        ax.axis("off")

    fig.suptitle(f"Predicted cascade forecast — shock originating at node {seed_node_id}", fontsize=14)
    fig.colorbar(nodes_drawn, ax=axes, label="P(disrupted)", shrink=0.7)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_node", type=int, default=None, help="node_id to simulate a shock at")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--list_nodes", action="store_true", help="print available node_ids + tiers and exit")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    nodes, edges, node_id_to_idx, idx_to_node_id, edge_index, edge_weight, static_features = \
        build_graph_objects(args.output_dir)

    if args.list_nodes:
        print(nodes[["node_id", "tier", "region"]].to_string(index=False))
        return

    if args.seed_node is None:
        # default: pick a tier3 (raw material) node, since that's the realistic origin tier
        seed_node = nodes[nodes.tier == "tier3_raw_material"].node_id.iloc[0]
        print(f"No --seed_node given, defaulting to node {seed_node} (tier3_raw_material).")
    else:
        seed_node = args.seed_node

    model = load_trained_model(args.output_dir)

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(int(row.source), int(row.target))
    G.add_nodes_from(nodes.node_id)

    prob_history, flagged = rollout(
        model, seed_node, node_id_to_idx, static_features, edge_index, edge_weight,
        steps=args.steps, disruption_threshold=args.threshold,
    )

    n_flagged = flagged.sum()
    print(f"\nSeed node: {seed_node}  ({nodes.set_index('node_id').loc[seed_node].tier})")
    print(f"Predicted total nodes disrupted within {args.steps} steps (threshold={args.threshold}): {n_flagged}")

    top_affected = pd.DataFrame({
        "node_id": [idx_to_node_id[i] for i in range(len(flagged))],
        "final_prob": prob_history[-1],
    }).merge(nodes[["node_id", "tier", "region"]], on="node_id").sort_values("final_prob", ascending=False)
    print("\nTop 10 nodes by predicted final disruption probability:")
    print(top_affected.head(10).to_string(index=False))

    visualize_cascade(G, idx_to_node_id, node_id_to_idx, prob_history, seed_node,
                       output_path=f"{args.output_dir}/cascade_forecast_node{seed_node}.png")


if __name__ == "__main__":
    main()
