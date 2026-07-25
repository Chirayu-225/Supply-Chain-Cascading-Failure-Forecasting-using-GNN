"""
cascade_sim.py
Simulates disruption cascades over the supply chain graph using an
independent-cascade / SIR-style diffusion model.

Mechanic:
- A node is shocked (SUSCEPTIBLE -> DISRUPTED) either exogenously (the initial
  event: fire, strike, flood) or endogenously (a disrupted upstream neighbor
  fails to deliver).
- Each disrupted node has a per-timestep probability of "infecting" each of its
  DOWNSTREAM neighbors (the node relies on it as an input). This probability is
  a function of:
    - edge criticality (sole-source edges transmit shocks more reliably)
    - the downstream node's resilience (buffer stock / alternate suppliers absorb shocks)
    - the downstream node's financial_health (weaker firms recover slower / are hit harder)
- A disrupted node recovers (DISRUPTED -> RECOVERED) after a random recovery
  time influenced by financial_health and capacity_score.
- This produces a per-timestep node-state trajectory for every simulation run --
  exactly the (X_t -> X_t+1) sequence a Temporal GNN needs to learn from.

This is deliberately kept dependency-light (numpy/networkx/pandas only) so it
runs anywhere. Swap in PyTorch Geometric Temporal only at the *training* stage.
"""

import numpy as np
import pandas as pd

SUSCEPTIBLE, DISRUPTED, RECOVERED = 0, 1, 2


def run_single_cascade(G, seed_node, rng, max_steps=26, base_transmission=0.35):
    """Run one cascade simulation starting at `seed_node`.
    Returns a list of dicts: one row per (timestep, node) with its state.
    """
    state = {n: SUSCEPTIBLE for n in G.nodes()}
    state[seed_node] = DISRUPTED
    disruption_start_step = {seed_node: 0}
    recovery_step = {}

    log = []

    def log_state(step):
        for n in G.nodes():
            log.append({"step": step, "node_id": n, "state": state[n]})

    log_state(0)

    for step in range(1, max_steps + 1):
        newly_disrupted = []
        currently_disrupted = [n for n, s in state.items() if s == DISRUPTED]

        # 1. propagate to downstream neighbors
        for u in currently_disrupted:
            for v in G.successors(u):
                if state[v] != SUSCEPTIBLE:
                    continue
                edge = G.edges[u, v]
                crit = edge["criticality"]
                downstream_resilience = G.nodes[v]["resilience"]
                downstream_health = G.nodes[v]["financial_health"]
                # transmission probability: up with criticality, down with resilience/health
                # floor of 0.25 on the criticality term so even low-criticality edges
                # carry some risk -- avoids cascades dying out almost instantly, which
                # would starve the model of "interesting" (far-reaching) training examples
                crit_factor = 0.25 + 0.75 * crit
                p_transmit = base_transmission * crit_factor * (1 - 0.6 * downstream_resilience) * (1 - 0.25 * downstream_health)
                p_transmit = float(np.clip(p_transmit, 0, 0.95))
                if rng.random() < p_transmit:
                    newly_disrupted.append(v)

        for v in newly_disrupted:
            state[v] = DISRUPTED
            disruption_start_step[v] = step

        # 2. recovery check for already-disrupted nodes
        for n, s in list(state.items()):
            if s == DISRUPTED and n not in recovery_step:
                health = G.nodes[n]["financial_health"]
                capacity = G.nodes[n]["capacity_score"]
                # healthier / higher-capacity firms recover faster (fewer steps)
                mean_recovery = 10 - 4 * health - 2 * capacity
                mean_recovery = max(2, mean_recovery)
                elapsed = step - disruption_start_step[n]
                # simple hazard: probability of recovering this step increases as elapsed approaches mean
                p_recover = 1 / mean_recovery
                if rng.random() < p_recover:
                    state[n] = RECOVERED
                    recovery_step[n] = step

        log_state(step)

        if all(s != DISRUPTED for s in state.values()):
            break  # cascade fully resolved

    return log, disruption_start_step


def run_many_cascades(G, n_simulations=2000, seed=123, max_steps=26):
    """Run many cascades from varied seed nodes, weighted toward nodes that make
    sense as real-world shock origins (e.g. upstream raw material / component tiers,
    which is where most real disruptions -- floods, strikes, mine closures -- originate).
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    tiers = np.array([G.nodes[n]["tier"] for n in nodes])

    # weight seed selection: upstream tiers more likely to be the shock origin
    tier_weight = {"tier3_raw_material": 3.0, "tier2_component": 2.0,
                   "tier1_subassembly": 1.0, "oem": 0.5, "retail": 0.2}
    weights = np.array([tier_weight[t] for t in tiers])
    weights = weights / weights.sum()

    all_logs = []
    sim_metadata = []

    for sim_id in range(n_simulations):
        seed_node = rng.choice(nodes, p=weights)
        base_transmission = float(np.clip(rng.normal(0.45, 0.15), 0.1, 0.9))
        log, disruption_start_step = run_single_cascade(
            G, seed_node, rng, max_steps=max_steps, base_transmission=base_transmission
        )
        for row in log:
            row["sim_id"] = sim_id
        all_logs.extend(log)

        n_affected = len(disruption_start_step)
        cascade_depth = max(disruption_start_step.values()) if disruption_start_step else 0
        sim_metadata.append({
            "sim_id": sim_id,
            "seed_node": seed_node,
            "seed_tier": G.nodes[seed_node]["tier"],
            "base_transmission": base_transmission,
            "n_nodes_affected": n_affected,
            "cascade_depth_steps": cascade_depth,
        })

    return pd.DataFrame(all_logs), pd.DataFrame(sim_metadata)


if __name__ == "__main__":
    from graph_builder import generate_supply_chain_graph
    from features import assign_node_features, assign_edge_features

    G, tiers = generate_supply_chain_graph()
    G = assign_node_features(G)
    G = assign_edge_features(G)

    log_df, meta_df = run_many_cascades(G, n_simulations=50)
    print(meta_df.describe())
    print(f"\nTotal (step, node, sim) rows generated: {len(log_df)}")
