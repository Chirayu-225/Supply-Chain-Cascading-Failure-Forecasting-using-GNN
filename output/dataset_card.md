# Synthetic Supply Chain Cascade Dataset

## Generation parameters
- Nodes: 460 total
  - tier3_raw_material: 250
  - tier2_component: 120
  - tier1_subassembly: 50
  - oem: 15
  - retail: 25
- Edges: 1054
- Simulations: 2000, max 26 timesteps each
- Graph random seed: 42, simulation random seed: 123

## Files
- `nodes.csv`: node_id, tier, capacity_score, financial_health, resilience, region, baseline_risk
- `edges.csv`: source, target, volume_share, criticality, lead_time_days
- `cascade_log.csv.gz`: sim_id, step, node_id, state (0=susceptible, 1=disrupted, 2=recovered)
- `cascade_meta.csv`: sim_id, seed_node, seed_tier, base_transmission, n_nodes_affected, cascade_depth_steps

## Cascade outcome summary
- Mean nodes affected per simulation: 4.74
- Median nodes affected: 2.0
- Max nodes affected in a single cascade: 51
- Mean cascade depth (steps until fully resolved or capped): 5.69

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
