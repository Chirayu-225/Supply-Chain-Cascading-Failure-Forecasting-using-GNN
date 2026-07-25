# Supply Chain Cascading Failure Forecasting

Predicting *cascading* disruption in supply chain networks with a Temporal Graph
Neural Network — not just per-firm demand forecasting, but how a shock at one
supplier ripples through its trading partners over time.

## Why this project

Standard supply chain forecasting (ARIMA, Prophet, per-firm LSTMs) treats each
company as an independent time series. That structurally can't capture what
actually happened in events like the 2011 Thailand floods or the 2021 chip
shortage: a disruption at one node propagating through relationships to affect
firms with no direct connection to the original event. This project frames
disruption forecasting as a graph problem instead, and tests whether that
framing is *actually* worth the added complexity — not just assumed.

## Key result

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Non-graph baseline (with hop-distance feature) | 0.989 | 0.575 |
| **Non-graph baseline (hop-distance feature removed)** | 0.773 | **0.039** |
| **Temporal GNN (DCRNN)** | 0.969 | **0.671** |

An ablation study shows the non-graph baseline's apparent strength depended
almost entirely on a single hand-engineered feature — hop count from the
shock. Remove it, and performance collapses to near-random. The GNN is never
given that feature explicitly, and still substantially outperforms the honest
baseline — evidence it's learning real structural propagation, not a proxy
shortcut. Full writeup: [`Supply_Chain_Cascade_Project_Report.docx`](./Supply_Chain_Cascade_Project_Report.docx).

## How it works

1. **Synthetic network generation** — a tiered (raw material → component →
   subassembly → OEM → retail), preferential-attachment graph, calibrated to
   match known structural properties of real supply chains (hub suppliers,
   criticality-weighted dependencies) rather than a random graph.
2. **Cascade simulation** — an independent-cascade / SIR-style diffusion model
   propagates shocks through the network, weighted by edge criticality and
   dampened by node resilience/financial health, with recovery dynamics.
   2,000 simulations → ~11.5M (simulation, timestep, node) observations.
3. **Baseline model** — RandomForest / HistGradientBoosting on hand-engineered
   structural features (shortest-path distance, centrality, node attributes).
4. **Temporal GNN** — a DCRNN (`torch_geometric_temporal`) predicting
   per-node disruption probability at each timestep from the evolving graph
   state.
5. **Counterfactual simulation** — pick any node, simulate a shock, and
   visualize the model's forecasted cascade forward in time.

## Project structure

```
graph_builder.py          Network topology generator
features.py                Node/edge feature assignment
cascade_sim.py              SIR-style cascade propagation simulator
build_dataset.py           Orchestrates generation, writes output/
baseline_features.py       Builds the non-graph baseline's training table
train_baseline.py          Trains + evaluates the baseline (RF / HistGBM)
ablation_baseline.py        Ablation: baseline performance without hop-distance
pyg_loader_example.py       Converts CSVs -> PyTorch Geometric Temporal format
train_gnn.py                Trains + evaluates the DCRNN
predict_cascade.py          Counterfactual rollout + visualization
output/                     Generated dataset + dataset_card.md (methodology notes)
```

## Getting started

```bash
pip install -r requirements.txt

# 1. Generate the synthetic network + cascade dataset
python build_dataset.py

# 2. Non-graph baseline
python baseline_features.py
python train_baseline.py

# 3. Ablation study (is the baseline's strength genuine, or a shortcut?)
python ablation_baseline.py

# 4. Temporal GNN
python train_gnn.py

# 5. Simulate a shock at any node and visualize the forecast
python predict_cascade.py --list_nodes
python predict_cascade.py --seed_node <node_id> --steps 15
```

`output/` isn't committed (the full feature table is 150MB+, over GitHub's file
size limit) — everything regenerates deterministically from the scripts above.

## Known limitations

- Network topology and propagation dynamics are synthetic and structurally
  calibrated, not fitted to a specific real company's disclosed supplier
  network (real bilateral contract data is proprietary).
- The cascade model doesn't explicitly represent inventory buffers or
  contractual substitution logic — `resilience` is a proxy for these.
- The GNN currently predicts one channel (disrupted / not), so autoregressive
  forecasts treat disruption as roughly monotonic rather than modeling
  recovery explicitly.

See `output/dataset_card.md` (generated alongside the dataset) and the full
report for a complete discussion.

## Requirements

Python 3.10+, `numpy`, `pandas`, `networkx`, `scikit-learn` for the pipeline
and baseline; `torch`, `torch_geometric`, `torch_geometric_temporal` for the
GNN (see `requirements.txt`).
