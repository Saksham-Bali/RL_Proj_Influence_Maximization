# Community-Aware Influence Maximization via Reinforcement Learning

> **Selecting seed nodes that don't just spread far — they spread wide.**

A deep reinforcement learning framework for Influence Maximization on large-scale social networks, extending [ToupleGDD](https://arxiv.org/abs/2108.04430) with community-diversity objectives. 

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Manual Setup](#manual-setup)
- [Training](#training)
- [Inference](#inference)
- [Running Baselines](#running-baselines)
- [Regenerating Training Data (Optional)](#regenerating-training-data-optional)
- [Key Design Decisions](#key-design-decisions)
- [Dependencies](#dependencies)
- [Authors](#authors)

---

## Overview

**Influence Maximization (IM):** Given a social network and a budget *k*, select *k* seed nodes that maximise information spread under the Independent Cascade (IC) model.

**The gap we fill:** Classical algorithms (CELF, IMM) maximise raw spread but are *community-blind* — cascades get trapped in echo chambers. We add a **community-diversity objective** so the agent learns to reach structurally distinct parts of the network.

**Our approach:**
| Component | Description |
|---|---|
| **Model** | ToupleGDD — three coupled GNNs (State / Source / Target) |
| **RL** | Double DQN with ε-greedy exploration + n-step returns |
| **Pre-training** | Personalized DeepWalk (PDW) for initial node embeddings |
| **Reward** | `α × (Δinf / N) + β × (Δcomm / C)` — jointly optimises spread and diversity |
| **Dataset** | LiveJournal (SNAP) — 4M nodes, 34.6M edges, 5,000 communities |

---

## Architecture

```
Raw LiveJournal Graph
        │
        ▼
  subsample.py ──────────────────────────────────────┐
        │                                             │
        ▼                                             ▼
70 Training Subgraphs                     1 Large Test Graph
(100 nodes, on avg. 5 communities)           (~350K nodes, ~3500 communities)
        │                                             │
        ▼                                             │
   main.py (Training)                                 │
        │                                             │
   ┌────┴────┐                                        │
   ▼         ▼                                        │
rl_agents  runner.py ──► Checkpoints + Plots          │
   │                                                  │
   ├──► models.py (ToupleGDD)                         │
   │         ├─ Personalized DeepWalk (pre-train)     │
   │         ├─ State GNN                             │
   │         ├─ Source GNN  ◄── Community Projection  │
   │         └─ Target GNN      (5-dim → embed_dim)   │
   │                                                  │
   └──► environment.py ◄── graph_utils.py             │
              (RR during train, MC during eval)        │
                                                      │
   inference.py ◄────────────────────────────────────┘
        │
        ▼
  Seed Nodes + Metrics
  (Influence Spread, Communities Reached)
```

### Community Feature Vector (5-dim, graph-size invariant)

| Column | Feature | Description |
|--------|---------|-------------|
| 0 | `log1p(num_memberships)` | How many communities this node belongs to |
| 1 | `mean(comm_sizes) / N` | Average relative size of node's communities |
| 2 | `intra_edges / total_edges` | Fraction of neighbours sharing a community |
| 3 | `max(comm_size) / N` | Relative size of the node's largest community |
| 4 | **Dynamic:** `log1p(uncovered_comms)` | Communities not yet covered by current seed set |

The fixed 5-dim representation is the key to zero-shot generalisation: training graphs (on avgerage 5 communities) and the test graph (~3500 communities) both produce identical `(N, 5)` tensors.

---

## Key Results

### RL Agent vs. Classical Baselines

| Method | Influence Spread | Communities Reached | Inference Time |
|--------|-----------------|--------------------|-|
| Random | low | low | instant |
| Degree Centrality | moderate | low (echo-chamber) | instant |
| CELF | 77 nodes (14% of 493-node graph) | low | **infeasible on large graphs** |
| IMM | 81.2 ± 17.8 nodes | low (echo-chamber) | slow |
| **Ours (α=1, β=10)** | 41 nodes avg. | **4.3 unique communities** | **milliseconds** |


- **Zero-shot inference:** Policy trained on 100-node subgraphs.
- **Convergence:** Stable within 50–100 epochs; TD loss in range 10⁻⁵ to 10⁻³.

### Training Diagnostics

Training and validation plots are saved automatically every 10 epochs:

| Plot | Description |
|------|-------------|
| `training_diagnostics.png` | 4-panel: TD loss, validation reward, influence spread, communities reached |
| `average_metrics.png` | 3-panel averaged metrics across all training graphs |

---

## Repository Structure

```
RL_Proj_Influence_Maximization/
├── run.sh                          # Master pipeline script (end-to-end)
├── subsample.py                    # Data preprocessing: graph → train/test splits
├── run_imm_celf.py                 # Standalone IMM & CELF baseline runner
├── top_5000_communities.txt        # Raw community data (LiveJournal)
├── EDA.ipynb                       # Exploratory Data Analysis
├── Influence_Maximization_Base_Models.ipynb
│
└── touplegdd/                      # Core model package
    ├── main.py                     # Training entry point (CLI)
    ├── inference.py                # Inference: load model → generate seeds
    ├── models.py                   # ToupleGDD neural network
    ├── rl_agents.py                # DQN agent (action selection, replay buffer)
    ├── environment.py              # RL environment (reward, episode management)
    ├── runner.py                   # Training orchestrator (epoch loop, plotting)
    ├── baseline.py                 # Random / degree / manual baselines
    ├── utils/
    │   └── graph_utils.py          # Graph class, I/O, Monte Carlo & RR estimation
    ├── train_graphs_new/           # 70 training subgraphs (edge lists)
    ├── train_comms_new/            # 70 matching community files
    ├── test_graph_new.txt          # Large test graph (~350K nodes)
    ├── test_comms_new.txt          # 3500 communities for test graph
    └── results_*/                  # Saved checkpoints & plots per experiment
```

---

## Quickstart

> **Linux/Mac only.** The `run.sh` script handles everything end-to-end.

```bash
git clone https://github.com/Saksham-Bali/RL_Proj_Influence_Maximization.git
cd RL_Proj_Influence_Maximization

chmod +x run.sh
./run.sh
```

This will:
1. Create a Python virtual environment at `.venv/`
2. Install all dependencies (PyTorch 2.5.1 CPU, PyG, torch_scatter)
3. Run training via `main.py`
4. Run inference via `inference.py`
5. Collect checkpoints, plots, and logs into a timestamped output directory

## Manual Setup

### Prerequisites

- Python 3.8+ (3.10+ recommended)
- Git
- 16 GB RAM minimum (the test graph is large)
- GPU optional — CPU training works but is slower

### Step 1 — Clone & Create Environment

```bash
git clone https://github.com/Saksham-Bali/RL_Proj_Influence_Maximization.git
cd RL_Proj_Influence_Maximization
```

```bash
# Using conda:
conda create -n touplegdd python=3.10 -y
conda activate touplegdd

# Or using venv:
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
```

### Step 2 — Install Dependencies

```bash
pip install numpy scipy matplotlib tqdm networkx

# CPU PyTorch:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# GPU PyTorch (if you have CUDA):
pip install torch==2.5.1

# PyG extensions:
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

# PyTorch Geometric:
pip install torch_geometric
```

---

## Training

The repository ships with **pre-generated training data** in `touplegdd/train_graphs_new/` and `touplegdd/train_comms_new/` — you can skip directly to this step.

```bash
cd touplegdd

python main.py \
    --graph train_graphs_new \
    --community_path train_comms_new \
    --budget 5 \
    --alpha 1.0 \
    --beta 2.0 \
    --bs 16 \
    --epoch 20000 \
    --model Tripling \
    --model_file tripling.ckpt \
    --n_step 2
```

**What happens:**
1. PDW embeddings are pre-trained for each training graph (30 epochs each)
2. Replay buffer is filled with 1,000 random episodes
3. Agent trains for the specified epochs with ε-greedy exploration
4. Checkpoints saved every 10 epochs to a timestamped directory (e.g., `2026-05-04_22:00:00/`)
5. `training_diagnostics.png` and `average_metrics.png` generated automatically

> **Quick test:** Use `--epoch 200` to verify your setup before a full run.

### Key Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--budget` | `5` | Number of seed nodes to select |
| `--epoch` | `20000` | Training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--bs` | `16` | Replay buffer minibatch size |
| `--n_step` | `2` | N-step return horizon |
| `--alpha` | `1.0` | Reward weight for influence gain |
| `--beta` | `2.0` | Reward weight for community diversity |
| `--T` | `3` | GNN message-passing iterations |
| `--embed_dim` | `50` | Embedding dimension |
| `--memory_size` | `50000` | Replay buffer capacity |

### Resume from Checkpoint

```bash
python main.py \
    --graph train_graphs_new \
    --community_path train_comms_new \
    --budget 5 --alpha 1.0 --beta 2.0 --bs 16 \
    --epoch 5000 \
    --model Tripling \
    --model_file tripling.ckpt \
    --resume <TIMESTAMP_DIR>/tripling.ckpt500 \
    --start_epoch 500
```

---

## Inference

```bash
cd touplegdd

python inference.py \
    --graph test_graph_new.txt \
    --community_path test_comms_new.txt \
    --model Tripling \
    --model_file <TIMESTAMP_DIR>/tripling.ckpt \
    --num_communities 1 \
    --budget 5
```

Replace `<TIMESTAMP_DIR>` with the directory created during training (e.g., `2026-05-04_22:00:00/`).

**Output:**
- Selected seed node IDs
- Seed generation time
- Expected influence spread (10K Monte Carlo trials)
- Expected unique communities reached

---

## Running Baselines

### Random & Degree Baselines

```bash
cd touplegdd

# Random baseline (averaged over 10 runs)
python baseline.py \
    --graph test_graph_new.txt \
    --community_path test_comms_new.txt \
    --mode random --budget 5 --runs 10

# Degree centrality (top-k by out-degree)
python baseline.py \
    --graph test_graph_new.txt \
    --community_path test_comms_new.txt \
    --mode degree --budget 5
```

### IMM & CELF Baselines

```bash
# Run from repo root — note: very slow on large graphs
cd ..
python run_imm_celf.py
```

> CELF is computationally infeasible on the full 800K-node test graph. Run on smaller subgraphs for comparison.

---


## Regenerating Training Data (Optional)

> The repo already includes pre-generated data. Only do this if you want to regenerate from the raw LiveJournal graph.

Place `actual_graph.txt` and `top_5000_communities.txt` in the repo root, then:

```bash
python subsample.py
```

This generates 70 training subgraphs using concurrent multi-anchor BFS with Jaccard deduplication filtering (see [Design Decisions](#-key-design-decisions)).

---

## Key Design Decisions

### Fixed 5-dim Community Features
The community feature vector has exactly 5 dimensions regardless of graph scale. This means the projection weights `Linear(5, 50)` learned on 12-community training graphs apply without modification to the 3500-community test graph. Dimension mismatch — the standard failure mode for cross-graph generalisation — is entirely avoided.

### Normalised Composite Reward
Both terms in `r = α × (Δinf / N) + β × (Δcomm / C)` are normalised to [0, 1]. The reward scale is consistent whether the graph has 100 nodes or 800,000, enabling zero-shot transfer.

### Dynamic Novelty Flag (Column 4)
At each RL step, column 4 of the community feature matrix is recomputed as `log1p(uncovered_comms_for_node)`. This causes the GNN to implicitly learn that selecting nodes with many uncovered communities yields high β reward, driving the policy toward diversity without any architectural changes.

### Dual Community Projections
Two separate linear projections (`comm_proj_source`, `comm_proj_target`) inject community features into the Source and Target GNNs independently. A node's capacity to *emit* influence and its susceptibility to *receive* influence are qualitatively different signals — separate projections let the model learn both.

### Concurrent Multi-Anchor BFS
Training subgraphs are extracted by growing from **four structurally distant anchor communities simultaneously**. Single-anchor BFS produces subgraphs dominated by one community, leaving the agent nothing to learn from. Four well-separated anchors force each subgraph to span distinct network regions with genuine diversity trade-offs.

---

## Dependencies

| Package | Version |
|---------|---------|
| Python | 3.8+ |
| PyTorch | 2.5.1 |
| PyTorch Geometric | latest |
| torch_scatter | (matched to torch) |
| torch_sparse | (matched to torch) |
| numpy | latest |
| scipy | latest |
| matplotlib | latest |
| tqdm | latest |
| networkx | latest |

---





---

## Authors

| Name | Program | Email |
|------|---------|-------|
| Aarav Singh Luthra | CSAI, Plaksha University | aarav.luthra.ug23@plaksha.edu.in |
| Harkirat Singh | DSEB, Plaksha University | harkirat.singh.ug23@plaksha.edu.in |
| Saksham Bali | CSAI, Plaksha University | saksham.bali.ug23@plaksha.edu.in |
| Uday Sodhi | CSAI, Plaksha University | uday.sodhi.ug23@plaksha.edu.in |

---

<p align="center">
  Built on <a href="https://arxiv.org/abs/2108.04430">ToupleGDD</a> · Dataset from <a href="https://snap.stanford.edu/data/com-LiveJournal.html">SNAP LiveJournal</a>
</p>
