"""
Run IMM and CELF using the exact functions provided from the notebook.
- Cost = 1
- Budget = 5
- Propagation Probability = Weighted Cascade (1 / in_degree)
"""

import random
import math
import time
import heapq
import networkx as nx
import numpy as np
from collections import defaultdict, deque

# ─── Configuration ────────────────────────────────────────────────────────────
GRAPH_PATH = "test_graph_new.txt"
BUDGET = 5.0
SIM_RUNS = 10000
SEED = 42

# ─── 1. Load graph ────────────────────────────────────────────────────────────
def load_graph(path):
    print(f"Loading graph from {path} ...")
    G = nx.DiGraph()
    nodes = set()
    raw_edges = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                nodes.add(u)
                nodes.add(v)
                raw_edges.append((u, v))

    # --- Remap node IDs to guaranteed contiguous 0..N-1 ---
    # This is required because DeepWalkNeg creates an embedding table of
    # size graph.num_nodes = len(nodes). If node IDs have gaps, embedding crashes.
    sorted_nodes = sorted(nodes)
    if len(sorted_nodes) > 0 and sorted_nodes[-1] != len(sorted_nodes) - 1:
        print("  Remapping non-contiguous node IDs to 0..N-1 ...")
        remap = {old: new for new, old in enumerate(sorted_nodes)}
        for u, v in raw_edges:
            G.add_edge(remap[u], remap[v])
            G.add_edge(remap[v], remap[u])   # Make graph bidirectional
    else:
        for u, v in raw_edges:
            G.add_edge(u, v)
            G.add_edge(v, u)   # Make graph bidirectional

    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    return G

# ─── 2. Propagation probability & Cost ───────────────────────────────────────
def propagation_prob(G, u, v):
    """
    Weighted Cascade model: p(u,v) = 1 / in_degree(v).
    This is the standard, best-practice way to assign propagation probabilities 
    when you don't have historical cascade data to learn from.
    """
    in_deg = G.in_degree(v)
    return 1.0 / in_deg if in_deg > 0 else 0.1

def node_cost(G, v):
    return 1.0


# ─── USER PROVIDED IC ────────────────────────────────────────────────────────
def independent_cascade(G, seeds, num_simulations=10000):
    """
    Estimate expected influence spread under the Independent Cascade model
    using the exact level-by-level BFS logic from computeMC.
    """
    if not seeds:
        return 0.0

    sources = set(seeds)
    total_inf = 0

    for _ in range(num_simulations):
        # Run one IC cascade from S
        activated = sources.copy()
        queue = deque(activated)
        while True:
            newly_activated = set()
            while queue:
                curr = queue.popleft()
                for child in G.successors(curr):
                    if child not in activated:
                        p = propagation_prob(G, curr, child)
                        if random.random() <= p:
                            newly_activated.add(child)
            if not newly_activated:
                break
            queue.extend(newly_activated)
            activated |= newly_activated

        total_inf += len(activated)

    return total_inf / num_simulations


# ─── USER PROVIDED CELF ──────────────────────────────────────────────────────
def celf(G, budget, num_simulations=10000, seed=42):
    """
    CELF (Cost-Effective Lazy Forward) greedy influence maximization.

    Parameters
    ----------
    G               : nx.DiGraph
    budget          : float — total spend allowed
    num_simulations : int   — Monte Carlo runs per IC evaluation
    seed            : int   — RNG seed for reproducibility

    Returns
    -------
    seeds   : list of selected seed nodes (in selection order)
    spread  : float — estimated influence spread of final seed set
    history : list of (node, cumulative_spread, cumulative_cost)
    """
    # ── Fix 1: seed RNG inside function for full reproducibility ──────────────
    random.seed(seed)
    np.random.seed(seed)

    nodes = list(G.nodes())
    seeds      = []
    seed_set   = set()
    total_cost = 0.0
    current_spread = 0.0
    history    = []

    print("CELF: Initialising marginal gains for all nodes...")

    # ── Fix 2: use node_cost() instead of G.nodes[node].get('cost') ──────────
    heap = []
    for i, node in enumerate(nodes):
        cost = node_cost(G, node)
        if cost <= budget:
            gain = independent_cascade(G, [node], num_simulations)
            heapq.heappush(heap, (-gain, 0, node))
        if (i+1) % 10000 == 0:
            print(f"  ... evaluated {i+1}/{len(nodes)} candidates")

    print(f"CELF: Heap initialised with {len(heap)} candidates.")

    round_num = 1
    while heap and total_cost < budget:
        neg_gain, last_round, node = heapq.heappop(heap)

        if node in seed_set:
            continue

        cost = node_cost(G, node)
        if total_cost + cost > budget:
            # Node doesn't fit — push back so cheaper nodes can still be considered
            # (we don't skip permanently; a later pop might fit if budget is updated)
            continue

        # ── Fix 3: correct lazy evaluation condition ──────────────────────────
        # Re-evaluate only if gain is stale (wasn't computed this round)
        if last_round < round_num - 1:
            new_gain = (
                independent_cascade(G, list(seed_set) + [node], num_simulations)
                - current_spread
            )
            heapq.heappush(heap, (-new_gain, round_num - 1, node))
            continue

        # ── Accept this node as the next seed ─────────────────────────────────
        seeds.append(node)
        seed_set.add(node)
        total_cost    += cost
        current_spread = independent_cascade(G, list(seed_set), num_simulations)
        history.append((node, current_spread, total_cost))

        print(f"  Round {round_num:2d}: node={node:8d}  cost={cost:.3f}  "
              f"total_cost={total_cost:.2f}  spread={current_spread:.1f}")

        round_num += 1

    return seeds, current_spread, history


# ─── USER PROVIDED IMM ───────────────────────────────────────────────────────
def generate_rr_set(G):
    """
    Generate one Reverse Reachable (RR) set.
    Picks a random target, then reverse-BFS following edges backwards
    with probability p(u,v) = 1/in_degree(v).
    """
    target  = random.choice(list(G.nodes()))
    visited = {target}
    queue   = [target]

    while queue:
        node = queue.pop()
        for pred in G.predecessors(node):
            if pred not in visited:
                # probability that pred could have activated node
                p = propagation_prob(G, pred, node)
                if random.random() < p:
                    visited.add(pred)
                    queue.append(pred)

    return visited


def imm(G, budget, epsilon=0.1, delta=0.1, num_simulations=10000, seed=42):
    """
    IMM — Influence Maximization via Martingales.

    Parameters
    ----------
    G               : nx.DiGraph
    budget          : float — total spend budget
    epsilon         : float — approximation error tolerance
    delta           : float — failure probability
    num_simulations : int   — Monte Carlo runs for final IC evaluation
    seed            : int   — RNG seed

    Returns
    -------
    seeds   : list of selected seed nodes
    spread  : float — estimated influence spread
    history : list of (node, cumulative_spread, cumulative_cost)
    """
    # ── Seed RNG ──────────────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)

    n = G.number_of_nodes()

    # Estimate k: max seeds affordable (sort by cost, greedily fill budget)
    sorted_nodes = sorted(G.nodes(), key=lambda v: node_cost(G, v))
    k, spent = 0, 0.0
    for node in sorted_nodes:
        c = node_cost(G, node)
        if spent + c <= budget:
            k    += 1
            spent += c
        else:
            break
    k = max(k, 1)

    # Number of RR sets (simplified IMM formula)
    Lambda = (8 + 2 * epsilon) * n * (
        (k - 1) * math.log(n) + math.log(1.0 / delta) + math.log(2)
    ) / (epsilon ** 2)
    num_rr = int(max(Lambda / n, 50))

    print(f"IMM: k≈{k}  generating {num_rr} RR sets...")

    # Generate RR sets
    rr_sets = [generate_rr_set(G) for _ in range(num_rr)]

    # Coverage index: node -> list of RR set indices it appears in
    coverage = defaultdict(list)
    for idx, rr in enumerate(rr_sets):
        for node in rr:
            coverage[node].append(idx)

    # Greedy max-coverage under budget
    seeds      = []
    seed_set   = set()
    total_cost = 0.0
    covered    = set()
    history    = []
    current_spread = 0.0

    eligible = {
        node for node in G.nodes()
        if node_cost(G, node) <= budget
    }

    round_num = 1
    while eligible:
        best_node, best_score = None, -1.0

        for node in eligible:
            cost = node_cost(G, node)
            if total_cost + cost > budget:
                continue
            # Marginal new RR sets covered per unit cost
            new_covered = sum(1 for idx in coverage[node] if idx not in covered)
            score = new_covered / cost   # coverage-per-cost

            # ── Fix: was `best_score = best_node, best_node` (tuple bug) ─────
            if score > best_score:
                best_node  = node
                best_score = score   # correctly assign float, not tuple

        if best_node is None:
            break

        cost = node_cost(G, best_node)
        seeds.append(best_node)
        seed_set.add(best_node)
        total_cost += cost
        eligible.discard(best_node)

        # Mark RR sets covered by this seed
        for idx in coverage[best_node]:
            covered.add(idx)

        # Evaluate actual IC spread for history/reporting
        current_spread = independent_cascade(G, list(seed_set), num_simulations)
        history.append((best_node, current_spread, total_cost))

        print(f"  Round {round_num:2d}: node={best_node:8d}  cost={cost:.3f}  "
              f"total_cost={total_cost:.2f}  spread={current_spread:.1f}")

        round_num += 1

        # Prune nodes that no longer fit remaining budget
        eligible = {
            node for node in eligible
            if total_cost + node_cost(G, node) <= budget
        }

    return seeds, current_spread, history


# ─── MAIN RUNNER ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G = load_graph(GRAPH_PATH)

    print(f"\n{'='*60}")
    print(f"Running IMM  budget={BUDGET}  simulations={SIM_RUNS}\n")
    t0 = time.time()
    imm_seeds, imm_spread, imm_history = imm(G, budget=BUDGET,
                                              num_simulations=SIM_RUNS, seed=SEED)
    imm_time = time.time() - t0

    print(f"\nIMM Results:")
    print(f"  Seeds selected : {imm_seeds}")
    print(f"  Total spread   : {imm_spread:.1f} nodes")
    print(f"  Runtime        : {imm_time:.2f}s")

    print(f"\n{'='*60}")
    print(f"Running CELF  budget={BUDGET}  simulations={SIM_RUNS}\n")
    t0 = time.time()
    celf_seeds, celf_spread, celf_history = celf(G, budget=BUDGET,
                                                  num_simulations=SIM_RUNS, seed=SEED)
    celf_time = time.time() - t0

    print(f"\nCELF Results:")
    print(f"  Seeds selected : {celf_seeds}")
    print(f"  Total spread   : {celf_spread:.1f} nodes")
    print(f"  Runtime        : {celf_time:.2f}s")