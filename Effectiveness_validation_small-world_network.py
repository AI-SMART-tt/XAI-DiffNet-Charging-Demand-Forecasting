import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings
import time

warnings.filterwarnings('ignore')


class AnalysisConfig:

    DATA_DIR = 'data_graph/graph2'

    EXISTING_SW_K = 3
    THRESHOLD = 0.5

    FUSION_WEIGHTS = {
        'od':       0.1,
        'cosine':   0.5,
        'distance': 0.1,
        'poi':      0.3,
    }
    K_VALUES_TO_SCAN = [1, 3, 5, 7, 9, 11, 13]

    RANDOM_GRAPH_SAMPLES = 10

    OUTPUT_DIR = './'
    OUTPUT_FILENAME = 'small_world_complete_analysis.csv'
    IMPROVEMENT_FILENAME = 'small_world_improvement_analysis.csv'


def read_csv_safe(filepath, **kwargs):
    for enc in ['utf-8', 'gbk', 'gb2312', 'latin1']:
        try:
            return pd.read_csv(filepath, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read file: {filepath}")


def load_base_data(config: AnalysisConfig):
    d = config.DATA_DIR
    print("=" * 80)
    print("  Data Loading")
    print("=" * 80)

    data = {}
    files = {
        'A':        ('adjacency_matrix.csv',            'Adjacency matrix'),
        'S_cosine': ('similarity_matrix_cosine.csv',    'Cosine similarity matrix'),
        'S_poi':    ('POI_matrix.csv',                  'POI similarity matrix'),
        'F_od':     ('OD_matrix.csv',                   'OD matrix'),
        'D':        ('distance_matrix.csv',             'Distance matrix'),
    }
    for key, (fname, desc) in files.items():
        fpath = f'{d}/{fname}'
        df = read_csv_safe(fpath, index_col=0)
        data[key] = df.values.astype(float)
        print(f"  + {desc}: {df.shape}")

    k = config.EXISTING_SW_K
    for suffix, desc in [('binary', 'binary'), ('weighted', 'weighted')]:
        fname = f'sw_fused_top_{k}_{suffix}.csv'
        fpath = f'{d}/{fname}'
        try:
            df = read_csv_safe(fpath, index_col=0)
            data[f'A_sw_{suffix}'] = df.values.astype(float)
            print(f"  + Small-world network ({desc}, K={k}): {df.shape}")
        except FileNotFoundError:
            print(f"  ! File {fname} not found, skipping")
            data[f'A_sw_{suffix}'] = None

    print(f"\n  Num nodes: {data['A'].shape[0]}")
    return data


def preprocess_matrices(S_cosine, S_poi, F_od, D, eps=1e-10):
    results = {}

    results['cosine'] = _normalize_and_clean(S_cosine)
    results['poi'] = _normalize_and_clean(S_poi)
    results['od'] = _normalize_and_clean(F_od)

    S_dist = np.zeros_like(D, dtype=float)
    mask = D > 0
    S_dist[mask] = 1.0 / (D[mask] + eps)
    results['distance'] = _normalize_and_clean(S_dist)

    print("\n  Preprocessing complete:")
    for k, v in results.items():
        print(f"    {k:10s}: [{v.min():.4f}, {v.max():.4f}]")

    return results


def _normalize_and_clean(M):
    M = M.copy().astype(float)
    scaler = MinMaxScaler()
    M_norm = scaler.fit_transform(M)
    np.fill_diagonal(M_norm, 0)
    return M_norm


def build_single_semantic_graph(S):
    return (S > 0).astype(int)


def build_fusion_full(A, matrices):
    G = A.astype(float)
    for M in matrices.values():
        G = G + M
    return (G > 0).astype(int)


def build_fusion_threshold(A, matrices, threshold=0.5):
    m = len(matrices)
    G = A.astype(float)
    for M in matrices.values():
        G = G + (1.0 / m) * M
    return (G > threshold).astype(int)


def build_fusion_topk_simple(A, matrices, k=6):
    m = len(matrices)
    G = np.zeros_like(A, dtype=float)
    for M in matrices.values():
        G += (1.0 / m) * _top_k_per_node(M, k)
    return (G > 0).astype(int)


def build_weighted_fusion_sw(A, matrices, weights, k):
    N = A.shape[0]

    S_fused = np.zeros((N, N), dtype=float)
    for key, w in weights.items():
        if w > 0 and key in matrices:
            S_fused += w * matrices[key]

    mask = (1 - A.astype(float)).astype(bool)
    np.fill_diagonal(mask, False)
    S_cand = S_fused * mask

    W_remote = np.zeros((N, N), dtype=float)
    for i in range(N):
        row = S_cand[i, :]
        if np.sum(row > 0) == 0:
            continue
        top_idx = np.argsort(row)[-k:]
        for j in top_idx:
            if row[j] > 0:
                W_remote[i, j] = row[j]

    W_combined = A.astype(float) + W_remote
    W_star = (W_combined + W_combined.T) / 2.0
    W_star_binary = (W_star > 0).astype(int)

    return W_star, W_star_binary, S_fused


def _top_k_per_node(S, k):
    n = S.shape[0]
    result = np.zeros_like(S)
    for i in range(n):
        idx = np.argsort(S[i])[-k:]
        result[i, idx] = S[i, idx]
    return np.maximum(result, result.T)


def compute_small_world_metrics(adj_matrix, graph_name, weighted=False,
                                n_random_samples=10):
    G = _matrix_to_nx(adj_matrix, weighted)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
    density = nx.density(G)

    clustering = _clustering(G, weighted)
    avg_path = _avg_path(G, weighted)
    diameter = _diameter(G)

    C_rand, L_rand = _random_graph_metrics(n_nodes, n_edges, n_random_samples)

    if C_rand > 1e-10 and L_rand > 1e-10 and avg_path > 1e-10:
        sigma = (clustering / C_rand) / (avg_path / L_rand)
    else:
        sigma = np.nan

    return {
        'Graph': graph_name,
        'Nodes': n_nodes,
        'Edges': n_edges,
        'Avg_Degree': round(avg_degree, 2),
        'Density': round(density, 6),
        'C': round(clustering, 4),
        'L': round(avg_path, 4),
        'Diameter': diameter,
        'C_rand': round(C_rand, 4),
        'L_rand': round(L_rand, 4),
        'Sigma': round(sigma, 4) if not np.isnan(sigma) else 'N/A',
    }


def _matrix_to_nx(adj, weighted=False):
    G = nx.Graph()
    n = adj.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                if weighted:
                    G.add_edge(i, j, weight=float(adj[i, j]))
                else:
                    G.add_edge(i, j)
    return G


def _clustering(G, weighted=False):
    if weighted and nx.number_of_edges(G) > 0:
        try:
            return nx.average_clustering(G, weight='weight')
        except Exception:
            return nx.average_clustering(G)
    return nx.average_clustering(G)


def _avg_path(G, weighted=False):
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if len(G) <= 1:
        return 0
    if weighted:
        try:
            return nx.average_shortest_path_length(G, weight='weight')
        except Exception:
            return nx.average_shortest_path_length(G)
    return nx.average_shortest_path_length(G)


def _diameter(G):
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if len(G) <= 1:
        return 0
    return nx.diameter(G)


def _random_graph_metrics(n, m, samples=10):
    Cs, Ls = [], []
    for _ in range(samples):
        Gr = nx.gnm_random_graph(n, m)
        Cs.append(nx.average_clustering(Gr))
        if nx.is_connected(Gr):
            Ls.append(nx.average_shortest_path_length(Gr))
        else:
            cc = max(nx.connected_components(Gr), key=len)
            Gsub = Gr.subgraph(cc).copy()
            if len(Gsub) > 1:
                Ls.append(nx.average_shortest_path_length(Gsub))
    return (np.mean(Cs) if Cs else 0), (np.mean(Ls) if Ls else 0)


def run_part_a(config, data, norm_matrices):
    print(f"\n{'='*80}")
    print(f"  Part A: Graph Structure Analysis (10 graph types)")
    print(f"{'='*80}")

    A = data['A']
    results = []
    n_rand = config.RANDOM_GRAPH_SAMPLES

    print("  Computing Adjacency graph (W_adj)...")
    results.append(compute_small_world_metrics(A, 'Adjacency graph (W_adj)',
                                               n_random_samples=n_rand))

    semantic_names = {
        'od':       'OD flow graph (W_od)',
        'cosine':   'Demand pattern graph (W_pat)',
        'poi':      'POI similarity graph (W_poi)',
        'distance': 'Distance graph (W_dis)',
    }
    for key, name in semantic_names.items():
        print(f"  Computing {name}...")
        G_single = build_single_semantic_graph(norm_matrices[key])
        results.append(compute_small_world_metrics(G_single, name,
                                                   n_random_samples=n_rand))

    print("  Computing Fused graph (unoptimized)...")
    G_full = build_fusion_full(A, norm_matrices)
    results.append(compute_small_world_metrics(G_full, 'Fused graph (unoptimized)',
                                               n_random_samples=n_rand))

    print("  Computing Fused graph (threshold)...")
    G_thresh = build_fusion_threshold(A, norm_matrices, config.THRESHOLD)
    results.append(compute_small_world_metrics(G_thresh, 'Fused graph (threshold)',
                                               n_random_samples=n_rand))

    print(f"  Computing Fused graph (top-k, k={config.EXISTING_SW_K})...")
    G_topk = build_fusion_topk_simple(A, norm_matrices, k=config.EXISTING_SW_K)
    results.append(compute_small_world_metrics(G_topk, f'Fused graph (top-k, k={config.EXISTING_SW_K})',
                                               n_random_samples=n_rand))

    if data['A_sw_binary'] is not None:
        print(f"  Computing Small-world binary (existing, k={config.EXISTING_SW_K})...")
        results.append(compute_small_world_metrics(
            data['A_sw_binary'],
            f'Small-world binary (existing, k={config.EXISTING_SW_K})',
            n_random_samples=n_rand))

    if data['A_sw_weighted'] is not None:
        print(f"  Computing Small-world weighted (existing, k={config.EXISTING_SW_K})...")
        results.append(compute_small_world_metrics(
            data['A_sw_weighted'],
            f'Small-world weighted (existing, k={config.EXISTING_SW_K})',
            weighted=True, n_random_samples=n_rand))

    return results


def run_part_b(config, data, norm_matrices):
    A = data['A']
    weights = config.FUSION_WEIGHTS
    k_values = config.K_VALUES_TO_SCAN
    n_rand = config.RANDOM_GRAPH_SAMPLES

    w_sum = sum(weights.values())
    print(f"\n{'='*80}")
    print(f"  Part B: Custom Weight Fusion + Multi-K Scan")
    print(f"{'='*80}")
    print(f"\n  Fusion weight configuration:")
    for key, w in weights.items():
        bar = '#' * int(w * 30)
        print(f"    {key:10s}: {w:.2f}  {bar}")
    print(f"    {'Total':10s}: {w_sum:.2f} {'OK' if abs(w_sum - 1.0) < 1e-6 else 'WARNING: not equal to 1!'}")
    print(f"\n  Scanning K values: {k_values}")

    if abs(w_sum - 1.0) > 0.01:
        print(f"\n  Warning: weight sum is {w_sum:.4f}, not equal to 1.0!")
        print(f"    Continuing, but results may not satisfy paper constraints.")

    results = []

    for k in k_values:
        print(f"\n  --- K = {k} ---")

        W_star, W_star_binary, S_fused = build_weighted_fusion_sw(
            A, norm_matrices, weights, k)

        name = f'Small-world graph (k={k})'

        print(f"    Computing metrics: {name}...")
        metrics = compute_small_world_metrics(
            W_star_binary, name, n_random_samples=n_rand)

        metrics['K'] = k
        metrics['Weights'] = str(weights)
        results.append(metrics)

        sigma_str = f"{metrics['Sigma']}" if metrics['Sigma'] != 'N/A' else 'N/A'
        print(f"    Edges={metrics['Edges']}, "
              f"Avg_Deg={metrics['Avg_Degree']}, "
              f"C={metrics['C']:.4f}, "
              f"L={metrics['L']:.4f}, "
              f"sigma={sigma_str}")

    return results


def summarize_and_save(part_a_results, part_b_results, config):
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in part_a_results:
        r['Section'] = 'Part_A'
        r['K'] = r.get('K', '-')
        r['Weights'] = r.get('Weights', '-')

    for r in part_b_results:
        r['Section'] = 'Part_B'

    all_results = part_a_results + part_b_results
    df_all = pd.DataFrame(all_results)

    col_order = ['Section', 'Graph', 'K', 'Weights', 'Nodes', 'Edges',
                 'Avg_Degree', 'Density', 'C', 'L', 'Diameter',
                 'C_rand', 'L_rand', 'Sigma']
    existing_cols = [c for c in col_order if c in df_all.columns]
    df_all = df_all[existing_cols]

    print(f"\n{'='*80}")
    print(f"  Complete Analysis Results")
    print(f"{'='*80}")

    print(f"\n  -- Part A: Existing Graph Structures --")
    df_a = df_all[df_all['Section'] == 'Part_A'].drop(columns=['Section', 'K', 'Weights'])
    print(df_a.to_string(index=False))

    print(f"\n  -- Part B: Custom Weight Scan --")
    print(f"  Weights: {config.FUSION_WEIGHTS}")
    df_b = df_all[df_all['Section'] == 'Part_B'].drop(columns=['Section', 'Weights'])
    print(df_b.to_string(index=False))

    print(f"\n  -- Part B Comparison Table (Paper Format) --")
    print(f"  {'K':>3} | {'Edges':>6} | {'Avg_Deg':>8} | {'C':>7} | {'L':>7} | {'Sigma':>7}")
    print(f"  {'─'*3}-+-{'─'*6}-+-{'─'*8}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}")

    baseline = part_a_results[0] if part_a_results else None

    for r in part_b_results:
        sigma_str = f"{r['Sigma']:.4f}" if r['Sigma'] != 'N/A' else '  N/A '
        print(f"  {r['K']:>3} | {r['Edges']:>6} | {r['Avg_Degree']:>8} | "
              f"{r['C']:>7.4f} | {r['L']:>7.4f} | {sigma_str}")

    if baseline:
        print(f"  {'─'*3}-+-{'─'*6}-+-{'─'*8}-+-{'─'*7}-+-{'─'*7}-+-{'─'*7}")
        b_sigma = f"{baseline['Sigma']:.4f}" if baseline['Sigma'] != 'N/A' else '  N/A '
        print(f"  {'Adj':>3} | {baseline['Edges']:>6} | {baseline['Avg_Degree']:>8} | "
              f"{baseline['C']:>7.4f} | {baseline['L']:>7.4f} | {b_sigma}  (baseline)")

    if baseline and baseline['Sigma'] != 'N/A':
        print(f"\n  -- Improvement Relative to Adjacency Graph --")
        b_C = baseline['C']
        b_L = baseline['L']

        for r in part_b_results:
            if r['C'] > 0 and r['L'] > 0 and b_C > 0 and b_L > 0:
                delta_C = (r['C'] - b_C) / b_C * 100
                delta_L = (r['L'] - b_L) / b_L * 100
                print(f"    K={r['K']:>2}: dC={delta_C:+6.1f}%, dL={delta_L:+6.1f}%")

    full_path = out_dir / config.OUTPUT_FILENAME
    df_all.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\n  Full results saved: {full_path}")

    if baseline:
        improvements = []
        all_for_improvement = part_a_results[1:] + part_b_results
        b_C = baseline['C']
        b_L = baseline['L']
        b_D = baseline['Diameter']

        for r in all_for_improvement:
            imp = {
                'Graph': r['Graph'],
                'Section': r.get('Section', ''),
                'K': r.get('K', '-'),
            }
            if b_C > 0:
                imp['dC (%)'] = round((r['C'] - b_C) / b_C * 100, 2)
            if b_L > 0:
                imp['dL (%)'] = round((r['L'] - b_L) / b_L * 100, 2)
            if b_D > 0:
                imp['dDiameter (%)'] = round(
                    (r['Diameter'] - b_D) / b_D * 100, 2)
            improvements.append(imp)

        df_imp = pd.DataFrame(improvements)
        imp_path = out_dir / config.IMPROVEMENT_FILENAME
        df_imp.to_csv(imp_path, index=False, encoding='utf-8-sig')
        print(f"  Improvement metrics saved: {imp_path}")

    return df_all


def main():
    config = AnalysisConfig()

    print("\n" + "=" * 80)
    print("  EV Charging Network Small-World Property Empirical Analysis (Full Enhanced Version)")
    print("=" * 80)
    print(f"\n  Configuration:")
    print(f"    Data directory: {config.DATA_DIR}")
    print(f"    Existing small-world network K: {config.EXISTING_SW_K}")
    print(f"    Threshold parameter: {config.THRESHOLD}")
    print(f"    Fusion weights: {config.FUSION_WEIGHTS}")
    print(f"    K values to scan: {config.K_VALUES_TO_SCAN}")
    print(f"    Random graph samples: {config.RANDOM_GRAPH_SAMPLES}")
    print(f"    Output file: {config.OUTPUT_FILENAME}")

    t_start = time.time()

    data = load_base_data(config)

    print(f"\n{'='*80}")
    print(f"  Matrix Preprocessing and Normalization")
    print(f"{'='*80}")
    norm_matrices = preprocess_matrices(
        data['S_cosine'], data['S_poi'], data['F_od'], data['D'])

    part_a_results = run_part_a(config, data, norm_matrices)
    part_b_results = run_part_b(config, data, norm_matrices)

    df_all = summarize_and_save(part_a_results, part_b_results, config)

    dt = time.time() - t_start
    print(f"\n  Total runtime: {dt//60:.0f}m {dt%60:.1f}s")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()