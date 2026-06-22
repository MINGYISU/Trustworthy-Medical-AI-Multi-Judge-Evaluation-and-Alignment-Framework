import numpy as np
import pandas as pd
from itertools import combinations

def cell_majority_agreement(judge_votes: np.ndarray) -> float:
    """
    Given a 1D array of binary votes from n judges on ONE item,
    return the fraction that agree with the majority vote.
 
    E.g. [1,1,0,1] -> majority=1, 3/4 agree -> 0.75
         [1,0,1,0] -> tie      , 2/4 agree -> 0.50  (maximum disagreement)
 
    Range: [0.5, 1.0] for binary. Subtract 0.5 and scale to [0,1] if desired.
    """
    n = len(judge_votes)
    majority = int(np.sum(judge_votes) >= n / 2)
    return np.sum(judge_votes == majority) / n

def cell_normalized_entropy(judge_votes: np.ndarray) -> float:
    """
    Normalized entropy of the vote distribution over binary outcomes.
    0 = full agreement, 1 = maximum disagreement (50/50 split).
    """
    n = len(judge_votes)
    p = np.sum(judge_votes) / n
    if p == 0 or p == 1:
        return 0.0
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return h   # already in [0,1] for binary
 
 
def cell_pairwise_agreement(judge_votes: np.ndarray) -> float:
    """
    Average pairwise agreement across all judge pairs for one cell.
    Equivalent to 1 - probability two random judges disagree.
    """
    n = len(judge_votes)
    if n < 2:
        return 1.0
    agree_count = sum(
        judge_votes[i] == judge_votes[j]
        for i, j in combinations(range(n), 2)
    )
    return agree_count / (n * (n - 1) / 2)

def build_agreement_tensor(
    scores: np.ndarray,
    metric: str = "majority",
) -> np.ndarray:
    """
    For every (input, generator, evaluator) cell, compute how much
    the judges agree on that specific case.
 
    scores : (n_inputs, n_judges, n_generators, n_evaluators)
    metric : 'majority'  -> fraction agreeing with majority  [0.5, 1.0]
             'entropy'   -> normalized entropy                [0, 1], 0=agreement
             'pairwise'  -> avg pairwise agreement            [0, 1]
 
    Returns: agreement tensor of shape (n_inputs, n_generators, n_evaluators)
    """
    n_inputs, n_judges, n_generators, n_evaluators = scores.shape
    fn = {
        "majority": cell_majority_agreement,
        "entropy":  cell_normalized_entropy,
        "pairwise": cell_pairwise_agreement,
    }[metric]
 
    tensor = np.empty((n_inputs, n_generators, n_evaluators))
    for i in range(n_inputs):
        for g in range(n_generators):
            for e in range(n_evaluators):
                votes = scores[i, :, g, e]   # shape (n_judges,)
                tensor[i, g, e] = fn(votes)
 
    return tensor

def marginal_agreement(
    agreement_tensor: np.ndarray,
    axis_names: tuple,
    metric_name: str,
    higher_is_better: bool,
) -> dict[str, pd.Series]:
    """
    Given agreement_tensor of shape (n_inputs, n_generators, n_evaluators),
    average over all-but-one axis to get per-axis agreement profiles.
 
    Returns a dict with keys matching axis_names, each a pd.Series.
    """
    dim_labels = {
        "inputs":     {"collapse": (1, 2), "axis_idx": 0},
        "generators": {"collapse": (0, 2), "axis_idx": 1},
        "evaluators": {"collapse": (0, 1), "axis_idx": 2},
    }
    results = {}
    for name, cfg in dim_labels.items():
        mean_over = agreement_tensor.mean(axis=cfg["collapse"])
        idx = axis_names[cfg["axis_idx"]]
        results[name] = pd.Series(mean_over, index=idx, name=metric_name)
 
    return results
 
def disagreement_hotspots(
    agreement_tensor: np.ndarray,
    input_names, gen_names, eval_names,
    top_k: int = 10,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Find the top-k (input, generator, evaluator) cells with the
    most judge disagreement.
 
    higher_is_better=True  for majority/pairwise metrics (sort ascending)
    higher_is_better=False for entropy metric              (sort descending)
    """
    n_inputs, n_generators, n_evaluators = agreement_tensor.shape
    rows = []
    for i in range(n_inputs):
        for g in range(n_generators):
            for e in range(n_evaluators):
                rows.append({
                    "input":     input_names[i],
                    "generator": gen_names[g],
                    "evaluator": eval_names[e],
                    "agreement": agreement_tensor[i, g, e],
                })
    df = pd.DataFrame(rows)
    ascending = higher_is_better   # lowest agreement = worst
    return df.sort_values("agreement", ascending=ascending).head(top_k).reset_index(drop=True)
 
 
# ---------------------------------------------------------------------------
# 5. Pairwise judge agreement profile (stratified)
# ---------------------------------------------------------------------------
 
def pairwise_judge_profiles(
    scores: np.ndarray,
    judge_names: list,
) -> dict[tuple, np.ndarray]:
    """
    For each pair of judges, compute their per-cell agreement as a tensor
    of shape (n_inputs, n_generators, n_evaluators).
 
    This lets you see WHERE a pair disagrees, not just their average kappa.
    """
    n_inputs, n_judges, n_generators, n_evaluators = scores.shape
    profiles = {}
    for i, j in combinations(range(n_judges), 2):
        # Element-wise match: 1 where they agree, 0 where they don't
        match = (scores[:, i, :, :] == scores[:, j, :, :]).astype(float)
        profiles[(judge_names[i], judge_names[j])] = match
    return profiles
 
 
def pairwise_marginals(
    profiles: dict[tuple, np.ndarray],
    input_names, gen_names, eval_names,
) -> dict[str, pd.DataFrame]:
    """
    For each pair, compute average agreement marginals.
    Returns DataFrames keyed by 'inputs', 'generators', 'evaluators'.
    """
    result = {"inputs": {}, "generators": {}, "evaluators": {}}
    for (j1, j2), match in profiles.items():
        label = f"{j1} vs {j2}"
        result["inputs"][label]     = pd.Series(match.mean(axis=(1, 2)), index=input_names)
        result["generators"][label] = pd.Series(match.mean(axis=(0, 2)), index=gen_names)
        result["evaluators"][label] = pd.Series(match.mean(axis=(0, 1)), index=eval_names)
 
    return {k: pd.DataFrame(v) for k, v in result.items()}

def stratified_report(
    scores: np.ndarray,
    judge_names=None,
    input_names=None,
    gen_names=None,
    eval_names=None,
    metric: str = "pairwise",
    top_k: int = 8,
):
    """
    Main entry point. Runs stratified agreement analysis on a 4D array.
 
    scores : (n_inputs, n_judges, n_generators, n_evaluators), binary
    metric : 'majority' | 'entropy' | 'pairwise'
    """
    n_inputs, n_judges, n_generators, n_evaluators = scores.shape
 
    judge_names = judge_names or [f"Judge_{j}"  for j in range(n_judges)]
    input_names = input_names or [f"Input_{i}"  for i in range(n_inputs)]
    gen_names   = gen_names   or [f"Gen_{g}"    for g in range(n_generators)]
    eval_names  = eval_names  or [f"Eval_{e}"   for e in range(n_evaluators)]
 
    higher_is_better = metric in ("majority", "pairwise")
    metric_label = {
        "majority": "Majority Agreement  [0.5=worst, 1.0=perfect]",
        "pairwise": "Pairwise Agreement  [0.0=worst, 1.0=perfect]",
        "entropy":  "Normalized Entropy  [0.0=perfect, 1.0=worst]",
    }[metric]
 
    # ── Build agreement tensor ──────────────────────────────────────────────
    atensor = build_agreement_tensor(scores, metric=metric)
 
    # ── Marginals ───────────────────────────────────────────────────────────
    marginals = marginal_agreement(
        atensor,
        axis_names=(input_names, gen_names, eval_names),
        metric_name=metric,
        higher_is_better=higher_is_better,
    )
 
    # ── Pairwise judge profiles ─────────────────────────────────────────────
    profiles   = pairwise_judge_profiles(scores, judge_names)
    pair_margs = pairwise_marginals(profiles, input_names, gen_names, eval_names)
 
    # ── Hotspots ────────────────────────────────────────────────────────────
    hotspots = disagreement_hotspots(
        atensor, input_names, gen_names, eval_names,
        top_k=top_k, higher_is_better=higher_is_better,
    )
 
    # ── Print ───────────────────────────────────────────────────────────────
    SEP = "=" * 64
    DIV = "─" * 48
 
    print(SEP)
    print("  STRATIFIED JUDGE AGREEMENT REPORT")
    print(f"  Metric: {metric_label}")
    print(SEP)
    print(f"\n  Shape : {scores.shape}")
    print(f"  Judges: {judge_names}")
    print(f"  Overall mean agreement: {atensor.mean():.4f}  "
          f"(std={atensor.std():.4f})\n")
 
    # — Per-input —
    print(DIV)
    print("1. Agreement by INPUT  (avg over all generators × evaluators)")
    print("   → Low values = inputs where judges systematically disagree")
    print()
    df_in = marginals["inputs"].sort_values(ascending=higher_is_better)
    for name, val in df_in.items():
        bar = "█" * int(val * 20)
        print(f"   {name:12s}  {val:.4f}  {bar}")
 
    # — Per-generator —
    print()
    print(DIV)
    print("2. Agreement by GENERATOR  (avg over all inputs × evaluators)")
    print("   → Low values = generators whose outputs judges rate inconsistently")
    print()
    df_gen = marginals["generators"].sort_values(ascending=higher_is_better)
    for name, val in df_gen.items():
        bar = "█" * int(val * 20)
        print(f"   {name:12s}  {val:.4f}  {bar}")
 
    # — Per-evaluator —
    print()
    print(DIV)
    print("3. Agreement by EVALUATOR DIMENSION  (avg over all inputs × generators)")
    print("   → Low values = evaluation criteria that are inherently ambiguous")
    print()
    df_ev = marginals["evaluators"].sort_values(ascending=higher_is_better)
    for name, val in df_ev.items():
        bar = "█" * int(val * 20)
        print(f"   {name:12s}  {val:.4f}  {bar}")
 
    # — Pairwise judge profiles across dims —
    print()
    print(DIV)
    print("4. PAIRWISE JUDGE AGREEMENT  by evaluator dimension")
    print("   (each cell = fraction of inputs×generators where this pair agrees)")
    print()
    print(pair_margs["evaluators"].round(3).to_string())
 
    print()
    print(DIV)
    print("5. PAIRWISE JUDGE AGREEMENT  by generator")
    print()
    print(pair_margs["generators"].round(3).to_string())
 
    # — Hotspots —
    print()
    print(DIV)
    word = "lowest" if higher_is_better else "highest"
    print(f"6. TOP-{top_k} DISAGREEMENT HOTSPOTS  ({word} agreement cells)")
    print("   These specific (input, generator, evaluator) combos are where")
    print("   judges disagree most — worth human review or judge calibration.")
    print()
    print(hotspots.to_string(index=False))
 
    print(f"\n{SEP}\n")
 
    return {
        "agreement_tensor":      atensor,
        "marginals":             marginals,
        "pairwise_profiles":     profiles,
        "pairwise_marginals":    pair_margs,
        "hotspots":              hotspots,
    }