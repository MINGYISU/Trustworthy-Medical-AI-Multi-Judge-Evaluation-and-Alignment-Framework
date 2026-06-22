import numpy as np
import pandas as pd
from itertools import combinations


# ===========================================================================
# PART 1: JUDGE SELECTION
# ===========================================================================
 
# ---------------------------------------------------------------------------
# Signal 1: Consensus Alignment
# ---------------------------------------------------------------------------
 
def majority_vote(scores: np.ndarray) -> np.ndarray:
    """
    Majority vote across judges for each (input, generator, evaluator) cell.
    scores : (n_inputs, n_judges, n_generators, n_evaluators)
    returns: (n_inputs, n_generators, n_evaluators)  -- binary consensus
    """
    return (scores.mean(axis=1) >= 0.5).astype(int)
 
 
def consensus_alignment(scores: np.ndarray) -> np.ndarray:
    """
    For each judge: fraction of cells where they agree with majority vote.
    High = judge tracks the panel consensus well.
 
    Returns: (n_judges,) array of alignment scores in [0, 1]
    """
    n_inputs, n_judges, n_generators, n_evaluators = scores.shape
    consensus = majority_vote(scores)   # (I, G, E)
 
    alignments = np.empty(n_judges)
    for j in range(n_judges):
        alignments[j] = (scores[:, j, :, :] == consensus).mean()
    return alignments
 
 
# ---------------------------------------------------------------------------
# Signal 2: Discrimination Power
# ---------------------------------------------------------------------------
 
def discrimination_power(scores: np.ndarray) -> np.ndarray:
    """
    How well does each judge *separate* generators from each other?
 
    For each judge, compute the variance of per-generator mean scores.
    High variance = judge makes meaningful distinctions between generators.
    Near-zero = judge scores everything the same (useless for ranking).
 
    Returns: (n_judges,) array — higher is better
    """
    n_inputs, n_judges, n_generators, n_evaluators = scores.shape
    power = np.empty(n_judges)
    for j in range(n_judges):
        # Per-generator mean for this judge, averaged over inputs & evaluators
        gen_means = scores[:, j, :, :].mean(axis=(0, 2))   # (n_generators,)
        power[j] = gen_means.var()
    return power
 
 
# ---------------------------------------------------------------------------
# Signal 3: Self-consistency (cross-evaluator stability)
# ---------------------------------------------------------------------------
 
def self_consistency(scores: np.ndarray) -> np.ndarray:
    """
    For each judge, measure how consistently they score the same
    (input, generator) across different evaluator dimensions.
 
    Lower std across evaluators = more internally consistent judge.
    We return 1 - normalized_std so higher = better (consistent with other signals).
 
    Returns: (n_judges,) array in [0, 1], higher = more consistent
    """
    n_inputs, n_judges, n_generators, n_evaluators = scores.shape
    consistency = np.empty(n_judges)
    for j in range(n_judges):
        # std across evaluator dim for each (input, generator) cell
        std_per_cell = scores[:, j, :, :].std(axis=-1)   # (n_inputs, n_generators)
        # max possible std for binary is 0.5 (50/50 split)
        consistency[j] = 1.0 - std_per_cell.mean() / 0.5
    return consistency
 
 
# ---------------------------------------------------------------------------
# Signal 4: Positive Rate Deviation (bias penalty)
# ---------------------------------------------------------------------------
 
def bias_penalty(scores: np.ndarray) -> np.ndarray:
    """
    How far is each judge's positive rate from the panel median?
    Large deviation = judge is systematically lenient or strict.
 
    Returns: (n_judges,) array — lower deviation is better (0 = unbiased)
    We return it as a penalty (higher = more biased).
    """
    n_judges = scores.shape[1]
    pos_rates = np.array([scores[:, j, :, :].mean() for j in range(n_judges)])
    median_rate = np.median(pos_rates)
    return np.abs(pos_rates - median_rate)
 
 
# ---------------------------------------------------------------------------
# Judge Ranking: combine all signals
# ---------------------------------------------------------------------------
 
def rank_judges(
    scores: np.ndarray,
    judge_names: list,
    weights: dict = None,
) -> pd.DataFrame:
    """
    Combine all signals into a single judge quality score.
 
    weights: dict with keys 'alignment', 'discrimination', 'consistency', 'bias'
             Values are relative weights (will be normalized).
             Default weights treat all signals equally.
    """
    if weights is None:
        weights = {
            "alignment":      1.0,   # agrees with consensus
            "discrimination": 1.0,   # separates generators meaningfully
            "consistency":    1.0,   # stable across evaluator dims
            "bias":          -1.0,   # penalize deviation from median rate
        }
 
    n_judges = scores.shape[1]
 
    raw = {
        "alignment":      consensus_alignment(scores),
        "discrimination": discrimination_power(scores),
        "consistency":    self_consistency(scores),
        "bias_penalty":   bias_penalty(scores),
    }
 
    # Normalize each signal to [0, 1] across judges for fair combination
    def normalize(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi > lo else np.ones_like(arr) * 0.5
 
    norm = {k: normalize(v) for k, v in raw.items()}
 
    # Weighted sum (bias penalty is subtracted)
    score = (
          weights["alignment"]      * norm["alignment"]
        + weights["discrimination"] * norm["discrimination"]
        + weights["consistency"]    * norm["consistency"]
        + weights["bias"]           * norm["bias_penalty"]   # negative weight
    )
 
    rows = []
    for j in range(n_judges):
        rows.append({
            "judge":           judge_names[j],
            "alignment":       raw["alignment"][j],
            "discrimination":  raw["discrimination"][j],
            "consistency":     raw["consistency"][j],
            "bias_penalty":    raw["bias_penalty"][j],
            "composite_score": score[j],
        })
 
    df = pd.DataFrame(rows).set_index("judge")
    df["rank"] = df["composite_score"].rank(ascending=False).astype(int)
    return df.sort_values("composite_score", ascending=False)
 
 
# ===========================================================================
# PART 2: GENERATOR SELECTION
# ===========================================================================
 
# ---------------------------------------------------------------------------
# Compute judge-weighted consensus scores
# ---------------------------------------------------------------------------
 
def judge_weights_from_ranking(judge_ranking: pd.DataFrame) -> np.ndarray:
    """
    Convert composite scores to normalized weights for a judge ensemble.
    Poor judges get downweighted; the best judge gets the most say.
    """
    scores = judge_ranking["composite_score"].values
    # Shift to positive if needed, then softmax-style normalization
    shifted = scores - scores.min() + 1e-6
    return shifted / shifted.sum()
 
 
def weighted_generator_scores(
    scores: np.ndarray,
    judge_weights: np.ndarray,
    eval_names: list,
) -> pd.DataFrame:
    """
    For each generator, compute a weighted average score per evaluator dimension,
    using judge_weights to give trusted judges more influence.
 
    scores       : (n_inputs, n_judges, n_generators, n_evaluators)
    judge_weights: (n_judges,) — sum to 1
 
    Returns: DataFrame of shape (n_generators, n_evaluators + aggregates)
    """
    n_inputs, n_judges, n_generators, n_evaluators = scores.shape
 
    # Weighted mean over judges and inputs: shape (n_generators, n_evaluators)
    # scores[:, j, :, :] shape: (n_inputs, n_generators, n_evaluators)
    weighted = sum(
        judge_weights[j] * scores[:, j, :, :].mean(axis=0)   # (n_generators, n_evaluators)
        for j in range(n_judges)
    )
 
    df = pd.DataFrame(weighted, columns=eval_names)
 
    # Three aggregation strategies across evaluator dims
    df["mean_score"]    = df[eval_names].mean(axis=1)
    df["min_score"]     = df[eval_names].min(axis=1)    # weakest-link
    df["weighted_rank"] = df["mean_score"].rank(ascending=False).astype(int)
 
    return df.sort_values("mean_score", ascending=False)
 
 
def rank_generators(
    scores: np.ndarray,
    judge_weights: np.ndarray,
    gen_names: list,
    eval_names: list,
    eval_weights: np.ndarray = None,
) -> pd.DataFrame:
    """
    Full generator ranking with:
      - per-evaluator score (weighted by judge quality)
      - mean aggregation
      - min aggregation (weakest-link criterion)
      - optional custom evaluator weights
 
    eval_weights: (n_evaluators,) — relative importance of each eval dim.
                  None = treat all dimensions equally.
    """
    n_inputs, n_judges, n_generators, n_evaluators = scores.shape
 
    if eval_weights is None:
        eval_weights = np.ones(n_evaluators) / n_evaluators
    else:
        eval_weights = np.array(eval_weights, dtype=float)
        eval_weights /= eval_weights.sum()
 
    # Weighted mean over judges and inputs
    weighted = sum(
        judge_weights[j] * scores[:, j, :, :].mean(axis=0)
        for j in range(n_judges)
    )  # (n_generators, n_evaluators)
 
    rows = []
    for g in range(n_generators):
        row = {"generator": gen_names[g]}
        for e, ename in enumerate(eval_names):
            row[ename] = weighted[g, e]
        row["mean_score"]         = weighted[g].mean()
        row["weighted_eval_score"]= (weighted[g] * eval_weights).sum()
        row["min_score"]          = weighted[g].min()
        rows.append(row)
 
    df = pd.DataFrame(rows).set_index("generator")
    df["rank_mean"]   = df["mean_score"].rank(ascending=False).astype(int)
    df["rank_min"]    = df["min_score"].rank(ascending=False).astype(int)
    df["rank_weighted"] = df["weighted_eval_score"].rank(ascending=False).astype(int)
    return df.sort_values("mean_score", ascending=False)
 
 
# ===========================================================================
# Full Report
# ===========================================================================
 
def selection_report(
    scores: np.ndarray,
    judge_names: list,
    gen_names: list,
    eval_names: list,
    judge_weights_override: dict = None,
    eval_weights: np.ndarray = None,
):
    SEP = "=" * 64
    DIV = "─" * 48
 
    n_inputs, n_judges, n_generators, n_evaluators = scores.shape
 
    print(SEP)
    print("  JUDGE & GENERATOR SELECTION REPORT")
    print(SEP)
 
    # ── PART 1: Judge Ranking ───────────────────────────────────────────────
    print(f"\n{DIV}")
    print("PART 1 — JUDGE RANKING")
    print(DIV)
    print("""
  Signals (all normalized to [0,1] per judge before combining):
  ┌─────────────────────┬─────────────────────────────────────────────┐
  │ alignment           │ agreement with majority vote consensus       │
  │ discrimination      │ variance of per-generator scores (spread)   │
  │ consistency         │ stability across evaluator dimensions        │
  │ bias_penalty        │ deviation of positive rate from panel median │
  └─────────────────────┴─────────────────────────────────────────────┘
  composite = alignment + discrimination + consistency - bias_penalty
""")
 
    judge_ranking = rank_judges(scores, judge_names, judge_weights_override)
    print(judge_ranking.round(4).to_string())
 
    best_judge = judge_ranking.index[0]
    worst_judge = judge_ranking.index[-1]
    print(f"\n  ✓ Best judge  : {best_judge}  (rank 1)")
    print(f"  ✗ Worst judge : {worst_judge}  (rank {n_judges})")
 
    # Explain what drove the ranking
    print(f"\n  Why {worst_judge} ranks last:")
    row = judge_ranking.loc[worst_judge]
    signals = {
        "alignment": row["alignment"],
        "discrimination": row["discrimination"],
        "consistency": row["consistency"],
        "bias_penalty": row["bias_penalty"],
    }
    for sig, val in sorted(signals.items(), key=lambda x: x[1]):
        print(f"    {sig:20s}: {val:.4f}")
 
    # ── PART 2: Generator Ranking ───────────────────────────────────────────
    print(f"\n{DIV}")
    print("PART 2 — GENERATOR RANKING  (judge-quality-weighted)")
    print(DIV)
 
    judge_w = judge_weights_from_ranking(judge_ranking)
    print(f"\n  Judge weights (derived from composite scores):")
    for jn, w in zip(judge_ranking.index, judge_w):
        bar = "█" * int(w * 80)
        print(f"    {jn:15s}: {w:.4f}  {bar}")
 
    gen_ranking = rank_generators(
        scores, judge_w, gen_names, eval_names, eval_weights
    )
    print(f"\n  Scores per generator (weighted judge ensemble):\n")
    print(gen_ranking.round(4).to_string())
 
    best_gen_mean = gen_ranking["mean_score"].idxmax()
    best_gen_min  = gen_ranking["min_score"].idxmax()
 
    print(f"\n  ✓ Best generator (mean score)        : {best_gen_mean}")
    print(f"  ✓ Best generator (weakest-link / min): {best_gen_min}")
    if best_gen_mean != best_gen_min:
        print(f"  ⚠ They differ — {best_gen_mean} has higher average quality,")
        print(f"    but {best_gen_min} is more balanced across eval dimensions.")
 
    # ── Per-evaluator winner ────────────────────────────────────────────────
    print(f"\n  Best generator per evaluator dimension:")
    for e in eval_names:
        winner = gen_ranking[e].idxmax()
        val    = gen_ranking[e].max()
        print(f"    {e:15s}: {winner}  ({val:.4f})")
 
    print(f"\n{SEP}\n")
 
    return {
        "judge_ranking":     judge_ranking,
        "judge_weights":     dict(zip(judge_ranking.index, judge_w)),
        "generator_ranking": gen_ranking,
    }
 