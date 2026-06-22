"""
scorer.py
=========
A single class that wraps all judge & generator selection logic.

Usage
-----
    from scorer import JudgeGeneratorScorer

    scorer = JudgeGeneratorScorer(
        scores,                          # (n_inputs, n_judges, n_generators, n_evaluators)
        judge_names     = [...],
        gen_names       = [...],
        eval_names      = [...],
        eval_weights    = [2.0, 1.0, 1.0],   # optional: weight eval dims
        judge_weights   = None,               # optional: override signal weights
    )

    result = scorer.select()             # run everything, returns SelectionResult

    result.best_judge                    # str  — name of the best judge
    result.best_generator                # str  — name of the best generator (by mean)
    result.best_generator_balanced       # str  — best generator by weakest-link
    result.best_generator_per_evaluator  # dict[eval_name -> gen_name]
    result.judge_ranking                 # pd.DataFrame
    result.generator_ranking             # pd.DataFrame
    result.judge_weights                 # dict[judge_name -> float]
    result.summary()                     # prints a compact human-readable summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    best_judge:                  str
    best_generator:              str          # highest mean score
    best_generator_balanced:     str          # highest min score (weakest-link)
    best_generator_per_evaluator: dict        # {eval_name: gen_name}

    judge_ranking:      pd.DataFrame
    generator_ranking:  pd.DataFrame
    judge_weights:      dict                  # {judge_name: weight}

    # raw signal arrays, keyed by judge name
    _signals: dict = field(default_factory=dict, repr=False)

    def summary(self) -> None:
        SEP = "=" * 56
        print(SEP)
        print("  SELECTION SUMMARY")
        print(SEP)

        print("\n── Judges ───────────────────────────────────────────")
        for jname, row in self.judge_ranking.iterrows():
            marker = " ◀ best" if jname == self.best_judge else ""
            print(f"  [{int(row['rank'])}] {jname:15s}  "
                  f"score={row['composite_score']:.3f}  "
                  f"weight={self.judge_weights[jname]:.3f}{marker}")

        print("\n── Generators ───────────────────────────────────────")
        for gname, row in self.generator_ranking.iterrows():
            marker = " ◀ best (mean)" if gname == self.best_generator else ""
            if gname == self.best_generator_balanced and gname != self.best_generator:
                marker = " ◀ best (balanced)"
            print(f"  [{int(row['rank_mean'])}] {gname:12s}  "
                  f"mean={row['mean_score']:.3f}  "
                  f"min={row['min_score']:.3f}{marker}")

        print("\n── Best generator per evaluator ─────────────────────")
        for eval_name, gen_name in self.best_generator_per_evaluator.items():
            print(f"  {eval_name:15s} →  {gen_name}")

        print(f"\n  ✓ best_judge      = {self.best_judge!r}")
        print(f"  ✓ best_generator  = {self.best_generator!r}")
        if self.best_generator != self.best_generator_balanced:
            print(f"  ✓ best_balanced   = {self.best_generator_balanced!r}  "
                  f"(more even across eval dims)")
        print(SEP)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class JudgeGeneratorScorer:
    """
    Wraps judge-quality estimation and generator ranking for a 4D binary
    score array of shape (n_inputs, n_judges, n_generators, n_evaluators).

    Parameters
    ----------
    scores : np.ndarray, shape (n_inputs, n_judges, n_generators, n_evaluators)
        Binary scores (0 or 1).
    judge_names : list[str], optional
    gen_names   : list[str], optional
    eval_names  : list[str], optional
    eval_weights : array-like, optional
        Relative importance of each evaluator dimension for generator ranking.
        Defaults to equal weights.
    judge_signal_weights : dict, optional
        Weights for each judge-quality signal.
        Keys: 'alignment', 'discrimination', 'consistency', 'bias'.
        Defaults: all 1.0 except bias = -1.0 (penalty).
    """

    def __init__(
        self,
        scores: np.ndarray,
        judge_names:          Optional[list]       = None,
        gen_names:            Optional[list]       = None,
        eval_names:           Optional[list]       = None,
        eval_weights:         Optional[np.ndarray] = None,
        judge_signal_weights: Optional[dict]       = None,
    ):
        assert scores.ndim == 4, \
            f"scores must be 4D (n_inputs, n_judges, n_generators, n_evaluators), got {scores.shape}"
        assert set(np.unique(scores)).issubset({0, 1}), \
            "scores must be binary (0 or 1)"

        self.scores = scores.astype(int)
        n_inputs, n_judges, n_generators, n_evaluators = scores.shape

        self.judge_names = judge_names or [f"Judge_{j}"  for j in range(n_judges)]
        self.gen_names   = gen_names   or [f"Gen_{g}"    for g in range(n_generators)]
        self.eval_names  = eval_names  or [f"Eval_{e}"   for e in range(n_evaluators)]

        assert len(self.judge_names) == n_judges
        assert len(self.gen_names)   == n_generators
        assert len(self.eval_names)  == n_evaluators

        # Evaluator weights
        if eval_weights is None:
            self._eval_weights = np.ones(n_evaluators) / n_evaluators
        else:
            ew = np.array(eval_weights, dtype=float)
            self._eval_weights = ew / ew.sum()

        # Judge signal weights
        self._signal_weights = {
            "alignment":      1.0,
            "discrimination": 1.0,
            "consistency":    1.0,
            "bias":          -1.0,
        }
        if judge_signal_weights:
            self._signal_weights.update(judge_signal_weights)

    # -----------------------------------------------------------------------
    # Judge-quality signals
    # -----------------------------------------------------------------------

    def _majority_vote(self) -> np.ndarray:
        """Majority vote across judges -> (n_inputs, n_generators, n_evaluators)."""
        return (self.scores.mean(axis=1) >= 0.5).astype(int)

    def _consensus_alignment(self) -> np.ndarray:
        """Fraction of cells each judge agrees with the majority vote. -> (n_judges,)"""
        consensus = self._majority_vote()
        return np.array([
            (self.scores[:, j, :, :] == consensus).mean()
            for j in range(self.scores.shape[1])
        ])

    def _discrimination_power(self) -> np.ndarray:
        """Variance of per-generator mean scores per judge. -> (n_judges,)"""
        return np.array([
            self.scores[:, j, :, :].mean(axis=(0, 2)).var()
            for j in range(self.scores.shape[1])
        ])

    def _self_consistency(self) -> np.ndarray:
        """1 - normalized cross-evaluator std per judge. -> (n_judges,)  higher=better"""
        return np.array([
            1.0 - self.scores[:, j, :, :].std(axis=-1).mean() / 0.5
            for j in range(self.scores.shape[1])
        ])

    def _bias_penalty(self) -> np.ndarray:
        """Absolute deviation of positive rate from panel median. -> (n_judges,)"""
        pos_rates = np.array([
            self.scores[:, j, :, :].mean()
            for j in range(self.scores.shape[1])
        ])
        return np.abs(pos_rates - np.median(pos_rates))

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo) if hi > lo else np.full_like(arr, 0.5)

    # -----------------------------------------------------------------------
    # Judge ranking
    # -----------------------------------------------------------------------

    def _rank_judges(self) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Returns (judge_ranking_df, judge_weights_array).
        judge_weights_array is ordered to match self.judge_names.
        """
        raw = {
            "alignment":      self._consensus_alignment(),
            "discrimination": self._discrimination_power(),
            "consistency":    self._self_consistency(),
            "bias_penalty":   self._bias_penalty(),
        }

        norm = {k: self._normalize(v) for k, v in raw.items()}

        composite = (
              self._signal_weights["alignment"]      * norm["alignment"]
            + self._signal_weights["discrimination"] * norm["discrimination"]
            + self._signal_weights["consistency"]    * norm["consistency"]
            + self._signal_weights["bias"]           * norm["bias_penalty"]
        )

        df = pd.DataFrame({
            "alignment":       raw["alignment"],
            "discrimination":  raw["discrimination"],
            "consistency":     raw["consistency"],
            "bias_penalty":    raw["bias_penalty"],
            "composite_score": composite,
        }, index=self.judge_names)
        df.index.name = "judge"
        df["rank"] = df["composite_score"].rank(ascending=False).astype(int)
        df = df.sort_values("composite_score", ascending=False)

        # Weights: shift composite scores to be non-negative, then normalize
        shifted = composite - composite.min() + 1e-9
        weights = shifted / shifted.sum()   # aligned with self.judge_names order

        return df, weights

    # -----------------------------------------------------------------------
    # Generator ranking
    # -----------------------------------------------------------------------

    def _rank_generators(self, judge_weights: np.ndarray) -> pd.DataFrame:
        """
        Returns generator_ranking_df.
        judge_weights is aligned with self.judge_names (original order).
        """
        n_inputs, n_judges, n_generators, n_evaluators = self.scores.shape

        # Weighted mean over judges and inputs -> (n_generators, n_evaluators)
        weighted = sum(
            judge_weights[j] * self.scores[:, j, :, :].mean(axis=0)
            for j in range(n_judges)
        )

        rows = []
        for g, gname in enumerate(self.gen_names):
            row = {"generator": gname}
            for e, ename in enumerate(self.eval_names):
                row[ename] = weighted[g, e]
            row["mean_score"]          = float(weighted[g].mean())
            row["weighted_eval_score"] = float((weighted[g] * self._eval_weights).sum())
            row["min_score"]           = float(weighted[g].min())
            rows.append(row)

        df = pd.DataFrame(rows).set_index("generator")
        df["rank_mean"]     = df["mean_score"].rank(ascending=False).astype(int)
        df["rank_min"]      = df["min_score"].rank(ascending=False).astype(int)
        df["rank_weighted"] = df["weighted_eval_score"].rank(ascending=False).astype(int)
        return df.sort_values("mean_score", ascending=False)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def select(self) -> SelectionResult:
        """Run all analyses and return a SelectionResult."""

        judge_ranking, judge_weights_arr = self._rank_judges()
        gen_ranking = self._rank_generators(judge_weights_arr)

        best_judge     = judge_ranking.index[0]
        best_gen_mean  = gen_ranking["mean_score"].idxmax()
        best_gen_min   = gen_ranking["min_score"].idxmax()
        best_per_eval  = {e: gen_ranking[e].idxmax() for e in self.eval_names}
        judge_weights  = dict(zip(self.judge_names, judge_weights_arr))

        return SelectionResult(
            best_judge                   = best_judge,
            best_generator               = best_gen_mean,
            best_generator_balanced      = best_gen_min,
            best_generator_per_evaluator = best_per_eval,
            judge_ranking                = judge_ranking,
            generator_ranking            = gen_ranking,
            judge_weights                = judge_weights,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _make_fake_scores(n_inputs=20, n_judges=4, n_generators=5,
                      n_evaluators=3, seed=42):
    rng = np.random.default_rng(seed)
    truth = rng.binomial(1, 0.6, size=(n_inputs, n_generators, n_evaluators))
    scores = np.empty((n_inputs, n_judges, n_generators, n_evaluators), dtype=int)
    flip_probs = [0.10, 0.15, 0.20, 0.40]
    for j in range(n_judges):
        noise = rng.binomial(1, flip_probs[j], size=(n_inputs, n_generators, n_evaluators))
        scores[:, j, :, :] = np.where(noise, 1 - truth, truth)
    leniency = rng.binomial(1, 0.85, size=(n_inputs, n_generators, n_evaluators))
    scores[:, 2, :, :] = leniency
    better_truth = rng.binomial(1, 0.85, size=(n_inputs, n_evaluators))
    for j in range(n_judges):
        noise = rng.binomial(1, flip_probs[j], size=(n_inputs, n_evaluators))
        scores[:, j, 0, :] = np.where(noise, 1 - better_truth, better_truth)
    return scores


if __name__ == "__main__":
    scores = _make_fake_scores()

    scorer = JudgeGeneratorScorer(
        scores,
        judge_names  = ["GPT-4o", "Claude-3.5", "Gemini-Pro", "Gemini-Flash"],
        gen_names    = ["Qwen-2.5", "Kimi-k1.5", "DS-R1", "DS-R1-Zero", "QwQ-32B"],
        eval_names   = ["Hallucination", "Adherence", "Coherence"],
        eval_weights = [2.0, 1.0, 1.0],   # hallucination counts double
    )

    result = scorer.select()
    result.summary()

    # Direct attribute access — the main point of the class
    print("\n── Direct access ────────────────────────────────────")
    print(f"result.best_judge                   = {result.best_judge!r}")
    print(f"result.best_generator               = {result.best_generator!r}")
    print(f"result.best_generator_balanced      = {result.best_generator_balanced!r}")
    print(f"result.best_generator_per_evaluator = {result.best_generator_per_evaluator}")