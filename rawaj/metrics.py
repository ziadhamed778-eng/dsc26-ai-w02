from __future__ import annotations
import math
import numpy as np

def _validate_inputs(y_true, scores):
    """Helper function to validate inputs for all metrics."""
    y = np.array(y_true, dtype=float)
    s = np.array(scores, dtype=float)
    
    if len(y) != len(s):
        raise ValueError("Lengths of y_true and scores differ.")
    if len(y) == 0:
        raise ValueError("Input lists are empty.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true contains values other than 0 or 1.")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN.")
        
    return y, s

def confusion_at(y_true, scores, threshold: float) -> dict:
    """Return {"tp": int, "fp": int, "tn": int, "fn": int} at `threshold`."""
    y, s = _validate_inputs(y_true, scores)
    
    flagged = s >= threshold
    
    tp = int(((y == 1) & flagged).sum())
    fp = int(((y == 0) & flagged).sum())
    tn = int(((y == 0) & ~flagged).sum())
    fn = int(((y == 1) & ~flagged).sum())
    
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Return (precision, recall) at `threshold`."""
    c = confusion_at(y_true, scores, threshold)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    return precision, recall

def precision_at_k(y_true, scores, k: int) -> float:
    """Share of positives among the k highest scores."""
    y, s = _validate_inputs(y_true, scores)
    
    if k <= 0:
        raise ValueError("k must be greater than 0.")
        
    n = len(y)
    if k > n:
        k = n
        
    # Rank by score descending, keeping original order on ties (stable sort)
    # Using kind='mergesort' with negated scores ensures stable descending sort
    indices = np.argsort(-s, kind='mergesort')
    top_k_indices = indices[:k]
    
    top_k_y = y[top_k_indices]
    precision = float(np.sum(top_k_y) / k)
    
    return precision

def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """Expected cost at a given threshold."""
    c = confusion_at(y_true, scores, threshold)
    return float((c["fp"] * cost_fp) + (c["fn"] * cost_fn))

def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Threshold with the lowest expected cost."""
    y, s = _validate_inputs(y_true, scores)
    
    candidates = np.unique(s).tolist() + [math.inf]
    best_cost = math.inf
    best_t = -math.inf
    
    for t in candidates:
        cost = expected_cost(y, s, float(t), cost_fp=cost_fp, cost_fn=cost_fn)
        # If lower cost, update. If tied, pick the higher threshold
        if cost < best_cost:
            best_cost = cost
            best_t = t
        elif cost == best_cost:
            if t > best_t:
                best_t = t
                
    return float(best_t)

def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest threshold whose precision is >= min_precision."""
    y, s = _validate_inputs(y_true, scores)
    
    candidates = np.unique(s)
    valid_thresholds = []
    
    for t in candidates:
        p, r = precision_recall_at(y, s, float(t))
        if p >= min_precision:
            valid_thresholds.append(t)
            
    if not valid_thresholds:
        return None
        
    return float(min(valid_thresholds))