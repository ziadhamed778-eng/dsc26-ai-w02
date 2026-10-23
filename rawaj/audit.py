import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it."""
    
    # 1. Check if required columns exist
    if not all(col in df.columns for col in [target, id_col, time_col]):
        raise KeyError("target, id_col, or time_col missing from DataFrame columns.")
        
    # Helper function to check if a single value is considered missing
    def is_missing(val):
        if pd.isna(val):
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        return False

    # Create a DataFrame of booleans (True if missing)
    missing_mask = df.map(is_missing)
    
    # --- Basic Counts & Missing ---
    n_rows = len(df)
    missing_dict = missing_mask.mean().to_dict()
    
    # --- Duplicates ---
    valid_ids = df.loc[~missing_mask[id_col], id_col]
    id_counts = valid_ids.value_counts()
    duplicate_ids = sorted(id_counts[id_counts > 1].index.tolist())
    
    duplicate_rows = int(df.duplicated().sum())
    
    # --- Constant Columns ---
    constant_columns = []
    for col in df.columns:
        if col == target:
            continue
        # Count distinct values excluding missing ones
        non_missing_vals = df.loc[~missing_mask[col], col]
        if non_missing_vals.nunique() <= 1:
            constant_columns.append(col)
    constant_columns = sorted(constant_columns)
    
    # --- Target / Label Analysis ---
    unlabelled_rows = int(missing_mask[target].sum())
    labelled_mask = ~missing_mask[target]
    
    if labelled_mask.sum() > 0:
        # positive rate is the share of 1s among labelled rows
        positive_rate = float((df.loc[labelled_mask, target] == 1).mean())
    else:
        positive_rate = 0.0
        
    # --- Timestamps ---
    valid_time_mask = ~missing_mask[time_col]
    parsed_times = pd.to_datetime(df.loc[valid_time_mask, time_col], errors="coerce")
    bad_timestamps = int(parsed_times.isna().sum())
    good_times = parsed_times.dropna()
    
    if len(good_times) > 0:
        time_range = [good_times.min().isoformat(), good_times.max().isoformat()]
    else:
        time_range = [None, None]
        
    # --- Leakage Suspects ---
    suspects = set()
    
    # We only check for leaks if we have >= 20 labelled rows and BOTH classes present
    if labelled_mask.sum() >= 20 and df.loc[labelled_mask, target].nunique() == 2:
        y = df.loc[labelled_mask, target].astype(float)
        
        for col in df.columns:
            if col in [target, id_col, time_col]:
                continue
                
            # Exclude columns with only 1 distinct value (missing counts as a value here)
            distinct_values_count = df[col].nunique(dropna=True) + (1 if missing_mask[col].any() else 0)
            if distinct_values_count <= 1:
                continue
            
            # Rule (a): Numeric or boolean columns (must have >= 20 non-missing labelled values)
            is_numeric_or_bool = pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
            if is_numeric_or_bool:
                mask_a = labelled_mask & ~missing_mask[col]
                if mask_a.sum() >= 20:
                    y_a = df.loc[mask_a, target].astype(float)
                    x_a = df.loc[mask_a, col].astype(float)
                    
                    if y_a.nunique() == 2: # roc_auc_score needs both classes
                        auc_a = roc_auc_score(y_a, x_a)
                        if auc_a >= 0.98 or auc_a <= 0.02:
                            suspects.add(col)
            
            # Rule (b): Missingness indicator (applies to ALL columns)
            x_b = missing_mask.loc[labelled_mask, col].astype(float)
            if x_b.nunique() == 2: # Only check if missingness varies (not all missing, not all present)
                auc_b = roc_auc_score(y, x_b)
                if auc_b >= 0.98 or auc_b <= 0.02:
                    suspects.add(col)

    leakage_suspects = sorted(list(suspects))

    return {
        "n_rows": n_rows,
        "missing": missing_dict,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects
    }