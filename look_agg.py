import json
import csv
from collections import defaultdict
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, LinearRegression 
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Fields to flatten and average
NESTED_FIELDS = ["direction", "topics", "timeline", "aggressiveness"]
SCALAR_FIELDS = ["significance_score", "ai_relevance_score", "overall_confidence"]

def flatten(record):
    """Flatten nested dicts into dot-notation keys."""
    flat = {}
    for field in NESTED_FIELDS:
        for k, v in record[field].items():
            flat[f"{field}.{k}"] = v
    for field in SCALAR_FIELDS:
        flat[field] = record[field]
    return flat

def analyze(rows, targets):
    # Build DataFrame from rows list
    df = pd.DataFrame(rows)
    df = df[df["cik"].isin(targets)]  # keep only CIKs with a target value
    df["target"] = df["cik"].map(targets)

    # Features and target
    X = df.drop(columns=["cik", "target"])
    y = df["target"]

    # Fit
    model = Lasso(alpha=0.01)
    model.fit(X, y)

    # See which features survived
    surviving = {k: v for k, v in zip(X.columns, model.coef_) if v != 0}
    print(surviving)
    # model = LinearRegression()
    # model.fit(X, y)

    coefficients = dict(zip(X.columns, model.coef_))

    # All coefficients
    print(coefficients)
    print("Intercept:", model.intercept_)

    # Just the ones Lasso kept
    surviving = {k: v for k, v in coefficients.items() if v != 0}
    print(f"\nSurviving features ({len(surviving)}):")
    print(surviving)

    y_pred = model.predict(X)

    pearson_r, pearson_p = stats.pearsonr(y_pred, y)
    spearman_r, spearman_p = stats.spearmanr(y_pred, y)

    print(f"Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
    print(f"Spearman: r={spearman_r:.4f}, p={spearman_p:.4f}")
    surviving_feature = list(surviving.keys())[0]

    # plt.scatter(X[surviving_feature], y)
    # plt.show()
    for k in [2, 3, 4]:
        selector = SelectKBest(f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_cols = X.columns[selector.get_support()]
        
        model = LinearRegression()
        model.fit(X_selected, y)
        y_pred = model.predict(X_selected)
        
        pearson_r, pearson_p = stats.pearsonr(y_pred, y)
        spearman_r, spearman_p = stats.spearmanr(y_pred, y)
        
        print(f"\nk={k}, Features: {list(selected_cols)}")
        print(f"Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
        print(f"Spearman: r={spearman_r:.4f}, p={spearman_p:.4f}")
        print(f"Coefficients: {dict(zip(selected_cols, model.coef_))}")

def read_data(infile = "output.txt"):
    # Accumulate values per CIK
    cik_data = defaultdict(list)
    with open(infile) as f:
        for line in f:
            record = json.loads(line.strip())
            cik = record["cik"]
            cik_data[cik].append(flatten(record))

    # Average across rows per CIK
    rows = []
    all_keys = list(flatten(json.loads(open("output.txt").readline())). keys())
    for cik, records in cik_data.items():
        n = len(records)
        avg = {f"{k}.mean": sum(r[k] for r in records) / n for k in all_keys}
        mn  = {f"{k}.min":  min(r[k] for r in records)     for k in all_keys}
        mx  = {f"{k}.max":  max(r[k] for r in records)     for k in all_keys}
        rows.append({"cik": cik, "n": n, **avg, **mn, **mx})

    # Write CSV
    fieldnames = ["cik", "n"] + [f"{k}.{stat}" for k in all_keys for stat in ["mean", "min", "max"]]
    with open("features.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows to features.csv")
    return rows

def read_target(infile = "Pilot_Truth_Data.csv"):
    
    targets = {}
    with open(infile) as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets[row["cik"].zfill(10)] = float(row["truth_score"])
        return targets
rows = read_data()
targets = read_target()
analyze(rows, targets)