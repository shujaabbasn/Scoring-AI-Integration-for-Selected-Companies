import json
import csv
from collections import defaultdict
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression 
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Updated fields based on your new JSONL structure
NESTED_FIELDS = ["direction", "topics", "timeline", "aggressiveness"]
# Added the new weight sums found in your sample
SCALAR_FIELDS = ["adoption_weight_sum", "raw_adoption_weight_sum"]

def flatten(record):
    """Flatten nested dicts into dot-notation keys."""
    flat = {}
    for field in NESTED_FIELDS:
        if field in record:
            for k, v in record[field].items():
                flat[f"{field}.{k}"] = v
    for field in SCALAR_FIELDS:
        if field in record:
            flat[field] = record[field]
    return flat

def read_data(infile="data.jsonl"):
    """
    Groups data by (CIK, Year). 
    Calculates Mean, Min, and Max for every feature per group.
    """
    # Key is (cik, year)
    grouped_data = defaultdict(list)
    
    with open(infile, 'r') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            # Ensure CIK is padded to 10 digits to match truth data
            cik = record["cik"].zfill(10)
            year = int(record["year"])
            grouped_data[(cik, year)].append(flatten(record))

    rows = []
    for (cik, year), records in grouped_data.items():
        # Get all possible keys from the first record to ensure consistency
        feature_keys = records[0].keys()
        
        n = len(records)
        stats_dict = {"cik": cik, "year": year, "n": n}
        
        for k in feature_keys:
            vals = [r[k] for r in records]
            stats_dict[f"{k}.mean"] = sum(vals) / n
            stats_dict[f"{k}.min"] = min(vals)
            stats_dict[f"{k}.max"] = max(vals)
            
        rows.append(stats_dict)

    print(f"Processed {len(rows)} (CIK, Year) groups.")
    return rows

def read_target(infile="truth_data.csv"):
    """
    Maps (CIK, Year) pairs to the ai_score.
    """
    targets = {}
    with open(infile, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create a tuple key of (Padded CIK, Year)
            cik = row["cik"].zfill(10)
            year = int(row["year"])
            targets[(cik, year)] = float(row["ai_score"])
    return targets

def plot_correlation_heatmap(df_means):
    # Set the size of the plot
    plt.figure(figsize=(12, 10))
    
    # Create the heatmap
    sns.heatmap(
        df_means.corr(), 
        annot=True,          # Show the correlation coefficient numbers
        fmt=".2f",           # Round to 2 decimal places
        cmap='coolwarm',     # Red for positive, Blue for negative
        center=0,            # Ensure 0 is the neutral color
        square=True,         # Make cells square
        linewidths=.5        # Add a tiny gap between cells
    )
    
    plt.title("Cross-Correlation of Feature Means")
    plt.savefig('corr.png', dpi=300, bbox_inches='tight')
    # plt.show()
    

def analyze(rows, targets):
    df = pd.DataFrame(rows)
    
    # Create a matching key in the dataframe to join with targets
    df["match_key"] = list(zip(df["cik"], df["year"]))
    
    # Filter for rows that exist in our truth data
    df = df[df["match_key"].isin(targets.keys())].copy()
    df["target"] = df["match_key"].map(targets)

    if df.empty:
        print("No matching records found between features and truth data.")
        return

    # Drop non-feature columns
    X = df.drop(columns=["cik", "year", "match_key", "target"])
    y = np.log1p(df["target"])

    X = X.fillna(0)

    X = X.drop(columns=[col for col in df.columns if col.endswith(('.min', '.max'))])
    # X = X.drop(columns=['adoption_weight_sum', 'raw_adoption_weight_sum'])
    # # Identify base feature names (without .min, .mean, .max)
    # base_features = set(col.rsplit('.', 1)[0] for col in X.columns if '.' in col)

    # print("\n--- Correlation: Min vs Max ---")
    # for base in base_features:
    #     min_col = f"{base}.min"
    #     max_col = f"{base}.max"
    #     mean_col = f"{base}.mean"
    #     # print(f"{base}")
    #     # print(f"{X[min_col]}")
    #     # print(f"{X[max_col]}")
    #     # print(f"{X[mean_col]}")
        
    #     if min_col in X.columns and max_col in X.columns:
    #         correlation = X[min_col].corr(X[max_col])
    #         print(f"{base:30} | Correlation: {correlation:.4f}")

    # mean_cols = [col for col in X.columns if col.endswith('.mean')]
    # df_means = X[mean_cols]
    # cols_to_drop = ["adoption_weight_sum.mean", "raw_adoption_weight_sum.mean"]
    # df_means = df_means.drop(columns=cols_to_drop, errors='ignore')
    # plot_correlation_heatmap(df_means)

    # Calculate the correlation matrix
    # corr_matrix = df_means.corr()
    # print(f"{corr_matrix}")

    # # To compare multiple features at once:
    # ranges_df = pd.DataFrame()
    # for base in base_features:
    #     ranges_df[base] = X[f"{base}.max"] - X[f"{base}.min"]

    # plt.figure(figsize=(12, 6))
    # sns.boxplot(data=ranges_df)
    # plt.xticks(rotation=45)
    # plt.title("Range Comparison Across All Features")
    # plt.show()
    print(f"Fitting model on {len(df)} samples...")

    # --- LASSO ANALYSIS ---
    model = Lasso(alpha=0.01)
    model.fit(X, y)
    
    coefficients = dict(zip(X.columns, model.coef_))
    surviving = {k: v for k, v in coefficients.items() if v != 0}
    
    print(f"\nLasso Surviving features ({len(surviving)}):")
    for k, v in surviving.items():
        print(f"  {k}: {v:.4f}")

    # --- K-BEST ANALYSIS ---
    for k in [1, 2, 3, 4, 5]:
        # Handle cases where we might have fewer features than K
        k_val = min(k, X.shape[1])
        print(f"k_val: {k_val}, k: {k}, shape: {X.shape[1]}")
        selector = SelectKBest(f_regression, k=k_val)
        X_selected = selector.fit_transform(X, y)
        selected_cols = X.columns[selector.get_support()]
        print(f"Selected: {selected_cols}")
        lr_model = LinearRegression()
        lr_model.fit(X_selected, y)
        print(f'Coefficients: {lr_model.coef_}')
        print(f'Intercept: {lr_model.intercept_}')
        y_pred = lr_model.predict(X_selected)
        
        r, p = stats.pearsonr(y_pred, y)
        print(f"\nSelectKBest (k={k_val}):")
        print(f"  Features: {list(selected_cols)}")
        print(f"  Pearson r: {r:.4f} (p={p:.4f})")

# Execute
if __name__ == "__main__":
    # Ensure filenames match your local environment
    feature_rows = read_data("llm_outputs/adoption_output_039.jsonl") 
    with open("aggregated_output.jsonl", "w") as f:
        for row in feature_rows:
            # Convert dict to JSON string and add a newline
            json_record = json.dumps(row)
            f.write(json_record + "\n")
    truth_targets = read_target("truth_data.csv")
    analyze(feature_rows, truth_targets)