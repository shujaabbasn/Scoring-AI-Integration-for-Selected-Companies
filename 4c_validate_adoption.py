import pandas as pd
import numpy as np
import json
import re
import os
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import argparse

# Configuration
TRUTH_FILE = "truth_data.csv"
REPORT_FILE = "Batch_A_Report_By_Year.xlsx"
STATS_FILE = "Batch_A_Stats.txt"

def validate_adoption(aifile):
    print(f"📊 Validating AI Adoption Scores by CIK-Year pairs...")

    # 1️⃣ Load AI Results (JSONL)
    if not os.path.exists(aifile):
        print(f"❌ Error: {aifile} not found.")
        return

    ai_data = []

    with open(aifile, 'r') as f:
        for line in f:
            try:
                
                rec = json.loads(line)
                # print(f"processing record {rec}")
                # Logic to extract CIK and Year from filename 
                # (Assumes format like "12345_2018.txt")
                # filename = rec.get("filename", "")
                cik = rec.get("cik")
                year = rec.get("year")
                # parts = filename.split('_')
                # cik = int(parts[0])
                # year = int(parts[1].split('.')[0]) # Extract year before extension
                # print(f"Processing {cik} for year {year}")
                # Get the score (adjust key name if your JSONL uses a different one)
                adoption_score = rec.get("adoption_weight_sum") or rec.get("ai_score")

                if adoption_score is not None:
                    ai_data.append({
                        "cik": cik,
                        "year": year,
                        "ai_adoption_score": float(adoption_score)
                    })
            except (ValueError, IndexError, KeyError):
                continue
    if not ai_data:
        print("❌ No valid adoption scores found in AI file.")
        return

    # Create DataFrame and drop duplicates for the same CIK-Year pair
    df_ai = pd.DataFrame(ai_data)
    df_ai = df_ai.drop_duplicates(subset=["cik", "year"], keep="last")
    
    # 2️⃣ Load Revised Truth Data
    if not os.path.exists(TRUTH_FILE):
        print(f"❌ Error: {TRUTH_FILE} not found.")
        return

    df_truth = pd.read_csv(TRUTH_FILE)
    
    # Ensure CIK and Year are integers for matching
    df_truth["cik"] = df_truth["cik"].astype(int)
    df_ai["cik"] = df_ai["cik"].astype(int)
    df_truth["year"] = df_truth["year"].astype(int)

    # 3️⃣ Merge on CIK AND YEAR
    # This is the critical change for your multi-year structure
    merged = pd.merge(df_ai, df_truth, on=["cik", "year"], how="inner")

    if merged.empty:
        print("❌ No overlapping CIK-Year pairs found between AI and Truth data.")
        return

    # Save Excel report
    merged.to_excel(REPORT_FILE, index=False)
    print(f"✅ Excel Report saved: {REPORT_FILE} (N={len(merged)})")

    # 4️⃣ Correlations
    if len(merged) > 2:
        p_corr, p_val = pearsonr(merged["ai_adoption_score"], merged["ai_score"])
        s_corr, s_val = spearmanr(merged["ai_adoption_score"], merged["ai_score"])

        output_str = (
            f"AI ADOPTION VALIDATION RESULTS (CIK-YEAR BASIS)\n"
            f"-----------------------------------------------\n"
            f"Sample Size (N): {len(merged)}\n"
            f"Pearson Correlation:  {p_corr:.4f} (p={p_val:.4f})\n"
            f"Spearman Correlation: {s_corr:.4f} (p={s_val:.4f})\n"
            f"-----------------------------------------------\n"
        )
        print("\n" + output_str)
        with open(STATS_FILE, "w") as f:
            f.write(output_str)

        # 5️⃣ Residual Plot (Updated X-axis labels)
        coef = np.polyfit(merged["ai_adoption_score"], merged["ai_score"], 1)
        merged["predicted_truth"] = coef[0] * merged["ai_adoption_score"] + coef[1]
        merged["residual"] = merged["ai_score"] - merged["predicted_truth"]

        plt.figure(figsize=(12,6))
        # Create a string label for the X-axis: "CIK-Year"
        x_labels = merged["cik"].astype(str) + "-" + merged["year"].astype(str)
        plt.bar(x_labels, merged["residual"])
        plt.xticks(rotation=90, fontsize=8)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Residuals per CIK-Year Pair (Truth − Predicted)")
        plt.ylabel("Residual")
        plt.tight_layout()
        plt.savefig("residuals_by_year.png", dpi=300)
        print("📈 Residual plot saved: residuals_by_year.png")
    else:
        print("⚠️ Not enough data (N <= 2) for correlation calculation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('aifile', type=str, help='The JSONL file containing AI results.')
    args = parser.parse_args()
    validate_adoption(args.aifile)