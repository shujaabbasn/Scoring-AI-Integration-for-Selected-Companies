"""
STEP 4a: VALIDATE STRATEGIC INTENSITY (BATCH A)
-----------------------------------------------
1. Reads AI scores from 'ai_scores_intensity.jsonl'
2. Merges with Ground Truth from 'Pilot_Truth_Data.csv'
3. Saves Excel Report & Correlation Stats.
"""
import pandas as pd
import numpy as np
import json
import re
import os
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import argparse

# Files
# AI_FILE = "ai_scores_intensity.jsonl"
AI_FILE = "llm_outputs/compact_results.jsonl"
# TRUTH_FILE = "Pilot_Truth_Data.csv"
TRUTH_FILE = "truth_data.csv"
REPORT_FILE = "Batch_A_Report.xlsx"
STATS_FILE = "Batch_A_Stats.txt"

def validate_adoption(aifile):
    print("📊 Validating AI Adoption Scores...")

    # 1️⃣ Load AI Results (JSONL)
    if not os.path.exists(aifile):
        print(f"❌ Error: {aifile} not found.")
        return

    ai_data = []

    with open(aifile, 'r') as f:
        for line in f:
            try:
                rec = json.loads(line)

                cik = int(rec["cik"])
                adoption_score = rec.get("raw_adoption_weight_sum", None)
                # Use the following line and comment out the above if you want to use the normalized weighted intensity.
                # adoption_score = rec.get("adoption_weight_sum", None)

                if adoption_score is not None:
                    ai_data.append({
                        "cik": cik,
                        "ai_adoption_score": float(adoption_score)
                    })

            except Exception:
                continue

    if not ai_data:
        print("❌ No valid adoption scores found in AI file.")
        return

    # If multiple years exist, keep most recent per CIK
    df_ai = pd.DataFrame(ai_data)
    df_ai = df_ai.sort_values("cik").drop_duplicates(subset="cik", keep="last")

    # 2️⃣ Load Truth Data
    if not os.path.exists(TRUTH_FILE):
        print(f"❌ Error: {TRUTH_FILE} not found.")
        return

    df_truth = pd.read_csv(TRUTH_FILE)

    # Ensure numeric CIK
    df_truth["cik"] = df_truth["cik"].astype(int)

    # 3️⃣ Merge
    merged = pd.merge(df_ai, df_truth, on="cik", how="inner")

    if merged.empty:
        print("❌ No overlapping CIKs between AI and Truth data.")
        return

    # Save Excel report
    merged.to_excel(REPORT_FILE, index=False)
    print(f"✅ Excel Report saved: {REPORT_FILE} (N={len(merged)})")

    # 4️⃣ Correlations
    if len(merged) > 2:
        p_corr, p_val = pearsonr(
            merged["ai_adoption_score"],
            merged["truth_score"]
        )

        s_corr, s_val = spearmanr(
            merged["ai_adoption_score"],
            merged["truth_score"]
        )

        output_str = (
            f"AI ADOPTION VALIDATION RESULTS\n"
            f"--------------------------------\n"
            f"Sample Size: {len(merged)}\n"
            f"Pearson Correlation (Linear):  {p_corr:.4f} (p={p_val:.4f})\n"
            f"Spearman Correlation (Rank):   {s_corr:.4f} (p={s_val:.4f})\n"
            f"--------------------------------\n"
        )

        print("\n" + output_str)

        with open(STATS_FILE, "w") as f:
            f.write(output_str)

        print(f"📄 Stats saved to: {STATS_FILE}")

        

    else:
        print("⚠️ Not enough data for correlation.")

    coef = np.polyfit(
    merged["ai_adoption_score"],
    merged["truth_score"],
    1
    )

    print("📈 Polynomial Fit Parameters:")
    print(f"   Coefficient (Slope): {coef[0]:.4f}")
    print(f"   Y-Intercept:         {coef[1]:.4f}")
    print(f"   Equation: y = {coef[0]:.4f}x + {coef[1]:.4f}\n")
    
    # Predicted truth
    merged["predicted_truth"] = (
        coef[0] * merged["ai_adoption_score"] + coef[1]
    )

    # Residuals
    merged["residual"] = (
        merged["truth_score"] - merged["predicted_truth"]
    )

    # Plot residuals by CIK
    plt.figure(figsize=(10,6))
    plt.bar(merged["cik"].astype(str), merged["residual"])
    plt.xticks(rotation=90)
    plt.axhline(0, linestyle='--')
    plt.title("Residuals (Truth − Predicted)")
    plt.ylabel("Residual")
    plt.xlabel("CIK")
    plt.tight_layout()
    plt.savefig("residuals.png", bbox_inches='tight', dpi=300)
    plt.show()
    


def validate_a():
    print("📊 Processing Batch A (Intensity)...")
    
    # 1. Load AI Results
    if not os.path.exists(AI_FILE):
        print(f"❌ Error: {AI_FILE} not found. Run 3a first.")
        return

    ai_data = []
    with open(AI_FILE, 'r') as f:
        for line in f:
            try:
                rec = json.loads(line)
                # Extract CIK from filename (e.g., "12345_2018.txt" -> 12345)
                cik = int(rec['filename'].split('_')[0])
                
                # Robust extraction of "ai_score": 5
                match = re.search(r'"ai_score"\s*:\s*(\d)', rec['response'])
                if match:
                    score = int(match.group(1))
                    ai_data.append({'cik': cik, 'ai_intensity_score': score, 'reasoning': rec['response']})
            except: pass

    if not ai_data:
        print("❌ No valid scores found in JSONL file.")
        return

    df_ai = pd.DataFrame(ai_data).drop_duplicates(subset='cik', keep='last')

    # 2. Merge with Truth
    if not os.path.exists(TRUTH_FILE):
        print(f"❌ Error: {TRUTH_FILE} not found. Run Step 1 first.")
        return

    df_truth = pd.read_csv(TRUTH_FILE)
    merged = pd.merge(df_ai, df_truth, on='cik', how='inner')

    # 3. Save Excel Report
    merged.to_excel(REPORT_FILE, index=False)
    print(f"✅ Excel Report saved: {REPORT_FILE} (N={len(merged)})")

    # 4. Calculate Correlations
    if len(merged) > 2:
        p_corr, p_val = pearsonr(merged['ai_intensity_score'], merged['truth_score'])
        s_corr, s_val = spearmanr(merged['ai_intensity_score'], merged['truth_score'])
        
        output_str = (
            f"BATCH A VALIDATION RESULTS\n"
            f"--------------------------\n"
            f"Sample Size: {len(merged)}\n"
            f"Pearson Correlation (Linear):  {p_corr:.4f} (p={p_val:.4f})\n"
            f"Spearman Correlation (Rank):   {s_corr:.4f} (p={s_val:.4f})\n"
            f"--------------------------\n"
            f"Target Baseline: ~0.586\n"
        )
        
        print("\n" + output_str)
        
        # Save Stats to File
        with open(STATS_FILE, "w") as f:
            f.write(output_str)
        print(f"📄 Stats saved to: {STATS_FILE}")

    else:
        print("⚠️ Not enough data for correlation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A script that requires a file name of AI generated intensity results file.')
    # 'aifile' is defined as a mandatory positional argument
    parser.add_argument('aifile', type=str, help='The name of the AI generated intensity results file.')

    args = parser.parse_args()

    print(f"Validating AI results file: {args.aifile}")
    validate_adoption(args.aifile)