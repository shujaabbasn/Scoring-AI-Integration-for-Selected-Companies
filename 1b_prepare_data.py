import pandas as pd

# Load the manual list we made
# df = pd.read_excel("Final_List_With_CIKs.xlsx")
df1 = pd.read_csv("bg_filtered.csv")
df2 = pd.read_csv("gvkey_cik_link.csv")

merged = df1.merge(df2[["gvkey", "cik"]], on="gvkey", how="left")
result = merged[["gvkey", "year", "cik", "ai_score"]]
result.to_csv("truth_data.csv", index=False)

missing = result[result["cik"].isna()]["gvkey"].unique()
if len(missing) > 0:
    print(f"Warning: {len(missing)} GVKEYs from df1 not found in df2:")
    for gvkey in missing:
        print(f"  {gvkey}")
else:
    print("All GVKEYs matched successfully.")

duplicates = df2[df2["gvkey"].isin(df1["gvkey"])].groupby("gvkey")["cik"].apply(list)
duplicates = duplicates[duplicates.apply(len) > 1]

if len(duplicates) > 0:
    print(f"{len(duplicates)} GVKEYs have multiple CIKs:")
    for gvkey, ciks in duplicates.items():
        print(f"  {gvkey}: {ciks}")
else:
    print("All GVKEYs have a unique CIK.")
# print(f"Done. Processed {len(df)} companies.")