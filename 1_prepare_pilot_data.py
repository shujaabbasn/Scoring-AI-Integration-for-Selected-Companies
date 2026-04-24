import pandas as pd

# Load the manual list we made
# df = pd.read_excel("Final_List_With_CIKs.xlsx")
df = pd.read_csv("company_ciks.csv")

# Drop rows without CIKs and make them integers
df = df.dropna(subset=['cik'])
df['cik'] = df['cik'].astype(int)

# Rename columns (sharenarrowai2018 score is being considered the truth score)
df = df.rename(columns={
    'comnam': 'company_name',
    'ai_score': 'truth_score'
})

# Save clean version
df.to_csv("Pilot_Truth_Data.csv", index=False)
print(f"Done. Processed {len(df)} companies.")