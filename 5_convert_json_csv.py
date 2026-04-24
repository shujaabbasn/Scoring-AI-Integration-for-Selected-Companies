import json
import pandas as pd
data = []
with open("llm_outputs/output.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip(): # Ignore empty lines
            data.append(json.loads(line))

# 2. Flatten the nested JSON structures
df_flat = pd.json_normalize(data)

# 3. Clean up the Column Names (Replace dots with underscores)
# Converts 'direction.positive' to 'direction_positive'
df_flat.columns = df_flat.columns.str.replace('.', '_', regex=False)

# 4. Convert CIK to standard integers (Removes leading zeros if needed for merging)
df_flat['cik'] = df_flat['cik'].astype(int)

# 5. Save to CSV
df_flat.to_csv("flattened_ai_data.csv", index=False)

print(f"Success! Converted {len(df_flat)} rows into flattened_ai_data.csv")
print("\nFlattened Columns detected:")
print(df_flat.columns.tolist())