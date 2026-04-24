import pandas as pd
import json

# 1. Load the Compustat CSV file
# We specify dtype={'cik': str} to ensure leading zeros aren't stripped if present
compustat_df = pd.read_csv('compustat_data2.csv', dtype=str)

col_name = '(cik) CIK Number' 
raw_ciks = compustat_df[col_name].dropna().unique()

# 3. Load the company_tickers.json file
with open('company_tickers.json', 'r') as f:
    ticker_data = json.load(f)

cik_to_name = {
    str(int(v['cik_str'])): v['title'] 
    for k, v in ticker_data.items()
}

# 5. Build the result list
results = []
for cik in raw_ciks:
    try:
        normalized_cik = str(int(float(cik))) 
        name = cik_to_name.get(normalized_cik, "Name Not Found")
    except ValueError:
        name = "Invalid CIK Format"
        
    results.append({'CIK': cik, 'CompanyName': name})

# 6. Export to companylist.csv
output_df = pd.DataFrame(results)
output_df.to_csv('companylist.csv', index=False)

print(f"Successfully exported {len(output_df)} unique CIK-Company pairs to companylist.csv")