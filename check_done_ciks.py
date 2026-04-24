import pandas as pd

# 1. Load the CIKs from the CSV
# We load 'cik' as a string to preserve any leading zeros
df = pd.read_csv('compustat_data2.csv', dtype={'cik': str})
csv_ciks = set(df['(cik) CIK Number'].unique())

# 2. Load and process the .txt file
with open('done_cik_list.txt', 'r') as f:
    content = f.read()
    # Split by space and strip '.json' from each filename
    txt_filenames = content.split()
    # txt_ciks = {name.replace('.json', '') for name in txt_filenames}
    txt_ciks = {int(name.replace('.json', '')) for name in txt_filenames}
print("TXT: ")
print(txt_ciks)
print("CSV: ")
print(csv_ciks)
# 3. Find the intersection
matching_ciks = csv_ciks.intersection(txt_ciks)

# Output results
print(f"Total unique CIKs in CSV: {len(csv_ciks)}")
print(f"Total unique CIKs in TXT: {len(txt_ciks)}")
print(f"Common CIKs found: {len(matching_ciks)}")

# Optional: Save the matches to a new file
with open('matching_ciks.txt', 'w') as f:
    output_list = [f"{cik:010d}" for cik in sorted(matching_ciks)]
    f.write('\n'.join(output_list))