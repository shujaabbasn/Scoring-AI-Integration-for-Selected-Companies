import json
import os
from pathlib import Path

DIR1 = "ai_mentions_regular_new"
DIR2 = "ai_mentions_compact_new"

def compare_filings(data1, data2):
    """Compare two filing dicts and return differences."""
    diffs = []
    all_years = set(data1["filings"].keys()) | set(data2["filings"].keys())

    for year in sorted(all_years):
        texts1 = {e["text"] for e in data1["filings"].get(year, []) if e.get("text")}
        texts2 = {e["text"] for e in data2["filings"].get(year, []) if e.get("text")}

        only_in_1 = texts1 - texts2
        only_in_2 = texts2 - texts1

        if only_in_1 or only_in_2:
            if (len(only_in_1) != len(only_in_2)):
                diffs.append({
                    "year": year,
                    "only_in_dir1_count": len(only_in_1),
                    "only_in_dir2_count": len(only_in_2),
                    "only_in_dir1": list(only_in_1),
                    "only_in_dir2": list(only_in_2),
                })

    return diffs

files1 = {f.name for f in Path(DIR1).glob("*.json")}
files2 = {f.name for f in Path(DIR2).glob("*.json")}

common_files = files1 & files2
only_in_dir1 = files1 - files2
only_in_dir2 = files2 - files1

if only_in_dir1:
    print(f"Files only in {DIR1}: {only_in_dir1}")
if only_in_dir2:
    print(f"Files only in {DIR2}: {only_in_dir2}")

files_with_diffs = []

for filename in sorted(common_files):
    with open(os.path.join(DIR1, filename)) as f:
        data1 = json.load(f)
    with open(os.path.join(DIR2, filename)) as f:
        data2 = json.load(f)

    # Skip if both are empty
    all_empty1 = all(len(v) == 0 for v in data1["filings"].values())
    all_empty2 = all(len(v) == 0 for v in data2["filings"].values())
    if all_empty1 and all_empty2:
        continue

    diffs = compare_filings(data1, data2)
    if diffs:
        files_with_diffs.append((filename, diffs))

print(f"\n{len(files_with_diffs)} files have differences:\n")
for filename, diffs in files_with_diffs:
    print(f"  {filename}:")
    for d in diffs:
        print(f"    Year {d['year']}: {d['only_in_dir1_count']} only in dir1, {d['only_in_dir2_count']} only in dir2")


print(f"\n{len(files_with_diffs)} files have differences:\n")
for filename, diffs in files_with_diffs:
    print(f"\n{'='*60}")
    print(f"File: {filename}")
    for d in diffs:
        print(f"\n  Year {d['year']}:")
        if d["only_in_dir1"]:
            print(f"    Only in {DIR1} ({d['only_in_dir1_count']} entries):")
            for text in d["only_in_dir1"]:
                print(f"      {text[:100]}...")
        if d["only_in_dir2"]:
            print(f"    Only in {DIR2} ({d['only_in_dir2_count']} entries):")
            for text in d["only_in_dir2"]:
                print(f"      {text[:100]}...")