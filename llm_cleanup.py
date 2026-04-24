import json
import re

# Regex to find the JSON object within the 'text' field
# It looks for the first '{' and the last '}'
JSON_EXTRACT_RE = re.compile(r'\{.*\}', re.DOTALL)

input_file = "/Users/muhammadsaqib/Downloads/msgbatch_01K1oip5kq9h3pt5B97QSsDE_results.jsonl"
output_file = "corrected2.txt"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
            
        try:
            data = json.loads(line)
            # Navigate to the text content
            content_text = data['result']['message']['content'][0]['text']
            
            # Check if it's malformed:
            # We assume malformed if it doesn't start directly with { or ```
            is_well_formed = content_text.startswith('{') or content_text.startswith('```json')
            
            if not is_well_formed:
                # 1. Extract IDs from custom_id
                # custom_id format: cik_VAL1_year_VAL2_...
                parts = data['custom_id'].split('_')
                cik = parts[1]   # Second part
                year = parts[3]  # Fourth part
                
                # 2. Extract the nested JSON from the messy text string
                match = JSON_EXTRACT_RE.search(content_text)
                if match:
                    inner_json_str = match.group(0)
                    inner_data = json.loads(inner_json_str)
                    
                    # 3. Construct the new flat dictionary
                    result = {
                        "cik": cik,
                        "year": int(year) if year.isdigit() else year
                    }
                    # Merge with the extracted JSON fields
                    result.update(inner_data)
                    
                    # 4. Write to file
                    f_out.write(json.dumps(result) + "\n")
                    
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Skipping line due to error: {e}")

print(f"Processing complete. Check {output_file} for results.")