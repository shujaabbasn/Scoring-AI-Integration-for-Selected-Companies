import os
import json
import re
import math
from collections import defaultdict
from datetime import datetime, UTC
import anthropic

# Configuration
LOG_FILE = "active_batches.txt"
OUTPUT_FILE = "adoption_output"
OUT_EXT=".jsonl"
OUTPUT_DIR = "llm_outputs"

client = anthropic.Anthropic()

def get_next_filename(base_filename, extension):
    """
    Generates the next available filename with an incrementing number suffix.

    Args:
        base_filename (str): The desired base name of the file (e.g., 'sample_data').
        extension (str): The file extension (e.g., '.txt', '.zip').

    Returns:
        str: The unique, incremented filename.
    """
    number = 1
    # Loop indefinitely until a non-existent filename is found
    while True:
        # Format the filename with the current number
        new_filename = f"{base_filename}_{number:03d}{extension}"
        
        # Check if the generated filename already exists
        if not os.path.exists(new_filename):
            # If it doesn't exist, this is the unique filename to use
            return new_filename
        
        # If it exists, increment the number and try again
        number += 1

def normalize(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()} if total > 0 else {k: 0.0 for k in d}

def extract_json_from_text(text):
    """
    Finds the first '{' and last '}' to extract a JSON string from a potentially
    noisy LLM response. Returns a Python dict or None if invalid.
    """
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx == -1 or end_idx == -1:
            return None
        
        json_str = text[start_idx:end_idx + 1]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

def aggregate_company_data(parsed_list):
    """Performs the weighted scoring and normalization for a single CIK/Year pair."""
    agg = {
        "raw_weight_sum": 0.0,
        "direction": defaultdict(float),
        "topics": defaultdict(float),
        "timeline": defaultdict(float),
        "aggressiveness": defaultdict(float),
        "direction.positive.mean": defaultdict(float),
        "timeline.current.mean": defaultdict(float),
        "aggressiveness.active.mean": defaultdict(float)
    }

    valid_paragraphs = 0
    dpm = 0
    tcm = 0
    aam = 0
    for parsed in parsed_list:
        relevance = parsed.get("ai_relevance_score", 0.0)
        if relevance < 0.05:
            continue

        valid_paragraphs += 1
        
        # Logic: Current or Happened probability * Active development
        timeline_prob = parsed["timeline"].get("current", 0.0) + parsed["timeline"].get("happened", 0.0)
        active_prob = parsed["aggressiveness"].get("active", 0.0)
        
        # Calculate weight for this specific paragraph
        weight = parsed.get("significance_score", 1.0) * timeline_prob * active_prob * relevance

        if weight <= 0:
            continue

        agg["raw_weight_sum"] += weight

        # Distribute weighted probabilities
        for d, v in parsed.get("direction", {}).items(): agg["direction"][d] += v * weight
        for t, v in parsed.get("topics", {}).items(): agg["topics"][t] += v * weight
        for t, v in parsed.get("timeline", {}).items(): agg["timeline"][t] += v * weight
        for a, v in parsed.get("aggressiveness", {}).items(): agg["aggressiveness"][a] += v * weight

        dp_val = parsed.get("direction", {}).get("positive", 0.0)
        dpm += dp_val

        tc_val = parsed.get("timeline", {}).get("current", 0.0)
        tcm += tc_val

        aa_val = parsed.get("aggressiveness", {}).get("active", 0.0)
        aam += aa_val


    # Final Normalization
    adoption_score = math.log1p(agg["raw_weight_sum"]) / valid_paragraphs if valid_paragraphs > 0 else 0.0

    aam = aam / valid_paragraphs if valid_paragraphs > 0 else 0.0
    tcm = tcm / valid_paragraphs if valid_paragraphs > 0 else 0.0
    dpm = dpm / valid_paragraphs if valid_paragraphs > 0 else 0.0
    
    return {
        "adoption_weight_sum": adoption_score,
        "raw_adoption_weight_sum": agg["raw_weight_sum"],
        "direction": normalize(agg["direction"]),
        "topics": normalize(agg["topics"]),
        "timeline": normalize(agg["timeline"]),
        "aggressiveness": normalize(agg["aggressiveness"]),
        "direction.positive.mean": dpm,
        "timeline.current.mean": tcm,
        "aggressiveness.active.mean": aam 
    }

def write_results_to_file(all_new_data):
    """Handles directory creation and unique filename generation for the output."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    base_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    unique_filename = get_next_filename(base_path, OUT_EXT)
    
    print(f"📊 Aggregating data for {len(all_new_data)} company-year pairs...")
    
    with open(unique_filename, "w") as outf:
        for (cik, year), parsed_list in all_new_data.items():
            final_stats = aggregate_company_data(parsed_list)
            
            # Combine stats with metadata
            final_record = {
                "timestamp": datetime.now(UTC).isoformat() + "Z",
                "cik": cik,
                "year": int(year),
                **final_stats
            }
            outf.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            
    print(f"✨ Successfully wrote results to: {unique_filename}")

def process_batch_results(batch_id):
    """Downloads and groups results from a specific Batch ID with error handling."""
    company_data = defaultdict(list)
    
    try:
        results_stream = client.messages.batches.results(batch_id)
    except Exception as e:
        print(f"⚠️ Could not stream results for {batch_id}: {e}")
        return company_data

    for entry in results_stream:
        if entry.result.type != "succeeded":
            print(f"❌ Request {entry.custom_id} failed in batch.")
            continue
        
        parts = entry.custom_id.split("_")
        cik = parts[1]
        year = parts[3]
        
        # Initialize parsed_llm to None at the start of every loop iteration
        parsed_llm = None
        
        # Get the actual text content from the LLM
        raw_text = entry.result.message.content[0].text
        
        # Clean up standard markdown if present
        cleaned_text = re.sub(r"^```json\s*|^```\s*|```$", "", raw_text.strip(), flags=re.MULTILINE)

        with open("output.txt", "a") as f:
            try:
                # Attempt 1: Direct JSON parse
                parsed_llm = json.loads(cleaned_text)
                record = {"cik": cik, "year": year, **parsed_llm}
                f.write(json.dumps(record) + "\n")
            except Exception:
                # Attempt 2: Rescue logic using your helper function
                # We pass the raw_text (or cleaned_text) to find the { } block
                parsed_llm = extract_json_from_text(raw_text)
                
                if parsed_llm:
                    record = {"cik": cik, "year": year, **parsed_llm}
                    f.write(json.dumps(record) + "\n")
                else:
                    print(f"❌ Failed to extract JSON for CIK {cik}")

        # Now parsed_llm is guaranteed to be defined (either as a dict or None)
        if parsed_llm and "significance_score" in parsed_llm:
            parsed_llm['_metadata'] = {'cik': cik, 'year': year}
            company_data[(cik, year)].append(parsed_llm)
        else:
            print(f"⚠️ Skipping malformed or empty JSON for CIK {cik} in {year}")
            
    return company_data

def run_retrieval():
    """Main orchestration logic for checking logs and processing batches."""
    if not os.path.exists(LOG_FILE):
        print(f"No log file found at {LOG_FILE}. Nothing to retrieve.")
        return

    with open(LOG_FILE, "r") as f:
        lines = f.readlines()

    remaining_batches = []
    all_new_data = defaultdict(list)

    print(f"🧐 Checking {len(lines)} batches from log...")

    for line in lines:
        if not line.strip() or "ID: " not in line:
            continue
        
        batch_id = line.split("ID: ")[1].split(" |")[0].strip()
        
        try:
            batch = client.messages.batches.retrieve(batch_id)
            
            if batch.processing_status == "ended":
                print(f"✅ Batch {batch_id} complete. Processing...")
                batch_results = process_batch_results(batch_id)
                for key, val in batch_results.items():
                    all_new_data[key].extend(val)
            else:
                print(f"⌛ Batch {batch_id} status: {batch.processing_status}. Keeping in log.")
                remaining_batches.append(line)
        except Exception as e:
            print(f"❌ Error checking batch {batch_id}: {e}")
            remaining_batches.append(line)

    # 1. Update the log file
    with open(LOG_FILE, "w") as f:
        f.writelines(remaining_batches)

    # 2. Process and Save
    if all_new_data:
        write_results_to_file(all_new_data)
    else:
        print("No new completed data to aggregate at this time.")

if __name__ == "__main__":
    run_retrieval()

# def run_retrieval():
#     if not os.path.exists(LOG_FILE):
#         print(f"No log file found at {LOG_FILE}. Nothing to retrieve.")
#         return

#     with open(LOG_FILE, "r") as f:
#         lines = f.readlines()

#     remaining_batches = []
#     all_new_data = defaultdict(list)

#     print(f"🧐 Checking {len(lines)} batches from log...")

#     for line in lines:
#         if not line.strip() or "ID: " not in line:
#             continue
        
#         # Extract ID from the log line format: "timestamp | ID: msgbatch_xyz | ..."
#         batch_id = line.split("ID: ")[1].split(" |")[0].strip()
        
#         try:
#             batch = client.messages.batches.retrieve(batch_id)
            
#             if batch.processing_status == "ended":
#                 print(f"✅ Batch {batch_id} complete. Processing...")
#                 batch_results = process_batch_results(batch_id)
                
#                 # Merge into our main collection
#                 for key, val in batch_results.items():
#                     all_new_data[key].extend(val)
#             else:
#                 # Still processing or expired/cancelled
#                 print(f"⌛ Batch {batch_id} status: {batch.processing_status}. Keeping in log.")
#                 remaining_batches.append(line)
                
#         except Exception as e:
#             print(f"❌ Error checking batch {batch_id}: {e}")
#             remaining_batches.append(line)

#     # 1. Update the log file (Remove successful, keep failed/pending)
#     with open(LOG_FILE, "w") as f:
#         f.writelines(remaining_batches)

#     # 2. Aggregate and write the data for the completed batches
#     if all_new_data:
#         print(f"📊 Aggregating data for {len(all_new_data)} company-year pairs...")
#         with open(OUTPUT_FILE, "a") as outf:
#             for (cik, year), parsed_list in all_new_data.items():
#                 agg = {
#                     "raw_weight_sum": 0.0,
#                     "direction": defaultdict(float),
#                     "topics": defaultdict(float),
#                     "timeline": defaultdict(float),
#                     "aggressiveness": defaultdict(float)
#                 }

#                 valid_paragraphs = 0

#                 for parsed in parsed_list:
#                     relevance = parsed.get("ai_relevance_score", 0.0)
#                     if relevance < 0.05:
#                         continue

#                     valid_paragraphs += 1
                    
#                     # Logic: Current or Happened probability * Active development
#                     timeline_prob = parsed["timeline"].get("current", 0.0) + parsed["timeline"].get("happened", 0.0)
#                     active_prob = parsed["aggressiveness"].get("active", 0.0)
                    
#                     # Calculate weight for this specific paragraph
#                     weight = parsed.get("significance_score", 1.0) * timeline_prob * active_prob * relevance

#                     if weight <= 0:
#                         continue

#                     agg["raw_weight_sum"] += weight

#                     # Distribute weighted probabilities
#                     for d, v in parsed.get("direction", {}).items(): agg["direction"][d] += v * weight
#                     for t, v in parsed.get("topics", {}).items(): agg["topics"][t] += v * weight
#                     for t, v in parsed.get("timeline", {}).items(): agg["timeline"][t] += v * weight
#                     for a, v in parsed.get("aggressiveness", {}).items(): agg["aggressiveness"][a] += v * weight

#                 # Final Normalization
#                 adoption_score = math.log1p(agg["raw_weight_sum"]) / valid_paragraphs if valid_paragraphs > 0 else 0.0

#                 final_record = {
#                     "timestamp": datetime.now(UTC).isoformat() + "Z",
#                     "cik": cik,
#                     "year": int(year),
#                     "adoption_weight_sum": adoption_score,
#                     "raw_adoption_weight_sum": agg["raw_weight_sum"],
#                     "direction": normalize(agg["direction"]),
#                     "topics": normalize(agg["topics"]),
#                     "timeline": normalize(agg["timeline"]),
#                     "aggressiveness": normalize(agg["aggressiveness"])
#                 }

#                 outf.write(json.dumps(final_record, ensure_ascii=False) + "\n")
#         print(f"✨ Successfully updated {OUTPUT_FILE}.")
#     else:
#         print("No new completed data to aggregate at this time.")

# if __name__ == "__main__":
#     run_retrieval()