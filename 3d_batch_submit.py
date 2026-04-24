import os
import json
import hashlib
from datetime import datetime, UTC
import argparse
import anthropic
from tqdm import tqdm # Import the progress bar

DEFAULT_BASE_DIR = "ai_extracted_compact"
MODEL = "claude-sonnet-4-6" 

# Grab API key from environment
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=API_KEY, default_headers={"anthropic-version": "2023-06-01"})

# (Insert your PROMPT string here as you have it written)
PROMPT = """### SYSTEM PROMPT

You are an expert financial analyst. Your task is to analyze corporate disclosure text from SEC 10-K filings to measure the realized economic integration of Artificial Intelligence (AI) technologies within the firm.

### DEFINITIONS
## Direction of Impact (Sentiment & Strategy)
This category measures the net effect of AI on the firm's value proposition.

**Positive**: The text describes AI as a value driver. Look for keywords like alpha generation, opex reduction, yield optimization, or "moat" expansion. Example: "Our AI-driven churn model reduced customer attrition by 15%."

**Negative**: The text identifies AI as a headwind or threat. This includes increased capital expenditure (CapEx) without immediate return, predatory competition from AI-native startups, or significant regulatory/compliance costs.

**Other**: Purely descriptive or "Safe Harbor" language. Statements like "We are subject to evolving AI regulations" should be categorized here unless a specific impact is cited.

## Significance Score (Economic Materiality)
This is a weighted measure of how central AI is to the company's "Bottom Line."

**Low (0.0 - 1.0)**: Experimental or peripheral. AI is mentioned as a "pilot," a "feature" of a third-party tool, or in a general list of industry trends.

**Moderate (1.1 - 2.0)**: AI is integrated into a core business process (e.g., a logistics firm using AI for route optimization). It is mentioned in the "Management’s Discussion and Analysis" (MD&A) section as a factor in results.

**High (2.1 - 3.0)**: The firm's identity or primary competitive advantage is shifting toward AI. This includes significant R&D shifts, M&A of AI companies, or AI being cited as the primary reason for revenue growth or margin expansion.

## Primary Economic Topic
Determine where the "AI dollar" is actually landing.

**Labor**: Replacement of human tasks, "copilot" augmentation, or changes in headcount/hiring strategy due to automation.

**Investment**: Significant CapEx or R&D spending directed toward compute, data centers, or model training.

**Revenue**: New product lines, AI-as-a-Service (AIaaS), or increased pricing power enabled by AI features.

**Competition**: Defensive mentions regarding the AI capabilities of peers or the risk of obsolescence.

**M&A**: Acquiring talent or intellectual property (IP) specifically to bolster the AI roadmap.

**Other**: Cybersecurity, ESG reporting, or general administrative overhead.

## Timeline (Realization Phase)
Distinguish between "Press Release AI" and "Balance Sheet AI."

**Happened**: The technology is fully deployed. Results (revenue or savings) are already appearing in the current financial statements.

**Current**: The firm is in the "deployment" or "migration" phase. Costs are being incurred now, but full utility is expected in the next 1-2 fiscal years.

**Planning**: Mentions of "intent," "exploration," or "evaluation." High uncertainty regarding if or when the tech will reach production.

**Other**: Historical context or industry-wide generalizations.

## Aggressiveness (Proprietary vs. Commodity)
**Active**: The firm is a Price Maker in AI. They are building proprietary models, fine-tuning open-source LLMs on private datasets, or hiring specialized ML engineers to build custom infrastructure.

**Passive**: The firm is a Price Taker. They are using "Off-the-shelf" solutions (e.g., Salesforce Einstein, Microsoft 365 Copilot, Adobe Firefly). Their AI advantage is tied to their software vendors rather than internal innovation.

## AI Relevance Score
A number between 0 (not relevant at all) and 1 (definitely relevant) indicating how relevant the text is to AI.

## Overall Confidence
A number between 0 and 1 describing how confident you are overall on your assessment of all the above metrics.

### OUTPUT FORMAT
- Return **ONLY** valid JSON. 
- Do **NOT** include any conversational filler, explanations, or markdown code blocks (like ```json).
- If the text is not about AI, set `ai_relevance_score` near 0 and `significance_score` to 0.

### FEW-SHOT EXAMPLES

#### Example 1: Active, High Significance, Current (Positive)
Text: "This year, we transitioned our primary underwriting engine to a proprietary deep learning architecture. By training on our internal dataset of 10 million historical loans, the AI now automates 85% of credit approvals with a 12% reduction in default rates."
JSON:
{
  "direction": {"positive": 0.95, "negative": 0.05, "other": 0.0},
  "significance_score": 2.8,
  "topics": {"labor": 0.2, "investment": 0.4, "revenue": 0.4, "competition": 0.0, "M&A": 0.0, "other": 0.0},
  "timeline": {"happened": 0.3, "current": 0.6, "planning": 0.1, "other": 0.0},
  "aggressiveness": {"active": 0.9, "passive": 0.1, "other": 0.0},
  "ai_relevance_score": 1.0,
  "overall_confidence": 0.95
}

#### Example 2: Passive, Moderate Significance, Happened (Risk/Negative)
Text: "Our reliance on third-party cloud-based AI productivity tools has introduced new data privacy risks. In Q3, we experienced a localized data leak when an employee uploaded proprietary source code into an external Large Language Model."
JSON:
{
  "direction": {"positive": 0.1, "negative": 0.8, "other": 0.1},
  "significance_score": 1.4,
  "topics": {"labor": 0.0, "investment": 0.0, "revenue": 0.0, "competition": 0.0, "M&A": 0.0, "other": 1.0},
  "timeline": {"happened": 0.8, "current": 0.2, "planning": 0.0, "other": 0.0},
  "aggressiveness": {"active": 0.1, "passive": 0.8, "other": 0.1},
  "ai_relevance_score": 0.85,
  "overall_confidence": 0.88
}

#### Example 3: Low Significance/Relevance (Neutral)
Text: "While we track industry trends in automation and smarter data processing, our current focus remains on physical retail expansion and traditional supply chain logistics."
JSON:
{
  "direction": {"positive": 0.0, "negative": 0.0, "other": 1.0},
  "significance_score": 0.1,
  "topics": {"labor": 0.0, "investment": 0.0, "revenue": 0.0, "competition": 0.0, "M&A": 0.0, "other": 1.0},
  "timeline": {"happened": 0.0, "current": 0.0, "planning": 0.2, "other": 0.8},
  "aggressiveness": {"active": 0.0, "passive": 0.2, "other": 0.8},
  "ai_relevance_score": 0.1,
  "overall_confidence": 0.9
}

#### Example 4: Active, Moderate Significance, Planning (M&A)
Text: "To accelerate our autonomous navigation roadmap, we signed a definitive agreement to acquire SignalBot, a leader in edge-computing AI. This transaction, closing in Q4, will integrate their proprietary computer vision stack into our fleet, reducing reliance on external software vendors."
JSON:
{
  "direction": {"positive": 0.85, "negative": 0.05, "other": 0.1},
  "significance_score": 1.9,
  "topics": {"labor": 0.0, "investment": 0.3, "revenue": 0.0, "competition": 0.0, "M&A": 0.7, "other": 0.0},
  "timeline": {"happened": 0.1, "current": 0.2, "planning": 0.7, "other": 0.0},
  "aggressiveness": {"active": 0.8, "passive": 0.2, "other": 0.0},
  "ai_relevance_score": 0.95,
  "overall_confidence": 0.92
}

### FINAL INSTRUCTION
Analyze the following text from a corporate 10-K filing and provide the JSON analysis.
"""

def get_year_entries(cik, filings, year):
    """Extracts paragraphs for a specific year and formats them for the Batch API."""
    batch_requests = []
    
    # Ensure we handle years as strings since JSON keys are strings
    year_str = str(year)
    entries = filings.get(year_str, [])
    
    for idx, entry in enumerate(entries, start=1):
        text = entry.get("text", "")
        if not text.strip():
            continue
            
        para_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        
        # FIX: Dynamically include the 'year' in the custom_id
        custom_id = f"cik_{cik}_year_{year_str}_p_{idx}_{para_hash[:8]}"

        batch_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": 1000,
                "system": [
                    {
                        "type": "text",
                        "text": PROMPT,
                        "cache_control": {"type": "ephemeral"} # This line enables caching
                    }
                ],
                "messages": [{"role": "user", "content": text}],
                "temperature": 0.2
            }
        })
    return batch_requests

def submit_multi_year_batch(start_year, end_year, base_dir, total_work, is_dry_run=False, chunk_size = 5000):
    """Processes all files in BASE_DIR for a range of years (inclusive)."""
    LOG_FILE = "active_batches.txt"
    ZEROS_FILE = "zeros.jsonl" 
    if not os.path.isdir(base_dir):
        print(f"Directory not found: {base_dir}")
        return
    
    all_requests = []
    DRY_RUN_FILE = "dry_run_payload.jsonl"

    if is_dry_run and os.path.exists(DRY_RUN_FILE):
        os.remove(DRY_RUN_FILE)

    # Create the range of years
    years_to_process = [str(y) for y in range(start_year, end_year + 1)]
    print(f"Targeting years: {', '.join(years_to_process)}")

    # Helper to log the batch details locally
    def log_batch(batch_id, count):
        timestamp = datetime.now(UTC).isoformat()
        with open(LOG_FILE, "a") as f:
            f.write(f"{timestamp} | ID: {batch_id} | Requests: {count} | Years: {start_year}-{end_year}\n")

    zeros_f = open(ZEROS_FILE, "w")
    # Initialize the progress bar
    pbar = tqdm(total=total_work, desc="Processing Filings", unit="item")
    for fname in sorted(os.listdir(base_dir)):
        if not fname.endswith(".json"):
            continue
            
        full_path = os.path.join(base_dir, fname)
        try:
            with open(full_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read {fname}: {e}")
            continue

        cik = data.get("cik", "<unknown>")
        filings = data.get("filings", {})

        # Loop through each year for this specific company
        for year in years_to_process:
            year_payloads = get_year_entries(cik, filings, year)
            if year in filings and len(year_payloads) == 0:
                zero_record = {
                    "timestamp": "dummy",
                    "cik": cik,
                    "year": int(year),
                    "adoption_weight_sum": 0.0,
                    "raw_adoption_weight_sum": 0.0,
                    "direction": {},
                    "topics": {},
                    "timeline": {},
                    "aggressiveness": {}
                }
                zeros_f.write(json.dumps(zero_record) + "\n")
            all_requests.extend(year_payloads)

            # Update the bar by 1 for every company-year pair completed
            pbar.update(1)

            # CHUNKING LOGIC: Submit whenever we hit the limit
            while len(all_requests) >= chunk_size:
                to_submit = all_requests[:chunk_size]
                all_requests = all_requests[chunk_size:] # Keep the remainder
                
                print(f"🚀 Submitting chunk of {len(to_submit)} requests...")
                try:
                    if is_dry_run:
                        with open(DRY_RUN_FILE, "a") as df:
                            for req in to_submit:
                                df.write(json.dumps(req, separators=(',', ':')) + "\n")
                        print(f"📝 Dry Run: Wrote {chunk_size} requests to {DRY_RUN_FILE}")
                    else:
                        res = client.messages.batches.create(requests=to_submit)
                        print(f"✅ Batch ID: {res.id}")
                        log_batch(res.id, len(to_submit))
                except Exception as e:
                    print(f"Submission failed: {e}")

    # Final submission for any leftover requests < chunk_size
    if all_requests:
        print(f"🚀 Submitting final chunk of {len(all_requests)}...")
        try:
            if is_dry_run:
                with open(DRY_RUN_FILE, "a") as df:
                    for req in all_requests:
                        df.write(json.dumps(req) + "\n")
                print(f"📝 Dry Run: Wrote 5000 requests to {DRY_RUN_FILE}")
            else:
                res = client.messages.batches.create(requests=all_requests)
                print(f"✅ Batch ID: {res.id}")
                log_batch(res.id, len(all_requests))
        except Exception as e:
            print(f"Submission failed: {e}")

    zeros_f.close()
    print(f"📝 Zero records written to {ZEROS_FILE}")
    pbar.close()
    if is_dry_run:
        print(f"\n✨ Dry run complete. Inspect '{DRY_RUN_FILE}' to verify your prompts.")
    else:
        print(f"\n✨ All batches submitted. Check '{LOG_FILE}' for the list of IDs.")
    

if __name__ == "__main__":
    # 1. Command line arguments specification
    parser = argparse.ArgumentParser(description="Submit SEC 10-K AI analysis batches to Anthropic.")
    
    # 2. Add the base_dir argument with a default
    parser.add_argument(
        "--dir", 
        type=str, 
        default=DEFAULT_BASE_DIR,
        help=f"Directory containing input JSON files (default: {DEFAULT_BASE_DIR})"
    )
    
    # 3. Add start/end year arguments for the multi-year logic
    parser.add_argument("--start", type=int, default=2018, help="Start year (default: 2018)")
    parser.add_argument("--end", type=int, default=2018, help="End year (default: 2018)")

    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Generate the JSONL file locally without calling the Anthropic API."
    )

    # 4. Parse the arguments
    args = parser.parse_args()
    # 5. Assign the parsed directory to a variable to pass into your functions
    target_dir = args.dir
    
    # Verify the directory exists before starting
    if not os.path.isdir(target_dir):
        print(f"❌ Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    print(f"📂 Using input directory: {target_dir}")

    # --- PROGRESS BAR MATH ---
    # 1. Count only the .json files in the directory
    files_to_process = [f for f in os.listdir(args.dir) if f.endswith(".json")]
    num_files = len(files_to_process)
    
    # 2. Calculate number of years in the range
    num_years = (args.end - args.start) + 1
    
    # 3. Total iterations = Companies * Years
    total_iterations = num_files * num_years

    print(f"📂 Directory: {args.dir}")
    print(f"🗓️  Years: {args.start} to {args.end} ({num_years} year(s))")
    print(f"📊 Total Company-Year pairs to scan: {total_iterations}")
    print("-" * 50)    
    # 6. Call your submission function (ensure you update it to accept the dir as a parameter)
    submit_multi_year_batch(
        start_year=args.start, 
        end_year=args.end, 
        base_dir=target_dir,
        total_work=total_iterations,
        is_dry_run=args.dry_run,
        chunk_size = 5000
    )