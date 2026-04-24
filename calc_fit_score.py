import json

filename = "llm_outputs/compact_with_means.jsonl"

with open(filename, "r", encoding="utf-8") as f, open("ai_score_compustat2.jsonl", "w") as f_out:
    for line in f:
        # Strip whitespace and skip empty lines
        line = line.strip()
        if not line:
            continue
            
        # Parse the JSON string into a Python dictionary
        record = json.loads(line)
        
        # Now you can access your data
        cik = record.get("cik")
        year = record.get("year")
        t = float(record.get("timeline.current.mean"))
        d = float(record.get("direction.positive.mean"))
        a = float(record.get("aggressiveness.active.mean"))

        c = [-0.02802295, 0.03156451, 0.05819324]
        intercept = 0.009016679734304109
        adopt = (c[0] * d) + (c[1] * t) + (c[2] * a) + intercept
        new_record = {"cik": cik, "year": year, "t": t, "d": d, "a": a, "ai_adoption": adopt}

        f_out.write(json.dumps(new_record) + "\n")
        