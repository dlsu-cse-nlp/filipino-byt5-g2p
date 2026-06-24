import json


def process_dictionary_jsonl(input_file, output_file):
    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):

        for line in infile:
            if not line.strip():
                continue

            entry = json.loads(line)
            matches = entry.get("matches", [])

            # Merge matches where "definitions" are the same
            def_map = {}
            for m in matches:
                def_key = tuple((m.get("definitions", [])))
                if def_key not in def_map:
                    def_map[def_key] = set()
                def_map[def_key].update(m.get("prons", []))

            step1_matches = [
                {"prons": (list(prons)), "definitions": list(def_key)}
                for def_key, prons in def_map.items()
            ]

            # Merge matches where "prons" are the same
            pron_map = {}
            for m in step1_matches:
                pron_key = tuple((m["prons"]))
                if pron_key not in pron_map:
                    pron_map[pron_key] = set()
                pron_map[pron_key].update(m["definitions"])

            final_matches = [
                {"prons": list(pron_key), "definitions": (list(defs))}
                for pron_key, defs in pron_map.items()
            ]

            # Remove entries with 1 or 0 matches
            if len(final_matches) > 1:
                entry["matches"] = final_matches
                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    INPUT_PATH = "data/wiktionary-scrape/ambiguous_prons_all.jsonl"
    OUTPUT_PATH = "data/wiktionary-scrape/ambiguous_prons_cleaned.jsonl"

    process_dictionary_jsonl(INPUT_PATH, OUTPUT_PATH)

    print(f"Results saved to {OUTPUT_PATH}")
