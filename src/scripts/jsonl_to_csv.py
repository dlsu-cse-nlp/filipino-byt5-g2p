#!/usr/bin/env python3
"""
Convert a JSONL file with nested word entries into a flat CSV.

Input JSONL structure (per line):
{
    "index": 1462,
    "success": true,
    "content": {
        "word": "upaw",
        "results": [
            {
                "pronunciation": "ʔu'paw",
                "definitions": [
                    {
                        "definition": "bald",
                        "sentences": ["Ang matanda ay upaw na.", ...]
                    },
                    ...
                ]
            },
            ...
        ]
    },
    "error": null
}

Output CSV columns:
    index, word, pronunciation, definition, sentence
"""

import argparse
import csv
import json
import sys


def extract_rows(obj):
    """Extract all rows from a single JSON object."""
    index = obj.get("index")
    content = obj.get("content")
    if not content:
        return []

    word = content.get("word", "")
    results = content.get("results", [])
    rows = []

    for result in results:
        pronunciation = result.get("pronunciation", "")
        definitions = result.get("definitions", [])
        for definition_obj in definitions:
            definition = definition_obj.get("definition", "")
            sentences = definition_obj.get("sentences", [])
            if sentences:
                for sentence in sentences:
                    rows.append([index, word, pronunciation, definition, sentence])
            else:
                # Still emit a row with an empty sentence if desired
                rows.append([index, word, pronunciation, definition, ""])
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to CSV with columns: index,word,pronunciation,definition,sentence"
    )
    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("output", help="Output CSV file path")
    args = parser.parse_args()

    try:
        with (
            open(args.input, "r", encoding="utf-8") as infile,
            open(args.output, "w", encoding="utf-8", newline="") as outfile,
        ):

            writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ["index", "word", "pronunciation", "definition", "sentence"]
            )

            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Line {line_num} is not valid JSON - skipping",
                        file=sys.stderr,
                    )
                    continue

                rows = extract_rows(data)
                for row in rows:
                    writer.writerow(row)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
