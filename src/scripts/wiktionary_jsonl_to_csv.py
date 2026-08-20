"""Converts the .jsonl produced by the Wiktionary-guided dataset into .csv"""

import csv
import json


def convert_clean_and_flatten_data(input_file, output_csv_file, output_jsonl_file):
    seen_pairs = set()
    cleaned_rows_count = 0
    total_rows = 0

    fieldnames = [
        "index",
        "word",
        "pronunciation",
        "definition",
        "original_sentence",
        "filled_sentence",
        "answers",
    ]

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_csv_file, "w", encoding="utf-8", newline="") as csv_outfile,
        open(output_jsonl_file, "w", encoding="utf-8") as jsonl_outfile,
    ):
        writer = csv.DictWriter(csv_outfile, fieldnames=fieldnames)
        writer.writeheader()

        for line_index, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_index}: {line}")
                continue

            total_rows += 1

            word = row.get("word")
            original_sentence = row.get("original_sentence")

            pair = (word, original_sentence)

            if pair not in seen_pairs:
                seen_pairs.add(pair)
                has_newline = False

                for key, value in row.items():
                    if isinstance(value, str):
                        if "\n" in value or "\r" in value:
                            has_newline = True
                            row[key] = value.replace("\n", " ").replace("\r", " ")
                    elif isinstance(value, list):
                        # Clean lists (like 'answers') if they contain strings with newlines
                        cleaned_list = []
                        for item in value:
                            if isinstance(item, str) and ("\n" in item or "\r" in item):
                                has_newline = True
                                cleaned_list.append(
                                    item.replace("\n", " ").replace("\r", " ")
                                )
                            else:
                                cleaned_list.append(item)
                        row[key] = cleaned_list

                # Log if we fixed newlines in this row
                if has_newline:
                    cleaned_rows_count += 1
                    print(
                        f"Line ~{line_index}: Found and removed newline in record for word: '{row.get('word')}'"
                    )

                jsonl_outfile.write(json.dumps(row, ensure_ascii=False) + "\n")

                # Prepare row for CSV writing (we copy it so we don't overwrite the original dict)
                csv_row = row.copy()

                if isinstance(csv_row.get("answers"), list):
                    # Using json.dumps keeps it formatted as '["ans1", "ans2"]' in the CSV string
                    csv_row["answers"] = json.dumps(
                        csv_row["answers"], ensure_ascii=False
                    )

                writer.writerow(csv_row)

    print(f"Raw lines: {total_rows}")
    print(f"Unique records: {len(seen_pairs)}")


if __name__ == "__main__":
    INPUT_FILEPATH = (
        "data/wiktionary-scrape/transcribed/homographs_gemini_corrected.jsonl"
    )
    OUTPUT_CSV_FILEPATH = "data/wiktionary-scrape/transcribed/homographs_flattened.csv"
    OUTPUT_JSONL_FILEPATH = (
        "data/wiktionary-scrape/transcribed/homographs_flattened.jsonl"
    )

    convert_clean_and_flatten_data(
        INPUT_FILEPATH, OUTPUT_CSV_FILEPATH, OUTPUT_JSONL_FILEPATH
    )
