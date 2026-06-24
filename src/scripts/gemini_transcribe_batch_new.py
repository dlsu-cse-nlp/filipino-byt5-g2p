"""
This script batch processes fun.csv to fill in pronunciation choices
for homographs using the Gemini API via Vertex AI (async + concurrent).
Adds exponential backoff retry for 429 errors and resume capability.
"""

# INFO: Updated as of 22 June 2026
# Uses the new Agent Platform

# TODO: Do an audit of this and other similar scripts

import asyncio
import collections
import csv
import json
from typing import List, Set

from dotenv import dotenv_values
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from src.utils.generate_prompt.transcribe import generate_prompt
from src.utils.homographs import fill_template, homographs

# ---------- Gemini / Vertex AI setup ----------
config = dotenv_values(".env")
PROJECT_ID = "basic-formula-489010-n5"
LOCATION = "global"

MODEL_NAME = "gemini-3.5-flash"
CONCURRENCY_LIMIT = 5

INPUT_CSV = "data/wiktionary-scrape/generated/sentences_no_ipa.csv"
OUTPUT_JSONL = "final_homograph_results_gemini_2.jsonl"
DEFINITIONS_JSONL = "data/wiktionary-scrape/ambiguous_prons_cleaned.jsonl"

# ---------- Retry settings ----------
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds, doubles each attempt


class SentenceResult(BaseModel):
    """Answers for a specific sentence"""

    answers: List[str]


class Response(BaseModel):
    """JSON schema for Gemini API's structured output for grouped sentences"""

    results: List[SentenceResult]


client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

gen_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_json_schema=Response.model_json_schema(),
    thinking_config=types.ThinkingConfig(thinking_level="LOW"),
)


def load_definitions(path: str = DEFINITIONS_JSONL) -> dict:
    """Load the definition map from a JSONL file."""
    defs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            word = entry["word"]
            defs[word] = entry["matches"]
    return defs


def load_existing_sentences(output_path: str) -> Set[str]:
    """Read already processed sentences from the output JSONL."""
    sentences = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "original_sentence" in data:
                        sentences.add(data["original_sentence"])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass  # No previous output, start fresh
    return sentences


async def call_gemini_with_retry(
    prompt: str,
    model: str,
    config: types.GenerateContentConfig,
) -> types.GenerateContentResponse:
    """Call Gemini with exponential backoff on 429 errors."""
    delay = BASE_DELAY
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except ClientError as e:
            # FIX: Check e.code (integer) or e.status (string) instead of e.status == 429
            if (
                e.code == 429 or e.status == "RESOURCE_EXHAUSTED"
            ) and attempt < MAX_RETRIES:
                print(
                    f"\n[429] Rate limit hit. Retrying in {delay}s (Attempt {attempt + 1}/{MAX_RETRIES})..."
                )
                await asyncio.sleep(delay)
                delay *= 2  # exponential backoff
                continue
            raise  # re-raise other errors or if max retries exhausted


async def process_word_group(
    target_word: str,
    rows: List[dict],
    definitions_map: dict,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
) -> None:
    """Process a group of rows sharing the same target word."""
    async with semaphore:
        grouped_data = []
        output_templates = []

        # 1. Extract homograph information for each sentence in the group
        for row in rows:
            sentence = row["sentence"]
            pronunciation = row["pronunciation"]

            ambiguous_words, choices, output_template = homographs(
                sentence, target_word, pronunciation
            )
            output_templates.append(output_template)

            grouped_data.append(
                {
                    "sentence": sentence,
                    "words": ambiguous_words,
                    "pronunciations": choices,
                }
            )

        # 2. Build the consolidated prompt
        prompt = generate_prompt(
            grouped_data=grouped_data,
            definitions_map=definitions_map,
        )

        # 3. Call Gemini with retry
        response = await call_gemini_with_retry(prompt, MODEL_NAME, gen_config)

        # 4. Parse structured output
        output = Response.model_validate_json(response.text)

        if len(output.results) != len(rows):
            print(
                f"\nWarning: Mismatch! Gemini returned {len(output.results)} results for {len(rows)} sentences (Word: {target_word}). Validating up to matched lengths."
            )

        # 5. Thread‑safe write to JSONL (append)
        async with file_lock:
            with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                # Zip strictly connects the sentence, template, and the returned answers
                for row, template, sent_result in zip(
                    rows, output_templates, output.results
                ):
                    filled_sentence = fill_template(template, sent_result.answers)

                    result = {
                        "index": int(row["index"]),
                        "word": row["word"],
                        "pronunciation": row["pronunciation"],
                        "definition": row["definition"],
                        "original_sentence": row["sentence"],
                        "filled_sentence": filled_sentence,
                        "answers": sent_result.answers,
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")


async def main() -> None:
    definitions_map = load_definitions()

    # 1. Load already processed sentences to resume
    processed = load_existing_sentences("final_homograph_results_gemini.jsonl")
    print(f"Found {len(processed)} already processed sentences in {OUTPUT_JSONL}")

    # 2. Read rows from fun.csv, skipping those already done
    rows_to_process = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sentence = row["sentence"].strip()
            if sentence in processed:
                continue
            rows_to_process.append(row)

    if not rows_to_process:
        print("All rows already processed. Nothing to do.")
        return

    # 3. Group remaining rows by target word
    word_groups = collections.defaultdict(list)
    for row in rows_to_process:
        word_groups[row["word"]].append(row)

    print(
        f"Processing {len(word_groups)} unique word groups spanning {len(rows_to_process)} entries (concurrency={CONCURRENCY_LIMIT})…"
    )

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()

    tasks = [
        asyncio.create_task(
            process_word_group(
                target_word, group_rows, definitions_map, semaphore, file_lock
            )
        )
        for target_word, group_rows in word_groups.items()
    ]

    await tqdm.gather(*tasks, desc="Filling homographs")
    print(f"Results appended to {OUTPUT_JSONL}")


if __name__ == "__main__":
    asyncio.run(main())
