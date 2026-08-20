"""
This script concurrently sends "generate a sentence for each possible
pronunciation" requests to the Gemini API via Vertex AI (new version).
"""

# INFO: Updated as of 22 June 2026
# Uses the new Agent Platform

import asyncio
import json
from typing import List

from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from src.utils.process_prompt import process_prompt

PROJECT_ID = "basic-formula-489010-n5"
LOCATION = "global"
CONCURRENCY_LIMIT = 25
MODEL_NAME = "gemini-3.5-flash"

INPUT_JSONL = "data/wiktionary-scrape/ambiguous_prons_cleaned.jsonl"
OUTPUT_FILENAME = "data/wiktionary-scrape/generated/results_gemini_3.5.jsonl"


class DefinitionSentences(BaseModel):
    definition: str
    sentences: List[str]


class PronunciationGroup(BaseModel):
    pronunciation: str
    definitions: List[DefinitionSentences]


class Response(BaseModel):
    """JSON schema for Gemini API's structured output"""

    word: str
    results: List[PronunciationGroup]


def load_word_data(jsonl_path: str) -> List[dict]:
    """Read each line as a JSON object containing 'word' and 'matches'."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def generate_prompt(word_data: dict) -> str:
    """
    Build the prompt using the fixed instructions and the word data as JSON.
    """
    instructions = """
<instructions>
Generate 20 Filipino sentences each for each pronunciation variation of the word below. Divide it evenly among all definitions. Respond only with the sentences according to the requested JSON schema. Do not inflect or conjugate the word; use it exactly as it appears. Generate fairly simple sentences and be as unambiguous with the definitions as possible, but there should be variety in structure. If multiple possible pronunciations are valid for a definition, take only the first.
</instructions>
"""
    data_json = json.dumps(word_data, ensure_ascii=False)
    return f"{instructions}\n<data>\n{data_json}\n</data>"


async def main():
    # Set up Gemini client via Vertex AI (uses ADC for auth)
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

    # Load all word entries from the JSONL file
    word_entries = load_word_data(INPUT_JSONL)
    prompts = [generate_prompt(entry) for entry in word_entries]

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()

    # Clear output file
    open(OUTPUT_FILENAME, "w").close()

    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=Response.model_json_schema(),
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    )

    print(f"Concurrency limit: {CONCURRENCY_LIMIT}")
    print(f"Processing {len(prompts)} words...")

    tasks = [
        asyncio.create_task(
            process_prompt(
                i,
                prompt,
                client,
                MODEL_NAME,
                semaphore,
                OUTPUT_FILENAME,
                file_lock,
                gen_config,
            )
        )
        for i, prompt in enumerate(prompts, start=1)
    ]

    # Wait for all tasks with a progress bar
    await tqdm.gather(*tasks, desc="Generating sentences")

    print("=" * 40)
    print(f"Saved results to {OUTPUT_FILENAME}.")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
