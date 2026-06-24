"""
This script concurrently sends "fill in the pronunciation" requests to the
Gemini API via Vertex AI.
"""

# WARNING: This currently uses the old configuration with Vertex AI

import argparse
import asyncio
import json
import os
from typing import List

from dotenv import dotenv_values
from google import genai
from google.genai import types
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from src.utils.generate_prompt.transcribe import generate_prompt
from src.utils.homographs import homographs
from src.utils.process_prompt import process_prompt

DEFAULT_DATASET_PATH = "data/newsph-nli/newsph-nli.txt"
OUTPUT_FILENAME = "results_gemini_2.5_ambiguous.jsonl"
MODEL_NAME = "gemini-3-flash-preview"
CONCURRENCY_LIMIT = 25

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
args = parser.parse_args()

config = dotenv_values(".env")
VERTEX_KEY = config["VERTEX_API_KEY"]


class Response(BaseModel):
    """JSON schema for Gemini API's structured output"""

    answers: List[str]


async def main():
    # Set up Gemini API
    client = genai.Client(api_key=VERTEX_KEY, vertexai=True)

    # NOTE: Can test with a smaller batch first if needed
    # sentences = sentences[:100]

    print("Here")
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"{len(sentences)} loaded from dataset. Generating prompts...")
    prompts = []
    for s in tqdm(sentences):
        words, choices, _ = homographs(s)
        if words:
            prompts.append(
                generate_prompt(sentence=s, pronunciations=choices, words=words)
            )
        else:
            prompts.append(None)

    # --- CHECKPOINT LOGIC: Identify completed indices ---
    completed_indices = set()
    if os.path.exists(OUTPUT_FILENAME):
        with open(OUTPUT_FILENAME, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("success"):
                        completed_indices.add(data["index"])
                except json.JSONDecodeError:
                    continue
    print(f"Skipping {len(completed_indices)} already processed items.")
    # ----------------------------------------------------

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()

    # Delete existing file if it exists...
    # open(OUTPUT_FILENAME, "w").close()

    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=Response.model_json_schema(),
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        # thinking_config=types.ThinkingConfig(
        #    include_thoughts=False,  # Keeps the 'thought' process out of the response
        #    thinking_budget=0,  # 0 strictly disables the reasoning stepthinking_level="MINIMAL"),
        # ),
    )

    # Set concurrency limits
    print(f"Concurrency limit: {CONCURRENCY_LIMIT}")
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
        if i not in completed_indices
    ]

    # Process all tasks while showing an async progress bar
    results = [await t for t in tqdm.as_completed(tasks, total=len(tasks))]

    print("=" * 40)
    print(f"Saved results to {OUTPUT_FILENAME}.")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
