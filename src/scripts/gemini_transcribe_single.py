"""
This script is used to test the Gemini API with a single "fill in the
pronunciation" request, reading the input from fun.csv.
"""

import csv
import json
from typing import List

from dotenv import dotenv_values  # still needed for other env vars if any
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.utils.generate_prompt.transcribe import generate_prompt
from src.utils.homographs import fill_template, homographs


# TODO: Use argparse
class Response(BaseModel):
    """JSON schema for Gemini API's structured output"""

    answers: List[str]


# Set up Gemini API – using a service account (no API key)
config = dotenv_values(".env")  # keep if you have other env variables
PROJECT_ID = "basic-formula-489010-n5"
LOCATION = "global"  # global endpoint for Flash models

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    # No api_key – authentication via GOOGLE_APPLICATION_CREDENTIALS
)

gen_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_json_schema=Response.model_json_schema(),
    thinking_config=types.ThinkingConfig(thinking_level="LOW"),
)


def load_definitions(path="finalfinalfinal.jsonl"):
    defs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            word = entry["word"]
            defs[word] = entry["matches"]
    return defs


# Read the first entry from fun.csv
with open("fun.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    # Take the first row only
    for row in reader:
        word = row["word"]
        pronunciation = row["pronunciation"]
        sentence = row["sentence"]
        break

# Extract homograph information given the sentence and the target word & pronunciation
ambiguous_words, choices, output_template = homographs(sentence, word, pronunciation)

print(ambiguous_words)
print(output_template)

definitions_map = load_definitions()
prompt = generate_prompt(
    sentence=sentence,
    pronunciations=choices,
    words=ambiguous_words,
    definitions_map=definitions_map,
)

print("=" * 40)
print("The following prompt will be fed to Gemini:")
print("-" * 40)
print(prompt)
print("=" * 40)

# Use the latest global Flash model
response = client.models.generate_content(
    model="gemini-3.5-flash",  # global, fast, supports structured output
    contents=prompt,
    config=gen_config,
)

output = Response.model_validate_json(response.text)

print(fill_template(output_template, output.answers))
