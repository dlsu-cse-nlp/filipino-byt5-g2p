"""
This script is used to test the Gemini API with a single "generate a sentence
for each possible pronunciation" request.
"""

# WARNING: This currently uses the old configuration with Vertex AI

from typing import List

from dotenv import dotenv_values
from google import genai
from google.genai import types
from pydantic import BaseModel


# TODO: Move these into a separate class
class Sentence(BaseModel):
    pronunciation: int
    sentence: str


class Response(BaseModel):
    """JSON schema for Gemini API's structured output"""

    word: str
    answers: List[Sentence]


# Set up Gemini API
config = dotenv_values(".env")
key = config["VERTEX_API_KEY"]
client = genai.Client(api_key=key, vertexai=True)

prompt = """
<instructions>
The Filipino word given below has several possible pronunciations, given in a
list. For each unique pronunciation, generate sentences that use the word
following the definition that corresponds to the pronunciation. The objective
is to create a list of sentences (in Filipino, not in IPA) that use the same
word in different meanings or contexts, four for each definition corresponding
to the pronunciations listed. Follow standard Filipino grammar. Exclude all
diacritics from your responses. You may perform conjugations if applicable.
Include the word itself and the index of its intended pronunciation in the list,
where the first entry is index 1. If any of the pronunciations listed are simply
variations and do not SIGNIFICANTLY change the meaning of the word, treat them
as a single pronunciation and generate four sentences using only the FIRST
pronunciation listed that has that specific meaning. Only in this case are you
to ignore the other variants. Otherwise, generate four sentences for each
pronunciation listed as instructed. Each sentence must be a separate entry
string in the lists of sentences in the output schema.
</instructions>
<word>bonifacio</word>
<pronunciations_list>
boni'paʃo
boni'faʃo
</pronunciations_list>
"""

print("=" * 40)
print("The following prompt will be fed to Gemini:")
print("-" * 40)
print(prompt)
print("=" * 40)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=Response.model_json_schema(),
        # This is where thinking_level lives
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,  # Keeps the 'thought' process out of the response
            thinking_budget=0,  # 0 strictly disables the reasoning stepthinking_level="MINIMAL"),
        ),
    ),
)

output = Response.model_validate_json(response.text)
print(output)
