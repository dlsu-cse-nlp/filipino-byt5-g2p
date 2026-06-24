from typing import List

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from pydantic import BaseModel

# INFO: Updated as of 22 June 2026
# Uses the new Agent Platform


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


client = genai.Client(
    vertexai=True,
    project="basic-formula-489010-n5",
    location="global",
    http_options=HttpOptions(api_version="v1"),
)

prompt = """
<instructions>
Generate 20 Filipino sentences each for each pronunciation variation of the word below.
Divide it evenly among all definitions. Respond only with the sentences according
to the requested JSON schema. Do not inflect or conjugate the word; use it exactly as it appears. Generate
fairly simple sentences and be as unambiguous with the definitions as possible, but there should be variety in structure.
</instructions>
<data>
{"word": "lukot", "matches": [{"prons": ["lu'kot"], "definitions": ["(Adjective) crumpled; rumpled; with creases or wrinkles"]}, {"prons": ["'lukot"], "definitions": ["(Noun) a species of small honeybee that makes bitterish honey", "(Noun) crease; wrinkle (in clothes, etc.)", "(Noun) crumpling; rumpling; wrinkling"]}]}
</data>
"""

print("=" * 40)
print("The following prompt will be fed to Gemini:")
print("-" * 40)
print(prompt)
print("=" * 40)

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Response,
    ),
)

output = Response.model_validate_json(response.text)
print(output.model_dump_json(indent=2))
