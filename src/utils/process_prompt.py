import asyncio
import json
from typing import List

from pydantic import BaseModel, ValidationError


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


async def process_prompt(
    index, prompt, client, model_name, semaphore, output_file, file_lock, gen_config
):
    if not prompt:
        return {"index": index, "succedelay = 2ss": False, "error": "ERR_EMPTY_PROMPT"}

    delay = 2

    while True:
        try:
            async with semaphore:
                response = await client.aio.models.generate_content(
                    model=model_name, contents=prompt, config=gen_config
                )

                output = Response.model_validate_json(response.text)
                result_entry = {
                    "index": index,
                    "success": True,
                    "content": output.model_dump(),
                    "error": None,
                }
            break

        except ValidationError as ve:
            print(f"\n[Index {index}] Validation Error: {ve}. Skipping...")
            result_entry = {
                "index": index,
                "success": False,
                "error": "ERR_VALIDATION_FAILED",
                "details": str(ve),
            }
            break

        except Exception as e:
            error_msg = str(e).lower()
            if (
                "429" in error_msg
                or "quota" in error_msg
                or "too many requests" in error_msg
            ):
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
            else:
                raise

    # Only write on success
    async with file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    return result_entry
