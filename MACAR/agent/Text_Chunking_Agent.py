import json
import re
from tool.prompt import Text_Chunking_Agent, Text_Chunking_Agent_0
from tool.API import generate

def text_chunking_agent(input_text):

    prompt_keyword = Text_Chunking_Agent_0.format(input_text=input_text)
    keyword_response = generate(prompt_keyword)

    match = re.search(r'关键信息：\[(.*?)\]', keyword_response)
    if match:
        keyword = match.group(1)
    else:

        keyword = keyword_response.strip()

    prompt_chunk = Text_Chunking_Agent.format(input_text=input_text)
    chunk_response = generate(prompt_chunk)

    chunk_response = chunk_response.strip()
    if chunk_response.startswith("```json"):
        chunk_response = chunk_response[7:]
    if chunk_response.startswith("```"):
        chunk_response = chunk_response[3:]
    if chunk_response.endswith("```"):
        chunk_response = chunk_response[:-3]
    chunk_response = chunk_response.strip()

    try:
        chunks = json.loads(chunk_response)
    except json.JSONDecodeError:

        chunks = re.findall(r'"([^"]*)"', chunk_response)
    return keyword, chunks

