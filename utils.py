import json
from openai import OpenAI
from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_fixed
import time
location = './LLM_cache'
memory = Memory(location, verbose=0)

def load_jsonl(input_path):
    with open(input_path, "r", encoding="UTF-8") as f:
        data = [json.loads(line) for line in f]
    return data

def dump2jsonl(lines, output_path):
    with open(output_path, "w", encoding="UTF-8") as f:
        for line in lines:
            f.write(json.dumps(line) + '\n') 

@memory.cache
@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def get_LLM_response(base_url, model, messages, **kwargs):
    client = OpenAI(base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return completion