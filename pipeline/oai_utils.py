""" The file implements a wrapper for OPENAI client with retry and cache """

from openai import OpenAI
from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_fixed
LOCATION = './LLM_cache'
memory = Memory(LOCATION, verbose=0)

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
