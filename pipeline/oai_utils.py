""" The file implements a wrapper for OPENAI client with retry and cache """
import os

from openai import OpenAI
from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_fixed

from .processor import Processor

LOCATION = './LLM_cache'
memory = Memory(LOCATION, verbose=0)

@memory.cache
@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def get_LLM_response(base_url, model, messages, api_key_env, extra_body=None, **kwargs):
    client = OpenAI(base_url=base_url,
                    api_key=os.getenv(api_key_env))
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body=extra_body,
        **kwargs
    )
    return completion

class OAICaller:
    """ Wrapper for a OpenAI compatible LLM call. """
    def __init__(self, model, max_tokens=2000, 
                 base_url="https://api.openai.com/v1", 
                 api_key_env="OPENAI_API_KEY",
                 extra_body=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.extra_body = extra_body

    def llm_call(self, messages, debug=False)->str:
        completion = get_LLM_response(
            base_url = self.base_url,
            model = self.model,
            api_key_env=self.api_key_env,
            messages = messages,
            temperature=0.0,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body
        )
        llm_return = completion.choices[0].message.content
        if debug:
            print(llm_return)
        return llm_return
