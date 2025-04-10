""" https://www.kaggle.com/datasets/deepmind/facts-grounding-examples?select=evaluation_prompts.csv """

import json

from .evaluator import EvaluationModel, MetricOutput
from ..oai_utils import OAICaller

PROMPT_JSON = \
"""You are a helpful and harmless AI assistant. You will be provided with a textual context and a model-generated response.
Your task is to analyze the response sentence by sentence and classify each sentence according to its relationship with the provided context.

**Instructions:**

1. **Decompose the response into individual sentences.**
2. **For each sentence, assign one of the following labels:**
    * **`supported`**: The sentence is entailed by the given context. Provide a supporting excerpt from the context. The supporting except must *fully* entail the sentence. If you need to cite multiple supporting excepts, simply concatenate them.
    * **`unsupported`**: The sentence is not entailed by the given context. No excerpt is needed for this label.
    * **`contradictory`**: The sentence is falsified by the given context. Provide a contradicting excerpt from the context.
    * **`no_rad`**: The sentence does not require factual attribution (e.g., greetings, questions, disclaimers).  No excerpt is needed for this label.
3. **For each label, provide a short rationale explaining your decision.**  The rationale should be separate from the excerpt.
4. **Be very strict with your `supported` and `contradictory` decisions.** Unless you can find straightforward, indisputable evidence excerpts *in the context* that a sentence is `supported` or `contradictory`, consider it `unsupported`. You should not employ world knowledge unless it is truly trivial.

**Input Format:**

The input will consist of two parts, clearly separated:

* **Context:**  The textual context used to generate the response.
* **Response:** The model-generated response to be analyzed.

**Output Format:**

For each sentence in the response, output a JSON object with the following fields:

* `"sentence"`: The sentence being analyzed.
* `"label"`: One of `supported`, `unsupported`, `contradictory`, or `no_rad`.
* `"rationale"`: A brief explanation for the assigned label.
* `"excerpt"`:  A relevant excerpt from the context. Only required for `supported` and `contradictory` labels.

Output each JSON object on a new line.

**Example:**

**Input:**

```
Context: Apples are red fruits. Bananas are yellow fruits.

Response: Apples are red. Bananas are green. Bananas are cheaper than apples. Enjoy your fruit!
```

**Output:**

{"sentence": "Apples are red.", "label": "supported", "rationale": "The context explicitly states that apples are red.", "excerpt": "Apples are red fruits."}
{"sentence": "Bananas are green.", "label": "contradictory", "rationale": "The context states that bananas are yellow, not green.", "excerpt": "Bananas are yellow fruits."}
{"sentence": "Bananas are cheaper than apples.", "label": "unsupported", "rationale": "The context does not mention the price of bananas or apples.", "excerpt": null}
{"sentence": "Enjoy your fruit!", "label": "no_rad", "rationale": "This is a general expression and does not require factual attribution.", "excerpt": null}

**Now, please analyze the following context and response:**

**Context:**
{{context_document}}

**Response:**
{{response}}"""

# Grounding json evaluation
def parse_structured_json(ans):
    if '```json' in ans:
        ans = ans.split('```json')[1].split('```')[0]
    ans = ans.strip()
    ans = ans.replace('}\n', '}\n@\n@\n')
    parsed_answers = []
    for line in ans.split('\n@\n@\n'):
        try:
            line = line.replace('\n', ' ')
            line = line.replace("\\'", "'")
            parsed = json.loads(line)
            parsed_answers.append(parsed)
        except Exception:
            pass
    if len(parsed_answers) > 0:
        bool_ans = all(d['label'] == 'supported' or d['label'] == 'no_rad' for d in parsed_answers)
    else:
        bool_ans = False
    return bool_ans, parsed_answers

class FACTSGJudge(OAICaller, EvaluationModel):
    """ FactsGrounding LLM-as-judge """
    def __init__(self, 
                 model="meta-llama/llama-3.3-70b-instruct",
                 base_url="https://openrouter.ai/api/v1",
                 **kwargs):
        super().__init__(model=model, base_url=base_url, **kwargs)

    def process_one(self, sample:dict, debug=False) -> MetricOutput:
        messages = [
            {
                "role": "user", 
                "content": PROMPT_JSON.replace("{{context_document}}", sample[self.context_column])
                                      .replace("{{response}}", sample[self.claim_column])
            },
        ]
        llm_return = self.llm_call(messages, debug)
        if debug:
            print(llm_return)
        judge, _ = parse_structured_json(llm_return)
        return MetricOutput(**{
            "score": int(judge),
            "judge_model": self.model_name,
            "extra_output": llm_return
        })
