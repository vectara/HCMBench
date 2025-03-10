from .evaluator import EvaluationModel, MetricOutput
import time
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from utils import get_LLM_response

SYSTEM_PROMPT = """You are given two texts, a source text and derived text. Verify if the derived text is factually correct with respect to the source. Use the following step-by-step instructions to assess factual correctness of derived text.
 Step 1 - Extract all the facts from the derived text.
 Step 2 - Check if the extracted facts can
 be verified from the source text.
 Step 3 - Rate the correctness of each fact on the scale of 1 to 5 based on the verification from previous step.
 Step 4 - Generate output in a consistent format following the format of the examples given below."""

EXAMPLE1_INPUT = """<Source Text>
Manchester City are keen to sign Anderlecht teenager Evangelos Patoulidis. The 14-year-old playmaker is regarded as one of the best talents to emerge from Anderlecht’s youth set-up and has also attracted attention from Arsenal and Barcelona. The Belgian starlet rejected a move to Barcelona’s La Masia academy when he was 12 as his family wanted him to continue his studies . He has continued to impress and City have held discussions with Anderlecht chairman Roger Vanden Stock in the hope of agreeing a compensation package. Manuel Pellegrini is looked to build for the future by snapping up hot property Evangelos Patoulidis.
</Source Text>

<Derived Text>
Evangelos patoulidis is regarded as one of the best players to emerge from anderlecht youth. He has also attracted attention from arsenal and barcelona. The belgian starlet rejected a move to barcelona’s la masia academy. The 14-year-old has attracted interest from barcelona to barcelona.
</Derived Text>"""

EXAMPLE1_OUTPUT = """Let’s verify the factual accuracy of the derived text step by step:

1. Evangelos Patoulidis is Regarded as One of the Best Players to Emerge from Anderlecht Youth:
    - **Derived Text:** Evangelos Patoulidis is regarded as one of the best players to emerge from Anderlecht youth.
    - **Source Text:** The source text states that Patoulidis is regarded as "one of the best talents to emerge from Anderlecht’s youth set-up".
    - **Verification:** Correct. Rating: 5

2. He Has Also Attracted Attention from Arsenal and Barcelona:
    - **Derived Text:** He has also attracted attention from Arsenal and Barcelona.
    - **Source Text:** This fact is mentioned verbatim in the source text.
    - **Verification:** Correct. Rating: 5

3. The Belgian Starlet Rejected a Move to Barcelona’s La Masia Academy:
    - **Derived Text:** The Belgian starlet rejected a move to Barcelona’s La Masia academy.
    - **Source Text:** The source text confirms this fact.
    - **Verification:** Correct. Rating: 5

4. The 14-Year-Old Has Attracted Interest from Barcelona to Barcelona:
    - **Derived Text:** The 14-year-old has attracted interest from Barcelona to Barcelona.
    - **Source Text:** This statement is confusing and not supported by the source text.
    - **Verification:** Incorrect. Rating: 1
"""

INPUT_TEMPLATE = """<Source Text>
{source_text}
</Source Text>

<Derived Text>
{derived_text}
</Derived Text>"""

def parse_output(output):
    lines = output.split('\n')
    scores = []
    for line in lines:
        if "Rating: " in line and "**Verification:**" in line:
            score = line.split(":")[-1].strip()
            scores.append(float(score))
    return min(scores)

class AXCEL(EvaluationModel):
    """AXCEL LLM-as-judge for evaluating generated output.
    """
    def __init__(self, model_path="anthropic/claude-3.5-sonnet", 
        base_url="https://openrouter.ai/api/v1",
        RPS=20):
        super().__init__(model_name=type(self).__name__ + '#' + model_path)
        self.model = model_path
        self.base_url = base_url
        self.RPS = RPS
        
    def predict_one(self, claim: str, context: str, debug=False) -> MetricOutput:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE1_INPUT},
            {"role": "assistant", "content": EXAMPLE1_OUTPUT},
            {"role": "user", "content": INPUT_TEMPLATE.format(source_text=context, derived_text=claim)}
        ]
        completion = get_LLM_response(
            base_url = self.base_url, 
            model = self.model, 
            messages = messages,
            temperature=0.0,
            frequency_penalty=1,
            max_tokens=1000)
        time.sleep(1.0/self.RPS)
        llm_return = completion.choices[0].message.content
        if debug:
            print(llm_return)
        score = parse_output(llm_return)
        return MetricOutput(**{
            "claim": claim,
            "context": context,
            "score": score / 5,
            "judge_model": self.model_name,
            "extra_output": llm_return
        })

def main():
    model = AXCEL()
    claim = "Chenyu has 10 followers on Twitter."
    context = "Chenyu only had 5 followers on Twitter before, but now his followers have doubled."
    output = model.predict_one(claim, context, debug=False)
    print(output)

if __name__ == '__main__':
    main()