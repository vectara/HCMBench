# HCMBench
Hallucination correction model benchmark

This repository provides a comprehensive benchmark for evaluating and comparing hallucination correction models. 
We offer tools and datasets to assess how effectively these models can mitigate factual inaccuracies in language generation.

## Download the dataset
- Download RAGTruth and FaithBench using the scripts below
```bash
cd data
./download_data.sh
```

- FAVA can be directly loaded with huggingface hub

- FACTSGrounding requires manual access and download, see [`data/FACTSGrounding/README.txt`](data/FACTSGrounding/README.txt).

## Run the evaluation 
An example configuration run:
```bash
python run.py sample_run.yaml
```

The pipeline runs as:
1. Run a correction model through a list of datasets containing **context** and llm generations(**claim**) based on the context, and generate the **corrected** response.
2. (Optional) The **corrected** response are pre-**processed** for metrics to evaluation.
3. Run a set of factuality metrics on the corrected or processed responses and report the numbers. 

Note:
- We use Huggingface's [datasets](https://huggingface.co/docs/datasets/en/index) for the data loading in [bench_data.py](bench_data.py).
- Each component in the pipeline is a processor (see [processor.py](pipeline/processor.py)) that takes a sample of the dataset as input, and store the outputs as new columns in the dataset. 
- The intermediate/final output is stored as `{output_path}/{data_name}/corrected.jsonl`.

## Result visualization
See [``display_results.ipynb``](display_results.ipynb)

## Hallucination Correction Model (HCM)
1. [IdenticalCorrectionModel](pipeline/correction/correction_model.py), returns the exact same text as the original input.
2. [FAVA](pipeline/correction/fava.py), (Source: https://fine-grained-hallucination.github.io/)

## Preprocessor
1. [ClaimExtractor](pipeline/preprocess/claim_extraction.py), extract atomic facts from a text.
2. [Sentencizer](pipeline/preprocess/sentence_split.py), sentencize the text, with optional decontextualization with LLM.

## Hallucination Evaluation Metric (HEM)
1. [HHEM](pipeline/evaluation/hhem.py) (Source: https://huggingface.co/vectara/hallucination_evaluation_model)
2. [Minicheck](pipeline/evaluation/minicheck.py) (Source: https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B)
3. [FACTSJudge](pipeline/evaluation/factsgrounding.py) (Source: https://arxiv.org/abs/2501.03200)
4. [AXCEL](pipeline/evaluation/axcel.py) (Source: https://arxiv.org/abs/2409.16984)

## Dataset
 - [RAGTruth](https://github.com/ParticleMedia/RAGTruth), only summarization task
 - [FAVA](https://huggingface.co/datasets/fava-uw/fava-data), 3% front data
 - [FaithBench](https://github.com/vectara/FaithBench), only "Unwanted" and "Consistent"
 - [FACTSGrounding](https://www.kaggle.com/datasets/deepmind/facts-grounding-examples), public set

## Add a new model / preprocessor / metric / dataset
- **For HCM:** Implement your own model class which inherits from [CorrectionModel](pipeline/correction/correction_model.py)
- **For preprocessor:** Implement your own preprocessor class which inherits from [Preproessor](pipeline/preprocess/preprocessor.py)
- **For HEM:** Implement your own model class which inherits from [EvaluationModel](pipeline/evaluation/evaluator.py)
- For new modules above, add import code to ``pipeline/__init__.py``
- **Add a new dataset:**
    - Add a new funciton with name ``load_{data_name}`` in ``bench_data.py``
    - Load the data in ``datasets.Dataset`` format, 
    - The dataset must have ``context`` and ``claim`` columns.

