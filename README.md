# HCMBench
Hallucination correction model benchmark

## Run the evaluation 
Run the correction model and hallucination detection metrics.
```
python run.py sample_run.yaml
```

## Result visualization
See [``display_results.ipynb``](display_results.ipynb)

## Hallucination Correction Model (HCM)
1. [IdenticalCorrectionModel](correction/CorrectionModel.py), returns the exact same text as the original input.

## Hallucination Evaluation Metric (HEM)
1. [HHEM](evaluation/HHEM.py) (Source: https://huggingface.co/vectara/hallucination_evaluation_model)
2. [Minicheck](evaluation/Minicheck.py) (Source: https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B)
3. [AXCEL](evaluation/Axcel.py) (Source: https://arxiv.org/abs/2409.16984)

## Dataset
 - [RAGTruth](https://github.com/ParticleMedia/RAGTruth), only summarization task
 - [FAVA](https://huggingface.co/datasets/fava-uw/fava-data), 10% front data
 - [FaithBench](https://github.com/vectara/FaithBench), only "Unwanted" and "Consistent"

## Add a new model / metric / dataset
- **Add a new HCM:**
    - Implement your own model class which inherits from [CorrectionModel](correction/CorrectionModel.py)
    - Add import code to ``correction/__init__.py``
- **Add a new HEM:**
    - Implement your own model class which inherits from [EvaluationModel](evaluation/Evaluator.py)
    - Add import code to ``evaluation/__init__.py``
- **Add a new dataset:**
    - Add a new funciton with name ``load_{data_name}`` in ``BenchData.py``
    - Load the data in ``datasets.Dataset`` format, 
    - The dataset must have ``context`` and ``claim`` columns.

