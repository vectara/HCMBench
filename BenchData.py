from datasets import load_dataset, Dataset, concatenate_datasets
from utils import load_jsonl
import os

def load_C2DD2C():
    c2d_d2c = load_dataset("lytang/C2D-and-D2C-MiniCheck")
    c2d_d2c = c2d_d2c.rename_column('doc', 'context')
    c2d_d2c = concatenate_datasets([c2d_d2c["c2d"], c2d_d2c["d2c"]])
    return c2d_d2c

def load_FAVA(data_dir='fava-uw/fava-data'):
    data = load_dataset(data_dir, split='train[:3%]')
    def process_fava(sample):
        context = sample["prompt"].partition("Read the following references:")[2].strip()
        context = context.rpartition("Please identify all the errors in the following passage using the references provided and suggest edits:\nText:")[0].strip()
        claim = sample["prompt"].rpartition("Please identify all the errors in the following passage using the references provided and suggest edits:\nText:")[2].strip()
        sample["context"] = context
        sample["claim"] = claim
        sample["extra_label"] = sample["completion"]
        return sample
    data = data.map(process_fava, remove_columns =["prompt", "completion"])
    return data

def load_RAGTruth(data_dir="data/RAGTruth/dataset", split="test"):
    response = load_jsonl(os.path.join(data_dir, "response.jsonl"))
    if split is not None:
        response = [item for item in response if item["split"] == split]
    source_info = load_jsonl(os.path.join(data_dir, "source_info.jsonl"))
    source_info = {item["source_id"]:item for item in source_info}

    data = []
    for line in response:
        source = source_info[line["source_id"]]
        sample = {
            "model": line["model"],
            "claim": line["response"],
            "label": 1 if len(line["labels"]) == 0 else 0,
            "extra_label": {
                "span": line["labels"]
            },
            "metadata": {
                "dataset": "RAGTruth",
                "split": split,
                "source_id": line["source_id"],
                "task_type": source["task_type"],
                "source": source["source"]
            }
        }
        if source["task_type"] == "Summary": 
            sample["context"] = source["source_info"]
        else:
            continue
        
        data.append(sample)
    data = Dataset.from_list(data)
    return data

def load_FaithBench(data_dir="data/FaithBench"):
    data = load_dataset("csv", data_files=os.path.join(data_dir, 'FaithBench.csv'), split='train')
    data = data.filter(lambda x: x['worst-label'] in ['Unwanted', 'Consistent'])
    def convert_label(sample):
        sample["label"] = 1 if sample["worst-label"] == 'Consistent' else 0
        sample["extra_label"] = {
            "best-label": sample["best-label"],
            "worst-label": sample["worst-label"],
        }
        return sample
    data = data.map(convert_label, remove_columns=['worst-label', 'best-label'])
    data = data.rename_column('summary', 'claim')
    data = data.rename_column('source', 'context')
    data = data.rename_column('LLM', 'model')
    return data

def load_FACTSGrounding(data_dir="data/FACTSGrounding"):
    data = load_dataset("csv", data_files=os.path.join(data_dir, 'data.csv'), split='train')
    response_columns = [column.partition('-response')[0] for column in data.column_names if column.endswith('-response')]
    procssed = []
    for sample in data:
        for llm in response_columns:
            procssed.append({
                "claim": sample[f"{llm}-response"].strip(),
                "context": sample['context_document'].strip(),
                "model": llm,
                "metadata": {
                    "system_instruction": sample["system_instruction"],
                    "user_request": sample["user_request"]
                }
            })
    return Dataset.from_list(procssed)

if __name__ == '__main__':
    # ragtruth = load_RAGTruth()
    # c2dd2c = load_C2DD2C()
    # mix_data = concatenate_datasets([ragtruth, c2dd2c])
    # print(mix_data)
    # data = load_FAVA()
    # faithbench = load_FaithBench()
    facts = load_FACTSGrounding()
    breakpoint()