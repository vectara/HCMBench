from datasets import load_dataset, Dataset, concatenate_datasets
from utils import load_jsonl
import os

def load_C2DD2C():
    c2d_d2c = load_dataset("lytang/C2D-and-D2C-MiniCheck")
    c2d_d2c = c2d_d2c.rename_column('doc', 'context')
    c2d_d2c = concatenate_datasets([c2d_d2c["c2d"], c2d_d2c["d2c"]])
    return c2d_d2c

def load_ragtruth(data_dir="RAGTruth/dataset", split="test"):
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

if __name__ == '__main__':
    ragtruth = load_ragtruth()
    c2dd2c = load_C2DD2C()
    mix_data = concatenate_datasets([ragtruth, c2dd2c])
    print(mix_data)
    breakpoint()