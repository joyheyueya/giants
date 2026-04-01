from datasets import load_dataset, DatasetDict

# Load your dataset (replace with your dataset name and config)
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
dataset = load_dataset("open-r1/DAPO-Math-17k-Processed", "en")  # or load_dataset("json", data_files={"test": "your_file.json"})

# Choose the split to update
split_name = "train"
split = dataset[split_name]

# Add the new column
def add_extra_info(example, idx):
    return {"extra_info": {"index": idx, "split": split_name}}

def add_ability(example, idx):
    return {"ability": 'math'}

def add_reward_model(example, idx):
    return {"reward_model": {"ground_truth": example['answer'], "style": 'rule'}}

def add_datasource(example, idx):
    return {"data_source": example['source']}

def add_prompt(example, idx):
    return {"prompt": [{"content": example['prompt'] + " Let's think step by step and output the final answer within \\boxed{}.", "role": "user"}]}

def add_level(example, idx):
    return {"level": "hard"}

# Update the dataset dictionary
dataset[split_name] = dataset[split_name].map(add_extra_info, with_indices=True)
dataset[split_name] = dataset[split_name].map(add_ability, with_indices=True)
dataset[split_name] = dataset[split_name].map(add_prompt, with_indices=True)
dataset[split_name] = dataset[split_name].map(add_level, with_indices=True)

dataset[split_name].to_parquet("data/dapo_17k.parquet")