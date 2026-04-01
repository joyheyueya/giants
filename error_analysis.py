import argparse
import os
import json
from tqdm import tqdm
import numpy as np

def read_json_files(directory):
    rollouts = {}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                # Process the JSON data here
                rollouts[filename] = data
    return rollouts


import re


def get_repeat_count(text):
    if len(text) < 50:
        return 0  # or 1 if you want to count the short input itself
    pattern = re.escape(text[-100:])
    return len(re.findall(f'(?={pattern})', text))

def is_terminated(text):
    return "</answer>" in text or "\\boxed{" in text


def process_rollout(rollout):
    input = rollout['input']
    output = rollout['output']
    score = rollout['score']
    index = rollout['index']
    return {
        'input': input,
        'output': output,
        'score': score,
        'index': index,
        'repeats': get_repeat_count(output),
        'terminated': is_terminated(output),
    }
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read JSON files in a directory')
    parser.add_argument('--path', type=str, help='Path to the directory containing JSON files')
    args = parser.parse_args()

    rollouts = read_json_files(args.path)
    processed_rollouts = {}

    for rollout_name, rollout_data in rollouts.items():
        print(f'Processing Rollout: {rollout_name}')
        processed_rollouts[rollout_name] = []   
        for rollout in tqdm(rollout_data, total=len(rollout_data)):
            processed_rollouts[rollout_name].append(process_rollout(rollout))



    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    rollout_names = list(processed_rollouts.keys())
    rollout_ids = [int(x.split("_")[0]) for x in rollout_names]
    sorted_indices = np.argsort(rollout_ids)
    rollout_names = np.array(rollout_names)[sorted_indices]
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    fig, axs = plt.subplots(len(rollout_names), 1, figsize=(10, 5 * len(rollout_names)))

    for i, rollout_name in enumerate(rollout_names):
        rollouts = processed_rollouts[rollout_name]
        repeats = np.array([rollout['repeats'] for rollout in rollouts if rollout['score']==0.0])
        terminated = np.array([rollout['terminated'] for rollout in rollouts if rollout['score']==0.0])
        terminated_length = np.array([len(tokenizer.encode(rollout['output'])) for rollout in rollouts if rollout['terminated']==True])
        ax = axs[i] if len(rollout_names) > 1 else axs
        ax.bar(['Repeats > 5', 'Terminated', 'Terminated length'], [sum(repeats > 5) / len(repeats), sum(terminated) / len(terminated), np.mean(terminated_length)])
        # ax.bar(['Repeats > 5', 'Terminated'], [sum(repeats > 5) / len(repeats), sum(terminated) / len(terminated)])
        ax.set_title(rollout_name)

    plt.tight_layout()
    plt.savefig('error_analysis_negatives.png')

            