from vllm import LLM, SamplingParams
import datasets
from transformers import AutoTokenizer
from math_eval import is_equiv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='openthoughts-sft-qwen3-1.7b-base-0501')
parser.add_argument('--output_name', type=str, default='aime_2025_responses_openthoughts-sft-qwen3-1.7b')
args = parser.parse_args()

ds = datasets.load_dataset('active-reasoning/math_reasoning_benchmark', split='AIME2025')
df = ds.to_pandas()

model_name = args.model_name
sampling_params = SamplingParams(temperature=0.6, n=8, max_tokens=8192)
model = LLM(model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Generating responses...')
math_prefix = "Solve the following math problem. Give your final answer as \\boxed{}."
apply_prefix = lambda x: f'{math_prefix}\n{x}'
all_prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': apply_prefix(problem)}], add_generation_prompt=True, tokenize=False) for problem in df['problem']]
all_responses = model.generate(all_prompts, sampling_params)
all_responses = [[response.outputs[i].text for i in range(len(response.outputs))] for response in all_responses]
responses_correct = [[is_equiv(df['answer'][i], response) for response in all_responses[i]] for i in range(len(df))]
success_rate = [np.mean(responses_correct[i]) for i in range(len(df))]

df['response'] = all_responses
df['correct'] = responses_correct
df['success_rate'] = success_rate

print('Average success rate: ', np.mean(success_rate))

ds = datasets.Dataset.from_pandas(df)
ds.push_to_hub(args.output_name)