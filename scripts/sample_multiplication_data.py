def apply_template(problem):
    messages = [
        {"role": "user", "content": problem}
    ]
    return messages
        
llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, enable_prefix_caching=True, max_model_len=20000)

convs = [apply_template(problem) for problem in batch_problems]
completions = llm.chat(
    messages=convs,
    sampling_params=SamplingParams(
        n=args.K,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )
)
import os