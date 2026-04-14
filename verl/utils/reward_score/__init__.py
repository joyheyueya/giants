# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    raise NotImplementedError(
        f"No built-in reward function for {data_source=}. "
        "Set `custom_reward_function.path` to point at your scorer, e.g. "
        "`verl/utils/reward_score/insight_similarity/compute_score.py`."
    )
