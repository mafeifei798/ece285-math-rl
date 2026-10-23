---
license: apache-2.0
base_model:
- Qwen/Qwen2.5-Math-7B
pipeline_tag: token-classification
datasets:
- HuggingFaceH4/prm800k-trl-dedup
---

> [!Warning]
> <div align="center">
> <b>
> 🚨 This repo differs from <a href=https://huggingface.co/Qwen/Qwen2.5-Math-7B-PRM800K>Qwen's PRM</a>. We trained our PRM based on <a href=https://huggingface.co/Qwen/Qwen2.5-Math-7B>Qwen2.5-Math-7B</a>, while Qwen's PRM is based on <a href=https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct>Qwen2.5-Math-7B-Instruct</a>.
> </b>
> </div>

# PURE's PRM based on Qwen2.5-Math-7B

## Introduction

**Our PRM is used to fine-tune LLM for better math reasoning capability.** See our [PURE GitHub repo](https://github.com/CJReinforce/PURE) for more details. It is obtained by fine-tuning **Qwen2.5-Math-7B** on the training set of open-source dataset [PRM800K](https://github.com/openai/prm800k). **We choose Qwen2.5-Math-7B instead of Qwen2.5-Math-7B-Instruct to keep the base model consistent with our baselines.** We treat the original 1 and 0 labels in PRM800K as our positive labels, while -1 as negative ones. To eliminate test data contamination, we also remove the PRM800K training samples that have the same math queries in MATH test set.


## Requirements
* `transformers>=4.40.0` for Qwen2.5-Math models. The latest version is recommended.

## Quick Start

> [!Important]
>
> **PURE's PRM** is a process reward model typically used for offering feedback on the quality of reasoning and intermediate steps rather than generation.

### Prerequisites
- Step Separation: We recommend using double line breaks ("\n\n") to separate individual steps within the solution.
- Reward Computation: After each step, we insert a token "`\n`". For reward calculation, we extract the probability score of this token and subtract negative probabilities from positive probabilities, resulting in a reward value between -1 and 1. We regard steps with reward > 0 as correct, otherwise as incorrect.

### 🤗 Hugging Face Transformers

1. Here we show a code snippet to show you how to use our PRM with `transformers`:

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def make_step_rewards(logits, token_masks):
    all_scores_res = []
    for sample, token_mask in zip(logits, token_masks):
        # sample: (seq_len, num_labels)
        probs = sample[token_mask].softmax(dim=-1)  # (num_steps, 2)
        process_reward = probs[:, 1] - probs[:, 0]  # (num_steps,)
        # weighted sum to approx. min, highly recommend when BoN eval and Fine-tuning LLM
        # weight = torch.softmax(
        #     -process_reward / 0.1, 
        #     dim=-1,
        # )
        # process_reward = weight * process_reward
        all_scores_res.append(process_reward.cpu().tolist())
    return all_scores_res

model_name = "jinachris/PURE-PRM-7B"
device = "auto"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    device_map=device, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval()

question = "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?"
steps = [
    "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
    "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.",
    "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.",
    "To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30})."
]

step_separator = "\n"
step_separator_token = tokenizer(
    step_separator, 
    add_special_tokens=False, 
    return_tensors='pt',
)['input_ids']
input_ids = tokenizer(
    question, 
    add_special_tokens=False, 
    return_tensors='pt',
)['input_ids']

score_ids = []
for step in steps:
    step_ids = tokenizer(
        step, 
        add_special_tokens=False, 
        return_tensors='pt',
    )['input_ids']
    input_ids = torch.cat(
        [input_ids, step_ids, step_separator_token], 
        dim=-1,
    )
    score_ids.append(input_ids.size(-1) - 1)

input_ids = input_ids.to(model.device)
token_masks = torch.zeros_like(input_ids, dtype=torch.bool)
token_masks[0, score_ids] = True
assert torch.all(input_ids[token_masks].to("cpu") == step_separator_token)

logits = model(input_ids).logits
step_reward = make_step_rewards(logits, token_masks)
print(step_reward)  # [[0.796875, 0.185546875, -0.0625, 0.078125]]

# For BoN eval, 
# uncomment the weighted sum part in `make_step_rewards` func, 
# then sum the rewards to get the final score (outcome reward): 
# torch.tensor(step_reward).sum(dim=-1)
```

2. For evaluation using Best-of-N method or on ProcessBench and PRMBench, refer to [our github repository](https://github.com/CJReinforce/PURE/tree/verl/PRM/eval).

## Citation

If you find our work useful, we would appreciate it if you could cite our work:

```
@article{cheng2025stop,
  title={Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning},
  author={Cheng, Jie and Qiao, Ruixi and Li, Lijun and Guo, Chao and Wang, Junle and Xiong, Gang and Lv, Yisheng and Wang, Fei-Yue},
  journal={arXiv preprint arXiv:2504.15275},
  year={2025}
}
```