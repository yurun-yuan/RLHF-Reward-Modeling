from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import os

# All the datasets should be pre-processed into standard format.
all_dirs = [
    "allenai/reward-bench"
]

tokenizer_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer_plain = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer_plain.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"
prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"

def process_example(example):
    prompt = example['prompt']
    response_chosen = example["chosen"]
    response_rejected = example["rejected"]
    instruction = [{"role": "user", "content": prompt}]
    context = tokenizer_plain.apply_chat_template(instruction, tokenize=False)
    responses = [response_chosen, response_rejected]

    # we swap order to mitigate position bias
    chosen_position = np.random.randint(2)
    response_A = responses[chosen_position]
    response_B = responses[1 - chosen_position]
    prompt = prompt_template.format(context=context, response_A=response_A, response_B=response_B)
    label = ['A', 'B'][chosen_position]
    response = label

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return {"messages": messages, }

all_datasets = []
for ds_dir in all_dirs:
    ds = load_dataset(ds_dir, split='train')
    ds_new = ds.map(process_example,num_proc=32, remove_columns=ds.column_names, )
    all_datasets.append(ds_new)


if len(all_datasets) == 1:
    combined_dataset = all_datasets[0]
else:
    tmp = concatenate_datasets([all_datasets[0], all_datasets[1]])
    for i in range(2, len(all_datasets)):
        tmp = concatenate_datasets([tmp, all_datasets[i]])
    combined_dataset = tmp

combined_dataset = combined_dataset.shuffle(seed=42)


DatasetDict({'train': combined_dataset}).push_to_hub(os.environ['HF_DATASET_ID'])
