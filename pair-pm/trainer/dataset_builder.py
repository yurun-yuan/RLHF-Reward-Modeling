########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Mistral model which should be handled carefully because it does not have an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# import evaluate
from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)

NUM_PROC = 4


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    train_set_path: Optional[str] = field(
        default="RLHFlow/preference_data_v2_80K_wsafety",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "The split of the dataset to use"},
    )
    hf_dataset_output_path: Optional[str] = field(
        default="RyanYr/RLHFlow-preference_data_v2_800K_wsafety-tokenized",
        metadata={"help": "HF repo for the dataset"},
    )
    max_length: Optional[int] = field(default=4096)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the value-head model and tokenizer.
tokenizer_name = script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast = False)

# Adjusted according to the base model
# Need to do this for the models that don't have an official pad token.
print("tokenizer.pad_token ", tokenizer.pad_token, "tokenizer.pad_token_id", tokenizer.pad_token_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print("tokenizer.padding_side", tokenizer.padding_side)
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length
# tokenizer.padding_side = "right"



# Get the dataset
train_path = script_args.train_set_path
output_name = script_args.hf_dataset_output_path


token_id_A = tokenizer.encode("A", add_special_tokens=False)
token_id_B = tokenizer.encode("B", add_special_tokens=False)
assert len(token_id_A) == 1 and len(token_id_B) == 1
LABELS = {
    "A": token_id_A[0],
    "B": token_id_B[0]
}


def build_dataset(tokenizer, train_path, eval_path):

    def tokenize(sample):
        message = sample['messages']
        candidates = message[0]
        label_id = LABELS[message[1]["content"]]
        candidates = tokenizer.apply_chat_template([candidates], tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
        candidates = tokenizer(candidates, truncation=True)
        return {
            "input_ids": candidates["input_ids"],
            "attention_mask": candidates["attention_mask"],
            "labels": label_id
        }

    ds = load_dataset(train_path, split=script_args.split).shuffle(seed=42)
    ds = ds.map(tokenize, num_proc=NUM_PROC)

    train_dataset = ds
    #eval_dataset = load_dataset(eval_path, split="train").shuffle(seed=42).select(range(500))
    eval_dataset = ds.select(range(500))
    return train_dataset, eval_dataset


train_dataset, eval_dataset = build_dataset(tokenizer, train_path, None)

train_dataset.push_to_hub(output_name)

train_dataset = train_dataset[:2]
print(train_dataset)
print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))

