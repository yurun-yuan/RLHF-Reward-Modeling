########################
# This script is modified from the TRL package https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/reward_modeling.py
# This script is designed for the reward modeling with Mistral model which should be handled carefully because it does not have an official pad token
# If you have any question, feel free to send me an email via wx13@illinois.edu
########################
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

NUM_PROC = 4


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})

    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    # for 8 GPU, the global batch size is 512
    gradient_accumulation_steps: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=2e-6)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default="RyanYr/RLHFlow-preference_data_v2_800K_wsafety-tokenized",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default="RyanYr/RLHFlow-preference_data_v2_800K_wsafety-tokenized",
        metadata={"help": "The dir of the subset of the eval data to use"},
    )
    local_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pretrained model. Used if not none to initialize pretrained model"},
    )
    output_path: Optional[str] = field(
        default="./models/llama3_rm_test",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        # default="adamw_hf",
        default="paged_adamw_32bit",
        # default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=4096)

    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables pushing checkpointing."},
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Model ID on the Hugging Face Hub"},
    )
    hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Your Hugging Face token"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the total amount of checkpoints."},
    )
    torch_random_seed: Optional[int] = field(
        default=0,
        metadata={"help": "The random seed for torch"},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

torch.manual_seed(script_args.torch_random_seed)

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


token_id_A = tokenizer.encode("A", add_special_tokens=False)
token_id_B = tokenizer.encode("B", add_special_tokens=False)
assert len(token_id_A) == 1 and len(token_id_B) == 1
LABELS = {
    "A": token_id_A[0],
    "B": token_id_B[0]
}


# Get the dataset
train_path = script_args.train_set_path
eval_path = script_args.eval_set_path
output_name = script_args.output_path

train_dataset = load_dataset(train_path, split="train")
eval_dataset = train_dataset.select(range(500))
print("Training set: ", len(train_dataset), " Eval set: ", len(eval_dataset))

# Define the trainer


# Define the trainer
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    hub_token=script_args.hub_token,
    save_total_limit=script_args.save_total_limit,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=3,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to='wandb',
)

print(
    "training_args",
    training_args,
)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name if script_args.local_model_path is None else script_args.local_model_path, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
)

model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.pad_token_id

# Need transformers>=4.43.4
model.resize_token_embeddings(len(tokenizer))

print(model.config)

num_proc = NUM_PROC  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_input = self.tokenizer.pad(
            {
                "input_ids": [feature["input_ids"] for feature in features], 
                "attention_mask": [feature["attention_mask"] for feature in features]
            },
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch_input["input_ids"],
            "attention_mask": batch_input["attention_mask"],
            "labels": [feature["labels"] for feature in features],
            "return_loss": True,
        }
        return batch


# Define the trainer
def compute_metrics(eval_pred):
    result = {}
    logit_A = eval_pred.predictions[:, -1, token_id_A].item()
    logit_B = eval_pred.predictions[:, -1, token_id_B].item()
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result['accuracy'] = np.sum(
        pos_predictions_scores > neg_predictions_scores) / len(pos_predictions_scores)
    return result


class PreferenceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        logits = logits[:, -1, :].contiguous().view(-1, model.config.vocab_size)
        labels = inputs["labels"].view(-1)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        if return_outputs:
            return loss, {"logits": logits, "labels": labels}
        return loss


# Train the model, woohoo.
trainer = PreferenceTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, 
        max_length=script_args.max_length, 
        padding='max_length'),
)


trainer.train()


print("Saving last checkpoint of the model")
#model.save_pretrained(output_name + "/last_checkpoint")
trainer.save_model(output_name + "/last_checkpoint")
tokenizer.save_pretrained(output_name + "/last_checkpoint")

if script_args.push_to_hub:
    try:
        trainer.push_to_hub()
    except Exception as e:
        print(f'Failed to push to hub: {e}')
