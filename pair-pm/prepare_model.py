import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

name = 'meta-llama/Meta-Llama-3-8B-Instruct'
tokenizer_name = name

model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.save_pretrained(os.environ['LLAMA_MODEL_SAVE_PATH'])
tokenizer.save_pretrained(os.environ['LLAMA_MODEL_SAVE_PATH'])

