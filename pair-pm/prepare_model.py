import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import yaml

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
model.resize_token_embeddings(len(tokenizer))

model_save_path = os.environ['LLAMA_MODEL_SAVE_PATH']
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

config_template_path = sys.argv[1]
with open(config_template_path, 'r') as f:
    config = yaml.safe_load(f)

config['base_model'] = model_save_path
config['output_dir'] = os.environ['LLAMA_MODEL_OUTPUT_PATH']
config['dataset_prepared_path'] = os.environ['LLAMA_MODEL_PREPARE_PATH']

with open(config_template_path, 'w') as f:
    yaml.dump(config, f)
