import torch
torch.cuda.empty_cache()
import numpy as np
import pandas as pd
import datasets
import os
import argparse
# disable annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer

from utils import *

def tokenization_strategy(entry):
    return tokenizer(entry['utterance'], max_length=512, truncation=True)

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

args = {
    'device': activate_gpu(),
    'rank': 16
}

clm_model = AutoModelForCausalLM.from_pretrained(  
    model_name,
    low_cpu_mem_usage=True,         # recommanded param
    return_dict=True,               # not used for now
    torch_dtype=torch.bfloat16,     # bfloat instead of float because it may help
    device_map=args["device"],      # send to the right device (balanced_low_GPUindex will prioritiez GPUindex GPU over the other available GPUs)
)

config = LoraConfig(
        r=args['rank'],                       # rank of lora module
        lora_alpha=(1/2)*args['rank'],        # resclaling weights parameters (alpha=1/4*r, alpha=2r, alpha=r, alpha=4r, alpha=1/2*r)
        target_modules=["q_proj", "v_proj"],  # use ["qkv_proj"] for phi models and ["q_proj", "v_proj"] for mistral, llama and qwen
        bias="lora_only",                     # should be better than default setting in our case
        lora_dropout=0.1,                     # conventional setting
    )

model = get_peft_model(clm_model, config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset('emo_label_wrapped')


lm_dataset = dataset.map(
    tokenization_strategy,
    batched=True,
    num_proc=4,
)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="mistral_emo",
    eval_strategy="epoch",
    learning_rate=5e-4, # value obtained from hyperparameter tuning
    weight_decay=0.01,
    num_train_epochs=20,
    seed=42,
    eval_accumulation_steps=1,
    prediction_loss_only=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    data_collator=data_collator,
)

trainer.train()
