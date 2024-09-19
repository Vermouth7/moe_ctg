import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from model import MoeModel
from peft import LoraConfig, TaskType
# import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, AutoModelForCausalLM, AutoTokenizer,
                          LlamaForCausalLM, LlamaForCausalLM_Moe, Trainer,
                          TrainerCallback, get_linear_schedule_with_warmup)
from trl import SFTConfig, SFTTrainer
from utils import *

import wandb

device = torch.device("cuda")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct")


@dataclass
class DataArguments:
    data_path: str = field(default='/home/chh/repos/moe_ctg/dataset/multi_constraints_dataset_5000.jsonl', metadata={"help": "Path to the training data."})
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    do_train: bool = field(
        default=True
    )
    per_device_train_batch_size: int = field(
        default=2
    )
    num_train_epochs: float = field(
        default=2
    )
    learning_rate: float = field(
        default=1e-4
    )
    seed: int = field(
        default=42
    )
    do_train: bool = field(
        default= True
    )
    do_predict: bool = field(
        default=False
    )
    output_dir: str = field(
        default='/home/chh/repos/moe_ctg/model_ckpt'
    )
    vector_pool_path: str = field(
        default='/home/chh/repos/moe_ctg/pool/train_vectors_5000.pt',
        metadata={"help": "Path to the vector pool file."}
    )
    gate_model_path: str = field(
        default='',
        metadata={"help": "Path to the saved MOE model checkpoint."}
    )
    inf_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum length for inference."}
    )
    gradient_accumulation_steps: int = field(
        default=1
    )
    remove_unused_columns:int = field(
        default=False
    )
    report_to: str = field(
        default="wandb"
    )
    
    save_strategy: str=field(
        default="no", metadata={"help": "Save model every epoch"}
    )
    logging_steps: float=field(
        default=10.0
    )
    
    bf16: bool=field(
        default=True,
    )
    gradient_checkpointing:bool=field(
        default=False
    )
def main(model_args, data_args, training_args):
    constraint_types=['content', 'situation', 'style', 'format', 'example', 'mixed']
        
    if training_args.do_train:
        wandb.init(project='moe_ctg',config={"learning_rate": training_args.learning_rate,"epochs": training_args.num_train_epochs,"total batch size": training_args.per_device_train_batch_size*training_args._n_gpu})
        vector_pool=torch.load(training_args.vector_pool_path)
        
        model = LlamaForCausalLM_Moe.from_pretrained(model_args.model_name_or_path,torch_dtype=torch.bfloat16).to(device)
        model.init_pool(vector_pool)
        # optimizer = AdamW(params=[p for name, p in model.named_parameters() if 'model.layers.20.MOE_gate' in name], lr=training_args.learning_rate,weight_decay=1e-5)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=training_args.max_steps)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,padding_side='right')
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        
        
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        
        trainer.train()
        
        save_moe_gate_params(model, training_args.output_dir)
        wandb.finish()
    elif training_args.do_predict=='eval':
        model = LlamaForCausalLM.from_pretrained(model_args.model_name_or_path,torch_dtype=torch.bfloat16).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_path,padding_side='left')
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        model = MoeModel(model,tokenizer)
        model.eval()
        for constraint in constraint_types:
            vector_pool=torch.load('./pool/{}_constraint_split.pt'.format(constraint))
            
            run_results = []
            test_data=[]
            with open(os.path.join("/home/chh/repos/my_ctg/instructions/followbench2/{}_constraint.jsonl".format(constraint)), 'r', encoding='utf-8') as test_file:
                for idx,line in enumerate(tqdm(test_file, desc=f"Processing {constraint}", unit="line")):
                    model.set_vector_pool(vector_pool[idx][idx])
                    temp_dict = json.loads(line.strip())
                    prompt=temp_dict['prompt_new']
                    prompt_tem=prompt_template(tokenizer,prompt)
                    test_data.append({'prompt_new':prompt,'prompt_input':prompt_tem})
                
                for idx,item in enumerate(tqdm(test_data, desc=f"Processing {constraint}")):
                    inputs = tokenizer(item['prompt_input'], return_tensors="pt").to(device)
                    generation_output=model.generate(**inputs,max_new_tokens=training_args.inf_max_length,use_cache=False)
                    res=tokenizer.decode(generation_output[0], skip_special_tokens=True)
                    run_results.append({'prompt_new':item['prompt_new'],'result': res})
                    print(res)
            with open(os.path.join(training_args.output_dir, f"{os.path.basename(model_args.model_name_or_path)}_{constraint}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
                for d in run_results:
                    output_file.write(json.dumps(d) + "\n")
    

if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    
    set_seed(training_args)
    main(model_args,data_args,training_args)