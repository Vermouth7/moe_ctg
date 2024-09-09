import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='4'

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM
from utils import *

random.seed(42)
def extract_and_store(text):
    # pattern = r'\d+\.\s*\[(.*?)\](?=\s*\d+\.|\s*$)'
    pattern = r'\[\s*(.*?)\s*\]'
    matches = re.findall(pattern, text)
    
    matches = [match.strip() for match in matches]

    result = {
        "original_text": text,
        "extracted_contents": matches
    }
    
    return result

model_name = "/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
constraint_types=['content', 'situation', 'style', 'format', 'example', 'mixed']
for constraint in constraint_types:
    feature=[]
    pool=[]
    
    
    with open("/home/chh/repos/my_ctg/instructions/followbench2/{}_constraint_split.jsonl".format(constraint), 'r', encoding='utf-8') as input_file:
        for idx,line in enumerate(tqdm(input_file, desc=f"Processing {constraint}", unit="line")):
            feature={}
            temp=json.loads(line.strip())
            res=extract_and_store(temp['split_ins'])
            
            hidden_states_list = []
            for sub_instruction in res['extracted_contents']:
                sub_instruction=prompt_template(tokenizer=tokenizer,message=sub_instruction)
                inputs = tokenizer(sub_instruction, return_tensors='pt')
                inputs.to(device)
                with torch.no_grad():
                    outputs = model(**inputs,output_hidden_states=True)
                
                hidden_states = outputs.hidden_states
                stacked_hidden_states = torch.stack([layer_output[:, -1:, :] for layer_output in hidden_states]) # 33 1 token_pos 4096
                
                # stacked_hidden_states = torch.mean(stacked_hidden_states, dim=2, keepdim=True)
                stacked_hidden_states = torch.transpose(stacked_hidden_states, 0, 1) # 1 33 1 4096
                hidden_states_list.append(stacked_hidden_states)

            hidden_states_tensor = torch.stack(hidden_states_list) # num_condi 1 33 1 4096
            hidden_states_tensor = hidden_states_tensor.squeeze(3)
            hidden_states_tensor = hidden_states_tensor.squeeze(1).to('cpu')
            feature[idx]=hidden_states_tensor
            pool.append(feature)


    for i in range(0,len(pool)):
        print(pool[i])
    save_path = "./pool/{}_constraint_split.pt".format(constraint)
    torch.save(pool, save_path)
    print(f"Task vectors saved to {save_path}")