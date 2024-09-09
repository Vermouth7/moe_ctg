import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='7'

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM
from utils import *

random.seed(42)

model_name = "/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

num_vectors = 5000
task_vectors = []

def extract_hidden_states(input_text):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get hidden states from all layers
    hidden_states = outputs.hidden_states  
    return hidden_states

constraints=None
feature=[]
with open('./dataset/multi_constraints.json','r') as f:
    constraints=json.load(f)
min_con=1000
for c in range(0,len(constraints)):
    temp={}
    temp[c]=[]
    for i in constraints[c].keys():
        if 'Constraints' in i:
            for j in constraints[c][i]:
                for k,v in j.items():
                    temp[c].append(v)
    feature.append(temp)
# print(feature)
# print(len(feature))
pool=[]
for sample in tqdm(feature[:100]):
    # print(sample)
    temp={}
    for k,v in sample.items():
        temp[k]=[]
        for i in v:
            input_text = prompt_template(tokenizer,i)
            hidden_states = extract_hidden_states(input_text)
            hidden_states=torch.stack(hidden_states)[:,0,-1].to('cpu')
            temp[k].append(hidden_states)
    
    pool.append(temp)

for i in range(0,len(pool)):
    print(pool[i])
save_path = "task_vectors.pt"
torch.save(pool, save_path)
print(f"Task vectors saved to {save_path}")