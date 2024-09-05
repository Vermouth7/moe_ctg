import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='3'

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
for c in constraints:
    for i in c.keys():
        if 'Constraints' in i:
            for j in c[i]:
                for k,v in j.items():
                    feature.append(v)
# print(len(feature))
feature=random.sample(feature,5000)
for sample in tqdm(feature):
    # print(sample)
    input_text = prompt_template(tokenizer,sample)
    hidden_states = extract_hidden_states(input_text)
    
    hidden_states=torch.stack(hidden_states)[:,0,-1].to('cpu')
    task_vectors.append(hidden_states)
task_vectors=torch.stack(task_vectors)
print(task_vectors.shape)
save_path = "task_vectors.pt"
torch.save(task_vectors, save_path)
print(f"Task vectors saved to {save_path}")