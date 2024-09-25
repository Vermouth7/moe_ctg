import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='3'
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from model import MoeModel
from transformers import (AdamW, AutoModelForCausalLM, AutoTokenizer,
                          LlamaForCausalLM, LlamaForCausalLM_Moe,
                          Qwen2ForCausalLM_Moe)
from utils import *

device = torch.device("cuda")

def main(args):
    constraint_types=['content', 'situation','style', 'format', 'example', 'mixed']
    
    model = LlamaForCausalLM_Moe.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16).to(device)
            
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    # model = MoeModel(model,tokenizer)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    for constraint in constraint_types:
        vector_pool=torch.load('/home/chh/repos/moe_ctg/pool/llama3/{}_constraint_split.pt'.format(constraint))
        
        run_results = []
        test_data=[]
        with open(os.path.join("/home/chh/repos/my_ctg/instructions/followbench2/{}_constraint.jsonl".format(constraint)), 'r', encoding='utf-8') as test_file:
            for idx,line in enumerate(tqdm(test_file, desc=f"Processing {constraint}", unit="line")):
                temp_dict = json.loads(line.strip())
                prompt=temp_dict['prompt_new']
                prompt_tem=prompt_template(tokenizer,prompt)
                test_data.append({'prompt_new':prompt,'prompt_input':prompt_tem})
            
            for batch_idx, i in enumerate(tqdm(range(0, len(test_data), args.batch_size), desc=f"Processing {constraint} in Batches")):
                batch_data = test_data[i:i+args.batch_size]  
                sample_ids = list(range(i, i+len(batch_data)))

                batch_prompts = [item['prompt_input'] for item in batch_data]
                pool={}
                if isinstance(sample_ids,list):
                    for i in range(0,len(sample_ids)):
                        pool[i]=vector_pool[sample_ids[i]]
                model.set_vector_pool_eval(pool)
                
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                model.set_pos_eval(inputs.input_ids)
                generation_output = model.generate(**inputs, max_new_tokens=args.max_length, use_cache=False)
                # print(generation_output)
                input_length = inputs['input_ids'].shape[1]
                for output,item in zip(generation_output,batch_data):
                    new_tokens = output[input_length:]  
                    res = tokenizer.decode(new_tokens, skip_special_tokens=True)  
                    run_results.append({'prompt_new':item['prompt_new'],'result': res})
                
        with open(os.path.join(args.output_dir, f"{os.path.basename(args.model_name_or_path)}_{constraint}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
            for d in run_results:
                output_file.write(json.dumps(d) + "\n")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    # parser.add_argument("--model_name_or_path", default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct', type=str)
    parser.add_argument("--model_name_or_path", default='/data1/chh/models/model_sft/llama3-8b/merge/moe1', type=str)
    
    # parser.add_argument("--model_name_or_path", default='/data1/chh/models/model_sft/qwen2/moe1', type=str)
    parser.add_argument('--output_dir', type=str, default='./results/res2')
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--max_length", default=1024, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    
    args = parser.parse_args()
    set_seed(args)
    
    main(args)
    
    