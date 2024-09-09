import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='4'
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from model import MoeModel
from peft import LoraConfig
# import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer
from utils import *

device = torch.device("cuda")



def main(args):
    constraint_types=['content', 'situation', 'style', 'format', 'example', 'mixed']
    
    model = LlamaForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='right')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    model = MoeModel(model,tokenizer)
    if args.stage=='eval':
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
                    generation_output=model.generate(**inputs,max_new_tokens=args.max_length,use_cache=False)
                    res=tokenizer.decode(generation_output[0], skip_special_tokens=True)
                    run_results.append({'prompt_new':item['prompt_new'],'result': res})
        
            with open(os.path.join(args.output_folder, f"{os.path.basename(args.model_path)}_{constraint}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
                for d in run_results:
                    output_file.write(json.dumps(d) + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/facebook/opt-350m')
    
    parser.add_argument('--vector_pool_path', type=str, default='./task_vectors.pt')
    parser.add_argument('--gate_model_path', type=str, default='./model_ckpt/moe_model_epoch_1.pt')
    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    parser.add_argument('--output_folder', type=str, default='./results/test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--stage', type=str, default='eval')
    
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    
    args = parser.parse_args()
    set_seed(args)
    main(args)