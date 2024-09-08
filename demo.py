import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
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
    
    model = LlamaForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='right')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    model = MoeModel(model,tokenizer)
    # model=AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to(device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/facebook/opt-350m')
    
    parser.add_argument('--vector_pool_path', type=str, default='./task_vectors.pt')
    parser.add_argument('--gate_model_path', type=str, default='./model_ckpt/moe_model_epoch_1.pt')
    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    parser.add_argument('--output_folder', type=str, default='./results/test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--stage', type=str, default='train')
    
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    
    args = parser.parse_args()
    set_seed(args)
    main(args)