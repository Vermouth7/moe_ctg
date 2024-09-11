import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from model import MoeModel
from peft import LoraConfig, TaskType
# import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, AutoModelForCausalLM, AutoTokenizer,
                          LlamaForCausalLM, get_linear_schedule_with_warmup)
from trl import SFTConfig, SFTTrainer
from utils import *

device = torch.device("cuda")
torch.autograd.set_detect_anomaly(True)

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = prompt_template(self.tokenizer,item["prompt"])
        completion = item["completion"]
        data_id = item["ID"] 
        input_text = f"{prompt} {completion}"
        inputs = self.tokenizer(input_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        inputs['labels'] = inputs['input_ids'].clone()

        prompt_length = len(self.tokenizer(prompt, truncation=True, max_length=self.max_length)['input_ids'])
        inputs['labels'][:, :prompt_length] = -100  

        return {key: val.squeeze(0) for key, val in inputs.items()}, data_id
    
def main(args):
    constraint_types=['content', 'situation', 'style', 'format', 'example', 'mixed']
    
    model = LlamaForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to(device)
    
    
    
    if args.stage=='train':
        wandb.init(project='moe_ctg',config={"learning_rate": args.lr,"epochs": args.epochs,"batch_size": args.batch_size})
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='right')
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        model = MoeModel(model,tokenizer)
        train_dataset = CustomDataset(args.dataset_path, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        total_steps = len(train_loader) * args.epochs  
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        vector_pool=torch.load('./pool/train_vectors.pt')
        # model.freeze_model_params()
        model.train()
        
        for epoch in range(args.epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", total=len(train_loader))  

            for step, (batch, ids) in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                pool={}
                ids=ids.tolist()
                for i in range(0,len(ids)):
                    temp=torch.stack(vector_pool[ids[i]][ids[i]])
                    pool[i]=temp
                model.set_vector_pool(pool)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                epoch_loss += loss.item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if step % 10 == 0:
                    wandb.log({"loss": loss.item(), "step": step + epoch * len(train_loader)})
                progress_bar.set_postfix(loss=loss.item())

            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")
        gate_params = {name: param for name, param in model.named_parameters() if 'gate' in name}
        torch.save(gate_params, './model_ckpt/gate_params.pth')
        wandb.finish()
    elif args.stage=='eval':
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
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
                    generation_output=model.generate(**inputs,max_new_tokens=args.max_length,use_cache=False)
                    res=tokenizer.decode(generation_output[0], skip_special_tokens=True)
                    run_results.append({'prompt_new':item['prompt_new'],'result': res})
                    print(res)
            with open(os.path.join(args.output_folder, f"{os.path.basename(args.model_path)}_{constraint}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
                for d in run_results:
                    output_file.write(json.dumps(d) + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/facebook/opt-350m')
    
    parser.add_argument('--dataset_path', type=str, default='./dataset/multi_constraints_dataset.jsonl')
    parser.add_argument('--vector_pool_path', type=str, default='./task_vectors.pt')
    parser.add_argument('--gate_model_path', type=str, default='./model_ckpt/moe_model_epoch_1.pt')
    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    parser.add_argument('--output_folder', type=str, default='./results/test2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--stage', type=str, default='eval')
    
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=8)
    
    args = parser.parse_args()
    set_seed(args)
    main(args)