import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig
# import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer
from utils import *

device = torch.device("cuda")

def load_moe_model(model, path):
    moe_state_dict = torch.load(path)
    for i, layer in enumerate(model.moe_layers):
        layer.load_state_dict(moe_state_dict[f"moe_layer_{i}"])
    print(f"Model loaded from {path}")

class MoEFeedForward(nn.Module):
    def __init__(self, hidden_dim,layer_idx, num_vector=8,task_vectors=None,layers=33):
        super(MoEFeedForward, self).__init__()
        self.num_vector = num_vector
        self.num_layers = layers
        self.layer_idx = layer_idx
        
        if task_vectors is not None:
            self.vector_pool = task_vectors
        else:
            self.vector_pool = torch.randn(num_vector, layers,hidden_dim, dtype=torch.bfloat16)
        
        self.gate = nn.Linear(hidden_dim, num_vector + 1).to(torch.bfloat16)
        self.token_pos=-1
    def forward(self, x):
        if isinstance(x, tuple):
            activation=x[0]
        batch_size, seq_len, hidden_dim = activation.size()
        # print(seq_len)

        # for i in range(seq_len):
        #     token_activation = activation[:, i, :]  # (batch_size, hidden_dim)

        #     # Pass the token activation through the gate network
        #     gate_scores = self.gate(token_activation)  # (batch_size, num_vector + 1)
        #     gate_probs = F.softmax(gate_scores, dim=-1)  # Convert to probabilities

        #     # Extract the max probability index for each batch item
        #     max_indices = torch.argmax(gate_probs, dim=-1)  # (batch_size,)

        #     # Replace or keep the original activation
        #     for b in range(batch_size):
        #         if max_indices[b] == self.num_vector:  # If the max index is the extra class (no replacement)
        #             output[b, i, :] = activation[b, i, :]  # Keep the original activation
        #         else:
        #             # Replace with the corresponding task vector from the vector pool
        #             replacement_vector = self.vector_pool[max_indices[b],self.layer_idx]  # (hidden_dim,)
        #             # print(replacement_vector.shape)
        #             output[b, i, :] = replacement_vector  # Weighted sum of original and replacement
        token_activation = activation[:, self.token_pos, :]  # (batch_size, hidden_dim)
        gate_scores = self.gate(token_activation)  # (batch_size, num_vector + 1)
        gate_probs = F.softmax(gate_scores, dim=-1)  # Convert to probabilities
        max_indices = torch.argmax(gate_probs, dim=-1)  # (batch_size,)
        # max_indices[0]=7
        for b in range(batch_size):
            if max_indices[b] == self.num_vector:  # If the max index is the extra class (no replacement)
                activation[b, self.token_pos, :] = activation[b, self.token_pos, :]  # Keep the original activation
            else:
                # print('test')
                # Replace with the corresponding task vector from the vector pool
                replacement_vector = self.vector_pool[max_indices[b],self.layer_idx]  # (hidden_dim,)
                # print(replacement_vector.shape)
                activation[b, self.token_pos, :] = replacement_vector  # Weighted sum of original and replacement
            
        # if(self.layer_idx==32):
        # self.token_pos-=1
        if isinstance(x, tuple):
            x = (activation,) + x[1:]
        else:
            x = activation
        # print(output.requires_grad)
        return x

class LLaMAWithMoE(nn.Module):
    def __init__(self, model_name, num_vector=8,vector_pool_path=None):
        super(LLaMAWithMoE, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16).to(device)
        self.num_layers = self.model.config.num_hidden_layers
        if vector_pool_path:
            self.vector_pool = torch.load(vector_pool_path)[:8].to(device) # extract last layer 
            assert self.vector_pool.size(0) == num_vector, "Number of vectors in pool must match num_vector"
        else:
            self.vector_pool = None
        self.moe_layers = nn.ModuleList([
            MoEFeedForward(self.model.config.hidden_size, i,num_vector,self.vector_pool,self.num_layers,).to(device) for i in range(1,self.num_layers+1)
        ])
        self.hooks = []
        
        for i, layer in enumerate(self.model.model.layers):  
            hook = layer.register_forward_hook(self._create_hook(self.moe_layers[i]))
            self.hooks.append(hook)
        
    def _create_hook(self, moe_layer):
        def hook(module, input, output):
            output = moe_layer(output)
            
            return output
        return hook
    
    def generate(self, **kwargs):

        return self.model.generate(**kwargs,use_cache=False)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels,use_cache=False)
        print(input_ids.shape)
        # outputs.logits.requires_grad_()
        # print(f"Logits requires_grad: {outputs.logits.requires_grad}")
        # print(f"Logits grad_fn: {outputs.logits.grad_fn}")
        # print(f"Loss grad_fn: {outputs.loss.grad_fn}")
        return outputs  
    
    def freeze_llm_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False


def main(args):
    
    model = LLaMAWithMoE(args.model_path, vector_pool_path=args.vector_pool_path)
    # model=AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16).to(device)
    
    
    if args.stage=='train':
        # wandb.init(project="moe_ctg")
        train_dataset=load_dataset('json',data_files='./dataset/multi_constraints_dataset.jsonl',split='train')
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='right')
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        
        moe_params = [param for name, param in model.named_parameters() if 'moe_layers' in name and param.requires_grad]
        optimizer = torch.optim.AdamW(moe_params, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=50)
        
        sft_config = SFTConfig(
            dataset_text_field="prompt",
            max_seq_length=512,
            output_dir='./model_ckpt',
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            report_to=None,
            logging_steps=5,
        )
        
        # model.freeze_llm_parameters()
        # for name, param in model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            args=sft_config,
            tokenizer=tokenizer,
            optimizers=(optimizer,scheduler)
        )
        trainer.train()
        
    elif args.stage=='eval':
        constraint_types=['content', 'situation', 'style', 'format', 'example', 'mixed']
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        
        # load_moe_model(model, args.gate_model_path)
        model.eval()
        # for constraint in constraint_types:
        #     run_results = []
        #     test_data=[]
        #     with open(os.path.join("/home/chh/repos/my_ctg/instructions/followbench2/{}_constraint.jsonl".format(constraint)), 'r', encoding='utf-8') as test_file:
        #         for line in test_file:
        #             temp_dict = json.loads(line.strip())
        #             prompt=temp_dict['prompt_new']
        #             prompt_tem=prompt_template(tokenizer,prompt)
        #             test_data.append({'prompt_new':prompt,'prompt_input':prompt_tem})
        inputs = tokenizer("Identify one category from the list below for the input text, and also infer the sentiment (positive, neutral, or negative) conveyed in the text. Your options for the category are - company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film, or written work. Michael DenDekker - Michael G. DenDekker (born July 11 1961) is an assemblyman for the state of New York's 34th district which includes the neighborhoods of Woodside, Jackson Heights, and East Elmhurst, all in the borough/county of Queens. ", return_tensors="pt").to(device)
        generation_output = model.generate(**inputs,max_new_tokens=args.max_length)
        print(tokenizer.decode(generation_output[0], skip_special_tokens=True))
    else:
        NotImplemented

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