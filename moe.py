import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='7'
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer
from utils import *

device = torch.device("cuda")

def save_moe_model(model, path):
    moe_state_dict = {f"moe_layer_{i}": layer.state_dict() for i, layer in enumerate(model.moe_layers)}
    torch.save(moe_state_dict, path)
    print(f"Model saved to {path}")

def load_moe_model(model, path):
    moe_state_dict = torch.load(path)
    for i, layer in enumerate(model.moe_layers):
        layer.load_state_dict(moe_state_dict[f"moe_layer_{i}"])
    print(f"Model loaded from {path}")

class MoEFeedForward(nn.Module):
    def __init__(self, hidden_dim,layer_idx, num_vector=5000,task_vectors=None,layers=33):
        super(MoEFeedForward, self).__init__()
        self.num_vector = num_vector
        self.num_layers = layers
        self.layer_idx = layer_idx
        
        if task_vectors is not None:
            self.vector_pool = task_vectors
        else:
            self.vector_pool = torch.randn(num_vector, layers,hidden_dim, dtype=torch.bfloat16)
        
        self.gate = nn.Linear(hidden_dim, num_vector + 1).to(torch.bfloat16)
    
    def forward(self, x):
        if isinstance(x, tuple):
            activation=x[0]
        batch_size, seq_len, hidden_dim = activation.size()
        
        output = torch.zeros_like(activation)

        for i in range(seq_len):
            token_activation = activation[:, i, :]  # (batch_size, hidden_dim)

            # Pass the token activation through the gate network
            gate_scores = self.gate(token_activation)  # (batch_size, num_vector + 1)
            gate_probs = F.softmax(gate_scores, dim=-1)  # Convert to probabilities

            # Extract the max probability index for each batch item
            max_indices = torch.argmax(gate_probs, dim=-1)  # (batch_size,)

            # Replace or keep the original activation
            for b in range(batch_size):
                if max_indices[b] == self.num_vector:  # If the max index is the extra class (no replacement)
                    output[b, i, :] = activation[b, i, :]  # Keep the original activation
                else:
                    # Replace with the corresponding task vector from the vector pool
                    replacement_vector = self.vector_pool[max_indices[b],self.layer_idx]  # (hidden_dim,)
                    # print(replacement_vector.shape)
                    output[b, i, :] = replacement_vector  # Weighted sum of original and replacement
        if isinstance(x, tuple):
            x = (output,) + x[1:] 
        else:
            x = output
        # print(output.requires_grad)
        return x

class LLaMAWithMoE(nn.Module):
    def __init__(self, model_name, num_vector=5000,vector_pool_path=None):
        super(LLaMAWithMoE, self).__init__()
        self.llama = LlamaForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16).to(device)
        self.num_layers = self.llama.config.num_hidden_layers
        if vector_pool_path:
            self.vector_pool = torch.load(vector_pool_path).to(device) # extract last layer 
            assert self.vector_pool.size(0) == num_vector, "Number of vectors in pool must match num_vector"
        else:
            self.vector_pool = None
        self.moe_layers = nn.ModuleList([
            MoEFeedForward(self.llama.config.hidden_size, i,num_vector,self.vector_pool,self.num_layers).to(device) for i in range(1,self.num_layers+1)
        ])
        self.hooks = []
        
        for i, layer in enumerate(self.llama.model.layers):  
            hook = layer.register_forward_hook(self._create_hook(self.moe_layers[i]))
            self.hooks.append(hook)
        
    def _create_hook(self, moe_layer):
        def hook(module, input, output):
            output = moe_layer(output)
            # Ensure gradients are tracked for the MoE layer's output
            
            return output
        return hook
    
    def generate(self, **kwargs):
        return self.llama.generate(**kwargs)
    
    def forward(self, input_ids):
        input_ids=input_ids.to(device)
        outputs = self.llama(input_ids, output_hidden_states=True)
        outputs.logits.requires_grad_()
        return outputs  # Model outputs after MoE processing in hooks
    
    def freeze_llm_parameters(self):
        for param in self.llama.parameters():
            param.requires_grad = False


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = prompt_template(self.tokenizer,item['New instruction'])
        output = item['Output']

        # Tokenize input and output using the tokenizer
        input_encodings = self.tokenizer(instruction, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        output_encodings = self.tokenizer(output, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')


        input_ids = input_encodings['input_ids'].squeeze(0)  # Shape: (seq_len,)
        target_ids = output_encodings['input_ids'].squeeze(0)  # Shape: (seq_len,)

        return {
            'prompt': input_ids,
            'completion': target_ids
        }
    
def main(args):
    
    model = LLaMAWithMoE(args.model_path, vector_pool_path=args.vector_pool_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    # for name, param in model.moe_layers.named_parameters():
    #     print(f"{name} requires_grad: {param.requires_grad}")
    # input_ids = torch.randint(0, 10000, (1, 50))  
    # output=model(input_ids)
    
    
    if args.stage=='train':
        # wandb.init(project="moe_ctg")
        
        train_dataset = InstructionDataset('./dataset/multi_constraints.json', tokenizer)

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
        model.freeze_llm_parameters()
        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            args=sft_config,
            tokenizer=tokenizer
        )
        trainer.train()

        
        # for epoch in range(args.epochs):
        #     model.train()
        #     running_loss = 0.0
        #     progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        #     for batch in progress_bar:
        #         input_ids = batch['input_ids'].to(device)
        #         target_ids = batch['target_ids'].to(device)
        #         optimizer.zero_grad()
        #         # print(input_ids.requires_grad)
        #         # Forward pass
        #         outputs = model(input_ids)
        #         logits = outputs.logits  # Shape: [batch_size * seq_len, vocab_size]
        #         logits = logits.view(-1, logits.shape[-1])  # Shape: [batch_size * seq_len, vocab_size]
        #         # print(f"Logits shape: {logits.shape}")
        #         # print(f"Logits requires_grad: {logits.requires_grad}")
                
        #         # Flatten target_ids to [batch_size * seq_len]
        #         target_flat = target_ids.view(-1)  # Shape: [batch_size * seq_len]
        #         # print(f"Target shape: {target_flat.shape}")
        #         loss = F.cross_entropy(logits, target_flat)

        #         # Backward pass
        #         loss.backward()

        #         # Update parameters
        #         optimizer.step()

        #         # Log loss to wandb
        #         wandb.log({"loss": loss.item()})

        #         # Update progress bar with loss
        #         running_loss += loss.item()
        #         progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        #     # Save the model after each epoch
        #     save_moe_model(model, f"./model_ckpt/moe_model_epoch_{epoch+1}.pt")
    elif args.stage=='eval':
        constraint_types=['content', 'situation', 'style', 'format', 'example', 'mixed']
        
        load_moe_model(model, args.gate_model_path)
        model.eval()
        for constraint in constraint_types:
            run_results = []
            test_data=[]
            with open(os.path.join("/home/chh/repos/my_ctg/instructions/followbench2/{}_constraint.jsonl".format(constraint)), 'r', encoding='utf-8') as test_file:
                for line in test_file:
                    temp_dict = json.loads(line.strip())
                    prompt=temp_dict['prompt_new']
                    prompt_tem=prompt_template(tokenizer,prompt)
                    test_data.append({'prompt_new':prompt,'prompt_input':prompt_tem})
            inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt").to(device)
            generation_output = model.generate(**inputs,max_new_tokens=args.max_length)
            print(tokenizer.decode(generation_output[0], skip_special_tokens=True))
    else:
        NotImplemented

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data1/chh/models/meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--vector_pool_path', type=str, default='./task_vectors.pt')
    parser.add_argument('--gate_model_path', type=str, default='./model_ckpt/moe_model_epoch_1.pt')
    # parser.add_argument('--model_path', type=str, default='/data1/chh/models/model_sft/llama3-8b/merge/qwen/sft3')
    parser.add_argument('--output_folder', type=str, default='./results/test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--stage', type=str, default='train')
    
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()
    set_seed(args)
    main(args)