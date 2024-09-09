# wrapping classes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

class WrappedBlock(torch.nn.Module):
    def __init__(self, block,layer_idx):
        super().__init__()
        self.block = block
        self.output = None
        self.vector_pool = None
        self.token_pos = -1
        self.sample_ids=None
        # self.gate = nn.Linear(hidden_dim, num_vector + 1).to(torch.bfloat16)
        self.gate = nn.Linear(4096, 6 + 1).to(device).to(torch.bfloat16)
        self.layer_idx=layer_idx
        
    
    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if isinstance(output, tuple):
            self.output = output[0]
            modified = output[0]
        else:
            self.output = output
            modified = output
        
        ## handle the activation
        batch_size, seq_len, hidden_dim = modified.size()
        
        token_activation = modified[:, self.token_pos, :].to(modified.device)  # (batch_size, hidden_dim)
        gate_scores = self.gate(token_activation)
        gate_probs = F.softmax(gate_scores, dim=-1)

        for b in range(batch_size):
            vectors_from_pool = self.vector_pool[b][:, self.layer_idx]  # Original vector pool

            # Handle vector_pool size
            if vectors_from_pool.size(0) < 6:
                # If vector_pool size is less than 6, pad with zeros
                pad_size = 6 - vectors_from_pool.size(0)
                padding = torch.zeros((pad_size, hidden_dim), device=vectors_from_pool.device, dtype=vectors_from_pool.dtype)
                vectors_from_pool = torch.cat((vectors_from_pool, padding), dim=0)
            elif vectors_from_pool.size(0) > 6:
                # If vector_pool size is greater than 6, randomly select 6 vectors
                perm = torch.randperm(vectors_from_pool.size(0))[:6]
                vectors_from_pool = vectors_from_pool[perm]

            # Combine vectors: original token_activation goes to the last position
            combined_vector = torch.cat((vectors_from_pool, token_activation[b].unsqueeze(0)), dim=0)  # (7, hidden_dim)

            # Apply soft gate to combine original and pool vectors
            new_vector = torch.matmul(gate_probs[b], combined_vector)  # (hidden_dim,)

            # Replace the activation with the new soft-gated vector
            modified[b, self.token_pos, :] = new_vector  
                    
        self.token_pos-=1
        if isinstance(output, tuple):
            output = (modified,) + output[1:] 
        else:
            output = modified
        
        return output

    def set_pool(self,pool):
        self.vector_pool=pool
        
    def reset(self):
        self.output = None
        self.vector_pool = None
        self.token_pos = -1
        self.sample_ids=None
        self.layer_idx=None
        
    def reset_token_pos(self):
        self.token_pos=-1



BLOCK_NAMES = [
    "self_attn",
    "mlp",
    "input_layernorm",
    "post_attention_layernorm"
    ]
    
class MoeModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.wrap_all_decoder()
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def generate(self, **kwargs):
        for layer_id, layer in enumerate(self.model.model.layers):
            layer.reset_token_pos()
        return self.model.generate(**kwargs)
        
    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.model.device)).logits
            return logits
        
    def run_prompt(self, prompt, **kwargs):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)
            output = self.model(input_ids, attention_mask=attention_mask)
            return output

    def wrap(self, layer_id, block_name):
        assert block_name in BLOCK_NAMES
        if self.is_wrapped(self.model.model.layers[layer_id]):
            block = getattr(self.model.model.layers[layer_id].block, block_name)
            if not self.is_wrapped(block):
                setattr(self.model.model.layers[layer_id].block, block_name, WrappedBlock(block))
        else:
            block = getattr(self.model.model.layers[layer_id], block_name)
            if not self.is_wrapped(block):
                setattr(self.model.model.layers[layer_id], block_name, WrappedBlock(block))

    def wrap_decoder_block(self, layer_id):
        block = self.model.model.layers[layer_id]
        if not self.is_wrapped(block):
            self.model.model.layers[layer_id] = WrappedBlock(block,layer_id+1)

    def wrap_all_decoder(self):
        for layer_id, layer in enumerate(self.model.model.layers):
            # for block_name in BLOCK_NAMES:
            #     self.wrap(layer_id, block_name)
            self.wrap_decoder_block(layer_id)
            
    def wrap_block(self, layer_ids, block_name):
        def _wrap_block(layer_id, block_name):
            if block_name in BLOCK_NAMES:
                self.wrap(layer_id, block_name)
            elif block_name == 'decoder_block':
                self.wrap_decoder_block(layer_id)
            else:
                assert False, f"No block named {block_name}."

        if isinstance(layer_ids, list) or isinstance(layer_ids, tuple) or isinstance(layer_ids, np.ndarray):
            for layer_id in layer_ids:
                _wrap_block(layer_id, block_name)
        else:
            _wrap_block(layer_ids, block_name)
        
    def reset(self):
        for layer in self.model.model.layers:
            if self.is_wrapped(layer):
                layer.reset()
                for block_name in BLOCK_NAMES:
                    if self.is_wrapped(getattr(layer.block, block_name)):
                        getattr(layer.block, block_name).reset()
            else:
                for block_name in BLOCK_NAMES:
                    if self.is_wrapped(getattr(layer, block_name)):
                        getattr(layer, block_name).reset()
    

    def is_wrapped(self, block):
        if hasattr(block, 'block'):
            return True
        return False
    
    def unwrap(self):
        for l, layer in enumerate(self.model.model.layers):
            if self.is_wrapped(layer):
                self.model.model.layers[l] = layer.block
            for block_name in BLOCK_NAMES:
                if self.is_wrapped(getattr(self.model.model.layers[l], block_name)):
                    setattr(self.model.model.layers[l],
                            block_name,
                            getattr(self.model.model.layers[l], block_name).block)

    def set_vector_pool(self,vector):
        for layer_id, layer in enumerate(self.model.model.layers):
            layer.set_pool(vector)
            layer.reset_token_pos()
    
    def freeze_model_params(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False  

        for layer in self.model.model.layers:
            if isinstance(layer, WrappedBlock):
                for name, param in layer.gate.named_parameters():
                    param.requires_grad = True  