# wrapping classes
import numpy as np
import torch
import torch.nn as nn

'''
Todo: 
1. each block attached to a gate network
2. assign correct feature vector (with sample id)
3. with a new training process
'''

class WrappedBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.output = None
        self.vector_pool = None
        self.token_pos = None
        self.sample_ids=None
        # self.gate = nn.Linear(hidden_dim, num_vector + 1).to(torch.bfloat16)
        self.gate = nn.Linear(4096, 6 + 1).to(torch.bfloat16)
        
        
    
    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if isinstance(output, tuple):
            self.output = output[0]
            modified = output[0]
        else:
            self.output = output
            modified = output
        
        ## handle the activation
        
        if isinstance(output, tuple):
            output = (modified,) + output[1:] 
        else:
            output = modified
        
        return output

    def set_pool(self, activations, token_pos=None, masks=None, normalize=False, operator='replace'):
        pass
        
    def reset(self):
        self.output = None
        self.vector_pool = None
        self.token_pos = None
        self.sample_ids=None
        



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
            self.model.model.layers[layer_id] = WrappedBlock(block)

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
