import copy
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda")
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    

def prompt_template(tokenizer,message,sys_prompt="You're an excellent assistant.") -> str:
    messages = [
    
    {"role": "user", "content": message},
    ]
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def convert_to_api_input(data_path, api_input_path, constraint_type):

    with open(os.path.join(data_path, "{}_constraints.json".format(constraint_type)), 'r', encoding='utf-8') as input_file:
        input_data = json.load(input_file)

    # check if the data format is correct
    num = 0
    for i in range(len(input_data)):
        if constraint_type != 'example':
            assert i % 6 == input_data[i]['level']
        if input_data[i]['level'] > 0:
            assert input_data[i]['instruction'] != ""
            num += 1
    print(f"\n[{constraint_type}] number of examples: {num}")

    with open(os.path.join(api_input_path, "{}_constraint.jsonl".format(constraint_type)), 'w', encoding='utf-8') as output_file:
        for d in input_data:
            if d['level'] > 0:
                output_file.write(json.dumps({'prompt_new': d['instruction'],'level':d['level']})+ "\n")
                


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = self.load_data(data_path)
        logging.warning("Formatting inputs...")
        
        self.sample_ids = [example['ID'] for example in list_data_dict]
        sources = [
            example['prompt']
            for example in list_data_dict
        ]
        targets = [f"{example['completion']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            return [json.loads(line.strip()) for line in file]
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i],sample_ids=self.sample_ids[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, sample_ids= tuple([instance[key] for instance in instances] for key in ("input_ids", "labels","sample_id"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            sample_ids=sample_ids,
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

                    
def print_block_gradients(layer_idx):
    def hook(module, grad_input, grad_output):
        if layer_idx==17:
            # print(grad_input)
            # print(grad_output)
            for name, param in module.MOE_gate.named_parameters():
                    print(name,param)
        # print('layer id',layer_idx)
        # print(grad_output)
    return hook

def register_hooks_for_gate(model):
    for layer in model.model.layers:
        layer.register_full_backward_hook(print_block_gradients(layer.layer_idx))

def save_moe_gate_params(model, output_dir):
    moe_gate_params = {name: param.cpu() for name, param in model.named_parameters() if 'MOE_gate' in name}
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "moe_gate_params.pth")
    
    torch.save(moe_gate_params, save_path)
    print(f"MOE gate parameters saved to {save_path}")
    
def load_moe_gate_params(model, load_path):
    
    moe_gate_params = torch.load(load_path)
    
    model_state_dict = model.state_dict()
    for name, param in moe_gate_params.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
        else:
            print(f"Warning: {name} not found in the model's state dict.")

    model.load_state_dict(model_state_dict, strict=False)
    print(f"MOE gate parameters loaded from {load_path}")
