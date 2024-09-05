import argparse
import json
import os
import random
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import spacy
import torch
from googleapiclient import discovery
from scipy.special import expit, softmax
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda")


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