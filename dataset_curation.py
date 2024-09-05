import json
import logging
import random

from datasets import load_dataset
from openai import OpenAI

random.seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE = "https://api.nextapi.fun"
API_KEY = "ak-Lfu504S5OrzjNvivYYdY6E8xvn1hiTY42texx7WvTIojB9MC"
MAX_API_RETRY = 5
template=None
def api_gpt(user_prompt: str, api_base:str,api_key: str):
    logging.basicConfig(level=logging.INFO)
    for i in range(MAX_API_RETRY):
        try:
            client = OpenAI(base_url=api_base,api_key=api_key)
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{
                    'role': 'user',
                    'content': user_prompt,
                }],
            )
            content = response.choices[0].message.content
            # logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
    logger.error(f'Failed after {MAX_API_RETRY} retries.')
    return 'error'
with open('./dataset/con_template.txt','r',encoding='utf-8') as f:
    template=f.read()

ds = load_dataset("/data1/chh/datasets/yahma/alpaca-cleaned",split='train')
ds= random.sample(list(ds), 2000)
new_data=[]

for i in ds:
    input_string=i['instruction']+i['input']
    output_string=i['output']
    # print(template%(input_string,output_string))
    
    content=api_gpt(user_prompt=template%(input_string,output_string),api_base=API_BASE,api_key=API_KEY)
    if content:
        try:
            if content.startswith("```json"): # remove markdown, used for gpt-4 turbo
                content = content[7:-3].strip()
            answer = json.loads(content)
        except Exception as e:
                print(f"json failed to parse: {e}")
                print(f"content: {content}")
        if answer:
            answer['Original instruction']=input_string
            answer['Output']=output_string
            new_data.append(answer)
    
    if len(new_data) >= 1000:
        break
    
with open("./dataset/multi_constraints.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print(f"save {len(new_data)} records")