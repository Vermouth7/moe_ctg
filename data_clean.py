import json

data_path='./dataset/multi_constraints.json'
with open(data_path, 'r') as f:
    data = json.load(f)
new_data={}
train=[]
for idx,i in enumerate(data):
    temp={}
    temp['ID']=idx
    temp['prompt']=i['New instruction']
    temp['completion']=i['Output']
    train.append(temp)
with open("./dataset/multi_constraints_dataset.jsonl", "w", encoding="utf-8") as f:
    for i in train:
        f.write(json.dumps(i,ensure_ascii=False)+'\n')