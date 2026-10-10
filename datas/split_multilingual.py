import json
from pathlib import Path
root=Path('datas/raw'); out=Path('datas/processed'); out.mkdir(parents=True,exist_ok=True)
train=out/'train.jsonl'; val=out/'validation.jsonl'; text=out/'all_text.txt'
paths=[root/'wiki_en/documents.jsonl',root/'wiki_fa/documents.jsonl']
with train.open('w',encoding='utf-8') as tr, val.open('w',encoding='utf-8') as va, text.open('w',encoding='utf-8') as tx:
  i=0
  for path in paths:
    if not path.exists(): continue
    for line in path.open(encoding='utf-8'):
      row=json.loads(line); tx.write(row['text']+'\n')
      (va if i%100==0 else tr).write(json.dumps(row,ensure_ascii=False)+'\n'); i+=1
print('documents',i)
