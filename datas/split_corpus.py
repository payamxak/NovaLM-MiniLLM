import json
from pathlib import Path
src=Path('datas/raw/wiki_en/documents.jsonl')
train=Path('datas/processed/train.jsonl'); val=Path('datas/processed/validation.jsonl'); text=Path('datas/processed/all_text.txt')
train.parent.mkdir(parents=True,exist_ok=True)
if src.exists():
 with src.open(encoding='utf-8') as s, train.open('w',encoding='utf-8') as tr, val.open('w',encoding='utf-8') as va, text.open('w',encoding='utf-8') as tx:
  for i,line in enumerate(s):
   row=json.loads(line); tx.write(row['text']+'\n')
   (va if i%100==0 else tr).write(json.dumps(row,ensure_ascii=False)+'\n')
 print('split complete')
else: print('collector output not present; run collect_wikipedia.py first')
