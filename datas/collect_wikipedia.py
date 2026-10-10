"""Resumably collect English Wikipedia records from the HF Dataset Server."""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
API = "https://datasets-server.huggingface.co/rows"
def fetch(offset, length, timeout, language):
    params = {"dataset":"wikimedia/wikipedia", "config":f"20231101.{language}", "split":"train", "offset":offset, "length":length}
    for attempt in range(5):
        try:
            r=requests.get(API, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", "5")) + attempt * 2); continue
            r.raise_for_status(); return offset, r.json().get("rows", [])
        except Exception:
            if attempt == 4: raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch offset {offset} after retries")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--target-tokens",type=int,default=600_000_000); p.add_argument("--batch-rows",type=int,default=100); p.add_argument("--workers",type=int,default=3); p.add_argument("--timeout",type=int,default=120); p.add_argument("--language",default="en"); p.add_argument("--output-name"); a=p.parse_args()
    root=Path("datas/raw") / ("wiki_fa" if a.language == "fa" else "wiki_en"); root.mkdir(parents=True,exist_ok=True); out=root/"documents.jsonl"; pp=root/"progress.json"
    state={"next_offset":0,"accepted":0,"estimated_chars":0};
    if pp.exists(): state.update(json.loads(pp.read_text(encoding="utf-8")))
    target_chars=a.target_tokens*4; offset=state["next_offset"]
    with out.open("a",encoding="utf-8") as dst:
      while state["estimated_chars"] < target_chars:
        offsets=[offset+i*a.batch_rows for i in range(a.workers)]; results={}
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
          for job in as_completed([pool.submit(fetch,x,a.batch_rows,a.timeout,a.language) for x in offsets]):
            off,rows=job.result(); results[off]=rows
        for off in offsets:
          for item in results[off]:
            row=item.get("row",{}); text=(row.get("text") or "").strip()
            if len(text)<200: continue
            rec={"id":row.get("id"),"title":row.get("title"),"url":row.get("url"),"text":text,"source":f"Wikimedia Wikipedia 20231101 {a.language}","license":"CC BY-SA 4.0","language":a.language,"mixture_source":"wikipedia","attribution_manifest":str(root / "attribution.json")}
            dst.write(json.dumps(rec,ensure_ascii=False)+"\n"); state["accepted"]+=1; state["estimated_chars"]+=len(text)
        dst.flush(); offset += a.batch_rows*a.workers; state["next_offset"]=offset; pp.write_text(json.dumps(state,indent=2),encoding="utf-8")
        if state["accepted"]%10000 < a.workers*a.batch_rows: print(state,flush=True)
    (root/"attribution.json").write_text(json.dumps({"dataset":"wikimedia/wikipedia","config":f"20231101.{a.language}","license":"CC BY-SA 4.0","source":"https://huggingface.co/datasets/wikimedia/wikipedia","collected_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},indent=2),encoding="utf-8")
    print("collection complete",state)
if __name__=="__main__": main()
