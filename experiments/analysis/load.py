import json, pandas as pd
import os
D=os.environ.get("M3KANG_DATA","data/").rstrip("/")+"/"
def load():
    rows=[]
    for s in ("train","val"):
        for l in open(D+s+".jsonl", encoding="utf-8"):
            r=json.loads(l); r["split"]=s; rows.append(r)
    df=pd.DataFrame(rows)
    df["has_opt"]=df.options.apply(lambda o: len(o)==5)
    return df
