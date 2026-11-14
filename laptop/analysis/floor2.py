import re, numpy as np, pandas as pd
from load import load
from difflib import SequenceMatcher
L='ABCDE'
NUMRE=r'-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?'
def tonum(t):
    t=t.replace(',','.').replace(' ','')
    if '/' in t:
        a,b=t.split('/')
        return float(a)/float(b) if float(b)!=0 else None
    return float(t)
def option_values(o):
    """Numeric value per option, robust to stray OCR glyphs: if an option has several
    numbers, keep the one whose magnitude best matches the other options."""
    cands=[[tonum(x) for x in re.findall(NUMRE,o[k].replace('−','-'))] for k in L]
    if any(len(c)==0 for c in cands): return None
    single=[c[0] for c in cands if len(c)==1]
    ref=np.median(single) if single else None
    vals=[]
    for c in cands:
        if len(c)==1 or ref is None: vals.append(c[-1])
        else: vals.append(min(c,key=lambda x: abs(np.log10(abs(x)+1)-np.log10(abs(ref)+1))))
    return vals if len(set(vals))==5 else None
def medoid(strs):
    d=[[1-SequenceMatcher(None,a,b).ratio() for b in strs] for a in strs]
    s=[sum(r) for r in d]; return min(range(5),key=lambda i:(s[i],i))
def predict(r):
    o=r['options']
    if len(o)!=5: return 'D','empty'
    v=option_values(o)
    if v:
        order=sorted(range(5),key=lambda i:v[i]); return L[order[2]],'numeric'
    return L[medoid([o[k] for k in L])],'text_opts'
if __name__=='__main__':
    df=load()
    for lang in ['eng','cat']:
        e=df[df.language==lang].copy()
        e[['pred','route']]=e.apply(lambda r: pd.Series(predict(r)),axis=1)
        e['ok']=e.pred==e.label
        print(lang,'overall',e.ok.mean().round(3),' val only',e[e.split=='val'].ok.mean().round(3)); print(e.groupby('route').ok.agg(['mean','size']).round(3))
