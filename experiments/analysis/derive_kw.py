import re, json, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from load import load; from copilot.topics import tag
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
df=load()
eng=df[df.language=='eng'].set_index('id'); cat=df[df.language=='cat'].set_index('id')
topic=eng.apply(lambda r: tag(r.text,r.text_only)[0],axis=1)
out={}
for lang,d in [('eng',eng),('cat',cat)]:
    txt=d.loc[topic.index].text.str.replace(r'^\s*\d+\.\s*','',regex=True).str.lower()
    cv=CountVectorizer(min_df=4,max_df=0.25,token_pattern=r"[a-zà-úçl·']{3,}",binary=True)
    X=cv.fit_transform(txt); voc=np.array(cv.get_feature_names_out())
    out[lang]={}
    for t in sorted(topic.unique()):
        y=(topic==t).values
        s,_=chi2(X,y); pos=np.asarray(X[y].mean(0)).ravel()>np.asarray(X[~y].mean(0)).ravel()
        s=np.where(pos,s,0); out[lang][t]=voc[np.argsort(-s)[:12]].tolist()
for t in out['eng']: print(f'{t:18} EN: {", ".join(out["eng"][t][:10])}\n{"":18} CA: {", ".join(out["cat"][t][:10])}')
os.makedirs('data/derived',exist_ok=True)
json.dump(out,open('data/derived/topic_keywords_derived.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
pd.DataFrame({'id':topic.index,'topic':topic.values,'level':eng.loc[topic.index].level.values}).to_csv('data/derived/topics_by_id.csv',index=False)
