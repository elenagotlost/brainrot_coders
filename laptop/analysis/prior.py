import pandas as pd, numpy as np, re
from load import load; from feats import *
from sklearn.linear_model import LogisticRegression
df=load(); e=df[df.language=='eng'].copy()
e['nv']=e.options.apply(numeric_opts)
L='ABCDE'
def tn(t): 
    xs=re.findall(r'\d+(?:[.,]\d+)?',t)[1:]
    return set(float(x.replace(',','.')) for x in xs)
def opt_feats(r):
    v=r.nv; s=sorted(v); t=tn(r.text); F=[]
    for i,x in enumerate(v):
        rk=s.index(x); f=[0]*5; f[rk]=1
        le=[0]*5; le[i]=1
        F.append(f+le+[x in t, float(x).is_integer(), v.count(x)>1, abs(x-np.median(v))/(np.std(v)+1e-9), v==s])
    return np.array(F,float)
n=e[e.nv.notna()]
def XY(sub):
    X=[];y=[];g=[]
    for _,r in sub.iterrows():
        F=opt_feats(r)
        for i in range(5): X.append(F[i]); y.append(int(L[i]==r.label))
    return np.array(X),np.array(y)
tr=n[n.split=='train']; va=n[n.split=='val']
X,y=XY(tr); m=LogisticRegression(C=1.0,max_iter=1000).fit(X,y)
Xv,yv=XY(va); p=m.predict_proba(Xv)[:,1].reshape(-1,5)
pred=p.argmax(1); true=np.array([L.index(l) for l in va.label])
print('val numeric problems',len(va),'acc',(pred==true).mean())
print('median heuristic', np.mean([sorted(r.nv).index(r.nv[L.index(r.label)])==2 for _,r in va.iterrows()]))
print('always D', (va.label=='D').mean(), 'always C', (va.label=='C').mean())
# train CV
from sklearn.model_selection import GroupKFold
accs=[];med=[]
ids=n.id.values
from sklearn.model_selection import KFold
kf=KFold(5,shuffle=True,random_state=0)
allp=[]
for a,b in kf.split(n):
    X,y=XY(n.iloc[a]); m=LogisticRegression(max_iter=1000).fit(X,y)
    Xv,yv=XY(n.iloc[b]); p=m.predict_proba(Xv)[:,1].reshape(-1,5)
    true=np.array([L.index(l) for l in n.iloc[b].label]); accs.append((p.argmax(1)==true).mean())
print('5fold acc', np.mean(accs), np.std(accs))
print('median heuristic all', np.mean([sorted(r.nv).index(r.nv[L.index(r.label)])==2 for _,r in n.iterrows()]))
print(dict(zip(['r0','r1','r2','r3','r4','A','B','C','D','E','intext','int','dup','dist','asc'],m.coef_[0].round(2))))
