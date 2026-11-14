import re
NUM=re.compile(r'^\s*[-−]?\d+(?:[.,]\d+)?\s*$')
def tonum(s):
    s=s.strip().replace('−','-').replace(',','.')
    try: return float(s)
    except: return None
def numeric_opts(o):
    if len(o)!=5: return None
    v=[tonum(o[k]) for k in 'ABCDE']
    return None if any(x is None for x in v) else v
