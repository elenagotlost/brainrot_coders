Laptop-side analysis scripts (need pandas + scikit-learn; never run on the board). Run from the repo root:

    M3KANG_DATA=data PYTHONPATH=laptop/analysis python3 laptop/analysis/floor2.py      # prior solver by route
    M3KANG_DATA=data PYTHONPATH=laptop/analysis python3 laptop/analysis/prior.py       # rank prior, LR on option features
    M3KANG_DATA=data PYTHONPATH=laptop/analysis python3 laptop/analysis/derive_kw.py   # topics + chi2 keywords (eng+cat)

derive_kw.py writes data/derived/topics_by_id.csv and data/derived/topic_keywords_derived.json
(not in git, like everything under data/). The topic tagger itself is copilot/topics.py.
On Windows, set PYTHONUTF8=1 too.
