import pandas as pd
ACUTE='́'
df=pd.read_csv('data/train.csv')
def ov(l,f):
    a=set(str(l).lower().replace(ACUTE,'')); b=set(str(f).lower().replace(ACUTE,''))
    return len(a&b)/len(a) if a else 0.0
o=df.apply(lambda r: ov(r.lemma_ru,r.form_vvz),axis=1)
print("rows",len(df),"flagged<0.2:",(o<0.2).sum(), f"({(o<0.2).mean()*100:.2f}%)")
for t in (0.3,0.4,0.5): print(f"thr {t}: {(o<t).sum()}")
