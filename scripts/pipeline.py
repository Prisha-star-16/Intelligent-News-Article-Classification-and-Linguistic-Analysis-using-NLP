"""NewsLens pipeline (from newslens_nlp.ipynb). Writes JSON/PNG artifacts into ../public and parity tests into ../tests."""
import pandas as pd, numpy as np, re, json, os, time, warnings, sys
from collections import Counter
warnings.filterwarnings('ignore'); np.random.seed(42)
D=sys.argv[1] if len(sys.argv)>1 else '/home/claude/work/data'
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PD,PP,PM,PT=[os.path.join(ROOT,p) for p in ('public/data','public/plots','public/model','tests')]
def dump(o,p,**k): json.dump(o,open(p,'w'),**k)
L={0:'World',1:'Sports',2:'Business',3:'Sci/Tech'}; NAMES=list(L.values())
def load(f):
    d=pd.read_csv(os.path.join(D,f),header=None,names=['label','title','desc']).fillna('')
    d['label']=d['label'].astype(int)-1
    d['text']=(d.title+' '+d.desc).str.replace('\\',' ',regex=False).str.replace(r'\s+',' ',regex=True).str.strip()
    return d
tr,te=load('train.csv'),load('test.csv')
n_dup=int(tr.text.duplicated().sum())
tr=tr.drop_duplicates('text').reset_index(drop=True); tr['name']=tr.label.map(L); te['name']=te.label.map(L)
from sklearn.model_selection import train_test_split
Xtr,Xva,ytr,yva=train_test_split(tr.text.values,tr.label.values,test_size=.1,random_state=42,stratify=tr.label.values)
Xte,yte=te.text.values,te.label.values
isub,_=train_test_split(np.arange(len(Xtr)),train_size=20000,random_state=42,stratify=ytr)
Xsub,ysub=Xtr[isub],ytr[isub]
print('split',len(Xtr),len(Xva),len(Xte),flush=True)

# ---------- EDA ----------
from nltk.corpus import stopwords
NLTK_SW=set(stopwords.words('english'))
tr['wc']=tr.text.str.split().str.len(); tr['cc']=tr.text.str.len()
def words(t): return [w for w in re.sub(r'[^a-zA-Z\s]',' ',t.lower()).split() if w not in NLTK_SW and len(w)>2]
top={}; from wordcloud import WordCloud
for l,n in L.items():
    c=Counter(); [c.update(words(t)) for t in tr.text[tr.label==l]]
    top[n]=[{'word':w,'count':k} for w,k in c.most_common(20)]
    WordCloud(width=600,height=400,background_color=None,mode='RGBA',colormap='viridis',max_words=120,random_state=42).generate_from_frequencies(dict(c.most_common(300))).to_file(os.path.join(PP,'wordcloud_'+n.lower().replace('/','_')+'.png'))
raw=set(); [raw.update(t.lower().split()) for t in tr.text]; cl=set(); [cl.update(words(t)) for t in tr.text]
eda={'train_size':len(tr),'test_size':len(te),'duplicates_removed':n_dup,'class_distribution':tr.name.value_counts().to_dict(),'vocab_raw':len(raw),'vocab_clean':len(cl),
 'length_stats':{n:{'mean_words':round(float(tr[tr.name==n].wc.mean()),1),'median_words':float(tr[tr.name==n].wc.median()),'mean_chars':round(float(tr[tr.name==n].cc.mean()),1)} for n in NAMES},
 'length_hist':{n:np.histogram(tr[tr.name==n].wc.clip(upper=100),bins=25,range=(0,100))[0].tolist() for n in NAMES},'top_words':top,
 'citation':'Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. NIPS.'}
dump(eda,f'{PD}/eda.json',indent=1); print('eda done',flush=True)

# ---------- Deployed model: web-reproducible cleaning (no lemmatization) ----------
from spacy.lang.en.stop_words import STOP_WORDS as SW
def clean_web(t):
    t=re.sub(r'<[^>]+>',' ',t.lower()); t=re.sub(r'https?://\S+|www\.\S+',' ',t); t=re.sub(r'[^a-z\s]',' ',t)
    return ' '.join(w for w in t.split() if w not in SW and len(w)>2)
Ctr,Cva,Cte=[np.array([clean_web(t) for t in x]) for x in (Xtr,Xva,Xte)]
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
tf=TfidfVectorizer(ngram_range=(1,2),max_features=30000,min_df=2,sublinear_tf=True,norm='l2')
Ttr=tf.fit_transform(Ctr); Tva=tf.transform(Cva); Tte=tf.transform(Cte)
svc=RandomizedSearchCV(LinearSVC(random_state=42),{'C':[.1,.3,.5,1,2,5],'max_iter':[2000,3000]},n_iter=5,cv=3,scoring='f1_macro',random_state=42,n_jobs=1).fit(Ttr,ytr)
best=svc.best_estimator_; print('svc',svc.best_params_,svc.best_score_,flush=True)
pt=best.predict(Tte); acc=accuracy_score(yte,pt); f1=f1_score(yte,pt,average='macro')
print('TEST',acc,f1,flush=True)
cm=confusion_matrix(yte,pt); rep=classification_report(yte,pt,target_names=NAMES,output_dict=True)
fn=np.array(tf.get_feature_names_out()); coef=best.coef_
tfm={n:[{'term':fn[i],'score':round(float(m[i]),4)} for i in np.argsort(m)[-15:][::-1]] for l,n in L.items() for m in [np.asarray(Ttr[ytr==l].mean(axis=0)).ravel()]}
topterms={n:[{'term':fn[i],'coef':round(float(coef[l][i]),4)} for i in np.argsort(coef[l])[-15:][::-1]] for l,n in L.items()}
rng=np.random.RandomState(42); mis=np.where(pt!=yte)[0]
mex=[{'text':Xte[i][:220],'true':L[yte[i]],'pred':L[pt[i]]} for i in rng.choice(mis,15,replace=False)]
lime=[]
for i in [0,1,2]:
    v=Tte[i]; c=pt[i]; contribs=sorted(({'term':fn[j],'contribution':round(float(v[0,j]*coef[c][j]),4)} for j in v.nonzero()[1]),key=lambda x:-abs(x['contribution']))[:10]
    lime.append({'text':Xte[i][:200],'true':L[yte[i]],'pred':L[c],'term_contributions':contribs})
ev={'model_name':'LinearSVC (tuned) + TF-IDF(1,2)','best_params':svc.best_params_,'val_accuracy':round(float(accuracy_score(yva,best.predict(Tva))),4),'test_accuracy':round(float(acc),4),'test_macro_f1':round(float(f1),4),
 'per_class':{n:{'precision':round(rep[n]['precision'],4),'recall':round(rep[n]['recall'],4),'f1':round(rep[n]['f1-score'],4),'support':int(rep[n]['support'])} for n in NAMES},
 'confusion_matrix':cm.tolist(),'labels':NAMES,'misclassified':mex,'top_terms':topterms,'tfidf_terms':tfm,'lime':lime}
dump(ev,f'{PD}/evaluation.json',indent=1)
dump({'labels':NAMES,'ngram_range':[1,2],'vocabulary':{k:int(v) for k,v in tf.vocabulary_.items()},'idf':[round(float(x),5) for x in tf.idf_],
 'coef':[[round(float(x),4) for x in r] for r in coef],'intercept':[round(float(x),4) for x in best.intercept_],'stopwords':sorted(SW),'test_accuracy':round(float(acc),4),'test_macro_f1':round(float(f1),4)},f'{PM}/model.json',separators=(',',':'))
idx=np.random.RandomState(1).choice(len(Xte),500,replace=False)
dump([{'text':Xte[i],'pred':int(pt[i])} for i in idx],f'{PT}/parity.json',separators=(',',':'))
print('model exported MB',os.path.getsize(f'{PM}/model.json')/1e6,flush=True)

# ---------- Subset analysis: lemmas, W2V, spaCy, model grid ----------
import spacy, scipy.sparse as sp
from tqdm import tqdm
nlp=spacy.load('en_core_web_sm',disable=['parser','ner'])
def lem(texts):
    pre=[clean_web(t) or 'empty' for t in texts]; out=[]
    for d in nlp.pipe(pre,batch_size=256):
        l=[t.lemma_ for t in d if t.lemma_ not in SW and len(t.lemma_)>2]; out.append(' '.join(l) or 'empty')
    return np.array(out)
Lsub=lem(Xsub); NV=5000; Xv=Xva[:NV]; yv=yva[:NV]; Lv=lem(Xv); print('lemmatized',flush=True)
from gensim.models import Word2Vec
w2v=Word2Vec([t.split() for t in Lsub],vector_size=100,window=5,min_count=5,sg=0,workers=1,epochs=10,seed=42)
def dv(t): v=[w2v.wv[x] for x in t.split() if x in w2v.wv]; return np.mean(v,0) if v else np.zeros(100)
Wsub=np.array([dv(t) for t in Lsub]); Wv=np.array([dv(t) for t in Lv])
from sklearn.decomposition import PCA
tw=sorted(w2v.wv.key_to_index,key=lambda w:-w2v.wv.get_vecattr(w,'count'))[:400]; co=PCA(2,random_state=42).fit_transform(np.array([w2v.wv[w] for w in tw]))
kw={0:['war','military','government','president','country'],1:['game','team','player','score','win'],2:['stock','market','company','profit','share'],3:['technology','software','computer','internet','science']}
cen={l:np.mean([w2v.wv[k] for k in ks if k in w2v.wv],0) for l,ks in kw.items()}
cs=lambda a,b:float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))
aff=[int(max(cen,key=lambda l:cs(w2v.wv[w],cen[l]))) for w in tw]
sim={w:[[a,round(float(s),3)] for a,s in w2v.wv.most_similar(w,topn=6)] for w in ['market','game','iraq','software','president','company','team','internet'] if w in w2v.wv}
dump({'words':tw,'x':[round(float(x),3) for x in co[:,0]],'y':[round(float(y),3) for y in co[:,1]],'affinity':aff,'similar':sim,'vocab':len(w2v.wv)},f'{PD}/embeddings.json')
nf=spacy.load('en_core_web_sm'); KP=['NOUN','VERB','ADJ','ADV','PROPN','NUM','PUNCT']
pos={n:Counter() for n in NAMES}; ents={n:{e:Counter() for e in ['PERSON','ORG','GPE','DATE','MONEY']} for n in NAMES}; sl={n:[] for n in NAMES}
def ling(texts,ys=None,stats=False):
    F=[]
    for k,d in enumerate(tqdm(nf.pipe([t[:600] for t in texts],batch_size=128),total=len(texts),mininterval=30)):
        tk=[t for t in d if not t.is_space]; tot=max(len(tk),1); pc=Counter(t.pos_ for t in tk); ec=Counter(e.label_ for e in d.ents); ss=[len(s) for s in d.sents]
        F.append([pc[p]/tot for p in KP[:5]]+[ec['PERSON'],ec['ORG'],ec['GPE'],np.mean(ss) if ss else 0,pc['NOUN']/(pc['VERB']+1e-5),len(texts[k])])
        if stats:
            n=L[ys[k]]; pos[n].update(pc); sl[n]+=ss
            for e in d.ents:
                if e.label_ in ents[n]: ents[n][e.label_][e.text.lower()]+=1
    return np.array(F,dtype=np.float32)
Fsub=ling(list(Xsub),ysub,True); Fv=ling(list(Xv))
dump({'pos_distribution':{n:{p:round(pos[n][p]/max(sum(pos[n].values()),1)*100,2) for p in KP} for n in NAMES},
 'top_entities':{n:{e:[{'entity':a,'count':c} for a,c in ents[n][e].most_common(6)] for e in ents[n]} for n in NAMES},
 'avg_sentence_length':{n:round(float(np.mean(sl[n])),2) for n in NAMES},'docs_analyzed':len(Xsub)},f'{PD}/linguistics.json',indent=1)
samples=[]
for l in range(4):
    for i in [j for j in np.where(yte==l)[0] if pt[j]==l][:2]:
        d=nf(Xte[i][:600]); samples.append({'label_name':L[l],'text':Xte[i][:500],'pred':L[pt[i]],'tokens':[{'text':t.text,'pos':t.pos_,'dep':t.dep_,'head':t.head.text,'lemma':t.lemma_} for t in d if not t.is_space][:40],
          'entities':[{'text':e.text,'label':e.label_} for e in d.ents][:12],'noun_chunks':[c.text for c in d.noun_chunks][:10]})
ex=['The Federal Reserve raised interest rates by 25 basis points amid inflation concerns.','Scientists discovered a new exoplanet orbiting a distant star in the Milky Way.','Manchester United won the Champions League final after a dramatic penalty shootout.','The United Nations called for an immediate ceasefire in the war-torn region.','Apple reported record quarterly revenue driven by iPhone sales and services growth.','Researchers developed a new AI model that can diagnose cancer with high accuracy.','The marathon runner broke the world record by over a minute.','Oil prices soared after OPEC announced surprise production cuts.']
dump({'samples':samples,'examples':ex},f'{PD}/samples.json',indent=1); print('linguistics done',flush=True)
from sklearn.naive_bayes import MultinomialNB; from sklearn.linear_model import LogisticRegression; from sklearn.ensemble import RandomForestClassifier; from sklearn.preprocessing import StandardScaler
bow=CountVectorizer(ngram_range=(1,2),max_features=30000,min_df=2); Bs,Bv=bow.fit_transform(Lsub),bow.transform(Lv)
tf2=TfidfVectorizer(ngram_range=(1,2),max_features=30000,min_df=2,sublinear_tf=True); Ts,Tv=tf2.fit_transform(Lsub),tf2.transform(Lv)
s1,s2=StandardScaler().fit(Wsub),StandardScaler().fit(Fsub); Ws,Wv_=s1.transform(Wsub),s1.transform(Wv); Fs,Fv_=s2.transform(Fsub),s2.transform(Fv)
Cs=sp.hstack([Ts,sp.csr_matrix(Ws),sp.csr_matrix(Fs)]).tocsr(); Cv=sp.hstack([Tv,sp.csr_matrix(Wv_),sp.csr_matrix(Fv_)]).tocsr()
fsets=[('BoW',Bs,Bv,1),('TF-IDF',Ts,Tv,1),('W2V',sp.csr_matrix(Ws),sp.csr_matrix(Wv_),0),('Combined',Cs,Cv,0)]
res=[]
for fname,A,B,nb in fsets:
    for cname,mk in [('MultinomialNB',lambda:MultinomialNB(alpha=.5)),('LogisticReg',lambda:LogisticRegression(max_iter=300,solver='saga',random_state=42)),('LinearSVC',lambda:LinearSVC(C=1,random_state=42)),('RandomForest',lambda:RandomForestClassifier(n_estimators=50,random_state=42))]:
        if (cname=='MultinomialNB' and not nb) or (cname=='RandomForest' and fname in('Combined','BoW')): res.append({'model':cname,'features':fname,'skipped':True,'note':'negative values' if cname=='MultinomialNB' else 'too slow'}); continue
        m=mk(); t0=time.time(); m.fit(A,ysub); tt=time.time()-t0; t1=time.time(); p=m.predict(B); ti=(time.time()-t1)/NV*1000
        res.append({'model':cname,'features':fname,'val_acc':round(float(accuracy_score(yv,p)),4),'val_f1':round(float(f1_score(yv,p,average='macro')),4),'train_s':round(tt,2),'infer_ms':round(ti,3)}); print(res[-1],flush=True)
dump({'trained_on':len(Xsub),'evaluated_on':NV,'results':res,'deployed':{'model':'LinearSVC (tuned)','features':'TF-IDF (full train, no lemmatization)','test_accuracy':round(float(acc),4),'test_macro_f1':round(float(f1),4),'size_mb':round(os.path.getsize(f'{PM}/model.json')/1e6,2)}},f'{PD}/models.json',indent=1)
print('ALL DONE',flush=True)
