# NewsLens

AG News topic classifier (World / Sports / Business / Sci-Tech) that runs **entirely in the browser**, plus the NLP analysis behind it (EDA, Word2Vec, spaCy linguistics, model comparison, evaluation).

## Deploy to Vercel
1. Push this folder to a Git repo and import it at vercel.com/new (framework auto-detected: Next.js), **or** run `npx vercel` in this folder.
2. No environment variables or server needed: everything is static.

## Develop
```bash
npm install
npm run dev      # http://localhost:3000
npm test         # TS classifier vs scikit-learn on 500 test articles
```

## Regenerate the data (optional)
`public/data`, `public/model` and `public/plots` were produced by `scripts/pipeline.py` (condensed from `newslens_nlp.ipynb`).
```bash
pip install scikit-learn pandas numpy scipy gensim spacy nltk wordcloud tqdm
python -m spacy download en_core_web_sm && python -c "import nltk; nltk.download('stopwords')"
python scripts/pipeline.py /path/to/folder/with/train.csv+test.csv   # AG News CSVs, no header
```
## Notes on differences from the notebook
- Deployed model is trained on the full training set with a cleaning pipeline the browser can reproduce exactly (no lemmatization), so TS and scikit-learn agree.
- The model comparison grid (BoW/TF-IDF/W2V/Combined) uses the 20k subset, scored on 5k validation articles.
- All 30k TF-IDF features are exported (no pruning), 1.7 MB.
