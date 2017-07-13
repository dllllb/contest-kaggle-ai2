# https://www.kaggle.com/c/the-allen-ai-science-challenge
"""
IR and word embedding sample: https://github.com/5vision/kaggle_allen
samples: https://github.com/kemaswill/Allen_AI_Science_Challenge_JunweiPan
Science Wikipedia articles: https://gist.github.com/sidharthshah/a3957238fcfcad53e46c
Wikipedia dump: https://en.wikipedia.org/wiki/Special:Export

Solutions:
1-st place: https://github.com/Cardal/Kaggle_AllenAIscience
2-nd place: https://github.com/bwilbertz/kaggle_allen_ai
3-rd place: https://github.com/amsqr/Allen_AI_Kaggle
"""

import re
from elasticsearch import Elasticsearch, ElasticsearchException
from zipfile import ZipFile
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from gensim import matutils
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.decomposition.pca import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from dstools.ml.ensemble import ModelEnsemble
from dstools.util.timer import Timer


def n_similarity(wv, ws1, ws2):
    v1 = [v for v in [wv[word] for word in ws1 if word in wv] if v is not None]
    v2 = [v for v in [wv[word] for word in ws2 if word in wv] if v is not None]

    if v1 and v2:
        return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))
    else:
        return 0


class StopwordTokenizer:
    def __init__(self):
        self.stopwords = set(stopwords.words("english"))

    def tokenize(self, text):
        clean_text = re.sub('[^a-zA-Z]', ' ', text)
        words = clean_text.lower().split()
        words = [w for w in words if w not in self.stopwords]
        return words


def predict_ir(df, index, fields, search_type):
    es = Elasticsearch()

    resutls = []
    for ix, row in df.iterrows():
        scores = {'id': ix}
        for letter in ['A', 'B', 'C', 'D']:
            query = {
                "query": {
                    "multi_match": {
                        "query": row.question,
                        "type": search_type,
                        "fields": fields
                    }
                },
                "rescore": {
                    "window_size": 50,
                    "query": {
                        "rescore_query": {
                            "multi_match": {
                                "query": row['answer'+letter],
                                "type": search_type,
                                "fields": fields
                            }
                        },
                        "query_weight": 0.5,
                        "rescore_query_weight": 2
                    }
                }
            }

            try:
                hits = es.search(index=index, doc_type='doc,page', body=query, _source=True)['hits']['hits']
                if len(hits) > 0:
                    scores[letter] = hits[0]['_score']
                else:
                    scores[letter] = 0
            except ElasticsearchException as e:
                print(e)

        resutls.append(scores)

    return pd.DataFrame(resutls).set_index('id')


def predict_ir_rescore_sum(df, index, fields, search_type):
    es = Elasticsearch()

    resutls = []
    for ix, row in df.iterrows():
        scores = {'id': ix}
        for letter in ['A', 'B', 'C', 'D']:
            query = {
                "query": {
                    "multi_match": {
                        "query": row.question,
                        "type": search_type,
                        "fields": fields
                    }
                },
                "rescore": {
                    "window_size": 50,
                    "query": {
                        "rescore_query": {
                            "multi_match": {
                                "query": row['answer'+letter],
                                "type": search_type,
                                "fields": fields
                            }
                        },
                        "query_weight": 0.5,
                        "rescore_query_weight": 2
                    }
                }
            }

            try:
                hits = es.search(index=index, doc_type='doc,page', body=query, _source=False, size=50)['hits']['hits']
                total_score = 0
                for hit in hits:
                    total_score += hit['_score']
                scores[letter] = total_score
            except ElasticsearchException as e:
                print(e)

        resutls.append(scores)

    return pd.DataFrame(resutls).set_index('id')


def predict_ir_sum(df, index, fields, search_type):
    es = Elasticsearch()

    resutls = []
    for ix, row in df.iterrows():
        scores = {'id': ix}
        for letter in ['A', 'B', 'C', 'D']:
            query = {
                "query": {
                    "multi_match": {
                        "query": row.question+' '+row['answer'+letter],
                        "type": search_type,
                        "fields": fields
                    }
                }
            }

            try:
                hits = es.search(index=index, doc_type='doc,page', body=query, _source=False, size=50)['hits']['hits']
                total_score = 0
                for hit in hits:
                    total_score += hit['_score']
                scores[letter] = total_score
            except ElasticsearchException as e:
                print(e)

        resutls.append(scores)

    return pd.DataFrame(resutls).set_index('id')


# http://www.ck12.org/flx/show/epub/user:anBkcmVjb3VydEBnbWFpbC5jb20./Concepts_b_v8_vdt.epub
def index_ck_12_conecpts_v8(index_name='ck12-concepts'):
    es = Elasticsearch()

    with ZipFile('Concepts_b_v8_vdt_html.zip') as zf:
        es.indices.delete(index=index_name, ignore=[404])

        index_settings = {
            'settings': {
                "analysis": {
                    "analyzer": {
                        "default": {
                            # "type": "snowball",
                            # "language": "English"
                            "type": 'standard',
                            "stopwords": "_english_"
                        }
                    }
                },
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
            },
            "mappings": {
                "doc": {
                    "properties": {
                        "text": {
                            "type": "string",
                            "similarity": "BM25"
                        },
                        "title": {
                            "type": "string",
                            "similarity": "BM25"
                        }
                    }
                }
            },
        }

        es.indices.create(index=index_name, body=index_settings)

        for name in zf.namelist():
            with zf.open(name) as f:
                bs = BeautifulSoup(f, 'lxml')
                tag = bs.find('h1')
                title = tag.text.strip()
                text = ''
                while True:
                    tag = tag.find_next_sibling()
                    if tag is None or tag.name == 'h1':
                        doc = {
                            'title': title,
                            'text': text
                        }
                        es.index(index=index_name, doc_type='doc', body=doc)
                        if tag is None:
                            break
                        else:
                            title = tag.text.strip()
                            text = ''
                    else:
                        text += tag.text

        es.indices.optimize(index=index_name, max_num_segments=1, wait_for_merge=True)


def predict_we(df, word2vec):
    resutls = []
    tokenizer = StopwordTokenizer()
    for ix, row in df.iterrows():
        q_tokens = set(tokenizer.tokenize(row.question))
        scores = {'id': ix}
        for letter in ['A', 'B', 'C', 'D']:
            a_tokens = set(tokenizer.tokenize(row['answer'+letter]))
            sim = n_similarity(word2vec, q_tokens, a_tokens)
            scores[letter] = sim

        resutls.append(scores)

    return pd.DataFrame(resutls).set_index('id')


# download a dataset from [GloVe](http://nlp.stanford.edu/projects/glove/)
# add a header in form of N_WORDS space VECTOR_LEN to the beginning of the file
class GloveEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, vectors):
        self.vectors = vectors

    def fit(self, X, y):
        self.word2vec = Word2Vec.load_word2vec_format(self.vectors, binary=False)
        return self

    def predict_proba(self, X):
        pred = predict_we(X, self.word2vec)
        return pred.div(pred.max(axis=1), axis=0).fillna(0)

    def predict(self, X):
        return self.predict_proba(X).idxmax(axis=1)


class IrEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, index, fields=None, search_type='most_fields'):
        self.index = index
        self.fields = fields or ['_all']
        self.search_type = search_type

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        pred = predict_ir(X, index=self.index, fields=self.fields, search_type=self.search_type)
        return pred.div(pred.max(axis=1), axis=0).fillna(0)

    def predict(self, X):
        return self.predict_proba(X).idxmax(axis=1)


class IrEstimatorRescoreSum(BaseEstimator, ClassifierMixin):
    def __init__(self, index, fields=None, search_type='most_fields'):
        self.index = index
        self.fields = fields or ['_all']
        self.search_type = search_type

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        pred = predict_ir_rescore_sum(X, index=self.index, fields=self.fields, search_type=self.search_type)
        return pred.div(pred.max(axis=1), axis=0).fillna(0)

    def predict(self, X):
        return self.predict_proba(X).idxmax(axis=1)


class IrEstimatorSum(BaseEstimator, ClassifierMixin):
    def __init__(self, index, fields=None, search_type='most_fields'):
        self.index = index
        self.fields = fields or ['_all']
        self.search_type = search_type

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        pred = predict_ir_sum(X, index=self.index, fields=self.fields, search_type=self.search_type)
        return pred.div(pred.max(axis=1), axis=0).fillna(0)

    def predict(self, X):
        return self.predict_proba(X).idxmax(axis=1)


def cv_test(est):
    with Timer('cv'):
        df = pd.read_csv('training_set.tsv', index_col='id', sep='\t')

        scores = cross_val_score(
            estimator=est,
            X=df.drop('correctAnswer', axis=1),
            y=df.correctAnswer,
            cv=5,
            n_jobs=1,
            verbose=1)
        print(est)
        print(scores)
        print('mean: {mean}, std: {std}'.format(mean=scores.mean(), std=scores.std()))


def submission(est, name='results'):
    with Timer('submission'):
        df_train = pd.read_csv('training_set.tsv', index_col='id', sep='\t')
        model = est.fit(df_train.drop('correctAnswer', axis=1), df_train.correctAnswer)

        df = pd.read_csv('test_set.tsv.gz', index_col='id', sep='\t')
        preds = model.predict(df)
        res = pd.Series(preds, index=df.index, name='correctAnswer')
        res.to_csv(name+'.csv', index_label='id', header=True)

# mean: 0.383613928769, std: 0.0224772900287
# cv execution time: 34.7804338932 sec
est1 = IrEstimator('ck12-concepts')  # Concepts_b_v8_vdt_html.zip, stopwords

# mean: 0.392806695044, std: 0.00974997673719
# cv execution time: 49.3968729973 sec
est1_1 = IrEstimatorRescoreSum('ck12-concepts')  # Concepts_b_v8_vdt_html.zip, stopwords

# mean: 0.367610590866, std: 0.00736769904979
# cv execution time: 49.5336530209 sec
est1_2 = IrEstimatorSum('ck12-concepts')  # Concepts_b_v8_vdt_html.zip, stopwords

# mean: 0.305605320779, std: 0.00359779255663
# cv execution time: 22.8514099121 sec
est3 = GloveEstimator('glove.6B.300d-ai2.txt.bz2')

# mean: 0.378805972527, std: 0.00684518647143
# cv execution time: 134.214504004 sec
est4 = ModelEnsemble(
    intermediate_estimators=[
        GloveEstimator('glove.6B.300d-ai2.txt.bz2'),
        IrEstimator('ck12-concepts'),
    ],
    assembly_estimator=LogisticRegression(C=1),
    ensemble_train_size=1
)

# mean: 0.378411475843, std: 0.0157684051548
# cv execution time: 219.582041979 sec
est5 = ModelEnsemble(
    intermediate_estimators=[
        GloveEstimator('glove.6B.300d-ai2.txt.bz2'),
        IrEstimator('ck12-concepts'),
    ],
    assembly_estimator=Pipeline([
        ('pca', PCA(n_components=4)),
        ('lr', LogisticRegression(C=1)),
    ]),
    ensemble_train_size=1
)

# mean: 0.387199480557, std: 0.00767928030034
# cv execution time: 305.623694897 sec
est6 = ModelEnsemble(
    intermediate_estimators=[
        GloveEstimator('glove.6B.300d-ai2.txt.bz2'),
        IrEstimatorRescoreSum('ck12-concepts'),
    ],
    assembly_estimator=LogisticRegression(C=1),
    ensemble_train_size=1
)

# cv_test(est1_1)
# submission(est10, 'results-sum2')
# index_ck_12_conecpts_v8()
