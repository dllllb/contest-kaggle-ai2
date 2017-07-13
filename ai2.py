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

