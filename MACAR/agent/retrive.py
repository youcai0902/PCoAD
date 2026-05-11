import math
from typing import List
from collections import Counter
import jieba
import re
import numpy as np
from typing import Optional


def tokenize(text: str) -> List[str]:
    return [w for w in jieba.lcut(text) if w.strip()]


class _BM25:

    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b


        self.doc_tokens = [tokenize(doc) for doc in docs]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_len) / len(docs) if docs else 0.0


        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]


        self.idf = self._calc_idf()

    def _calc_idf(self) -> dict:
        N = len(self.docs)
        df = {}
        for tokens in self.doc_tokens:
            unique_words = set(tokens)
            for w in unique_words:
                df[w] = df.get(w, 0) + 1

        idf = {}
        for word, freq in df.items():

            idf[word] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def get_scores(self, query: str) -> List[float]:
        query_tokens = tokenize(query)
        scores = []
        for i, tokens in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = self.doc_len[i]
            term_freqs = self.doc_freqs[i]
            for word in query_tokens:
                if word not in self.idf:
                    continue
                tf = term_freqs.get(word, 0)

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += self.idf[word] * numerator / denominator
            scores.append(score)
        return scores


def BM25(query: str, docs: List[str]) -> List[float]:
    model = _BM25(docs)
    return model.get_scores(query)



_zh_model: Optional[object] = None
_en_model: Optional[object] = None

def _contains_chinese(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def _get_model(lang: str):
    global _zh_model, _en_model
    if lang == 'zh':
        if _zh_model is None:
            from FlagEmbedding import FlagModel
            _zh_model = FlagModel('BAAI/bge-large-zh-v1.5',
                                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                  use_fp16=True)
        return _zh_model
    else:
        if _en_model is None:
            from FlagEmbedding import FlagModel
            _en_model = FlagModel('BAAI/bge-large-en-v1.5',
                                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                  use_fp16=True)
        return _en_model

def sim(text1: str, text2: str) -> float:

    if _contains_chinese(text1) or _contains_chinese(text2):
        model = _get_model('zh')
    else:
        model = _get_model('en')


    emb1 = np.array(model.encode(text1))
    emb2 = np.array(model.encode(text2))


    sim_val = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    return sim_val
