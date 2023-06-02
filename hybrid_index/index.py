import pickle
from abc import abstractmethod
from typing import List, Tuple, Callable

import openai
import faiss
import nltk
from faiss import IndexIDMap, IndexFlatIP
from nltk import RegexpTokenizer
import numpy as np


def openai_emb(passages: List[str]):

    chunk_size = 100
    arr = np.empty((0, 1536)).astype('float32')

    for i in range(0, len(passages), chunk_size):
        chunk = passages[i:min(i + chunk_size, len(passages))]
        res = openai.Embedding.create(
            input=chunk,
            engine='text-embedding-ada-002'
        )
        arr = np.concatenate((arr, np.array([r['embedding'] for r in res['data']]).astype('float32')), axis=0)

    return arr

async def openai_aemb(passages: List[str]):

    chunk_size = 100
    arr = np.empty((0, 1536)).astype('float32')

    for i in range(0, len(passages), chunk_size):
        chunk = passages[i:min(i + chunk_size, len(passages))]
        res = await openai.Embedding.acreate(
            input=chunk,
            engine='text-embedding-ada-002'
        )
        arr = np.concatenate((arr, np.array([r['embedding'] for r in res['data']]).astype('float32')), axis=0)

    return arr


def normalize_array(array: np.ndarray, infimum: int = None):
    """
    This technique was taken from
    :param array:
    :param infimum:
    :return:
    """
    min_value = infimum if infimum is not None else np.min(array)
    max_value = np.max(array)
    normalized_array = (array - min_value) / (max_value - min_value)
    return normalized_array


class BM25:
    def __init__(self, tokenizer: Callable[[str], List[str]]):
        self.corpus_size = 0
        self.avgdl = 0
        self.nd = {}  # word -> number of documents with word
        self.doc_freqs = []  # For each document there is a list of frequencies for each word
        self.idf = {}
        self.doc_len = []  # Length of each document in the corpus
        self.num_words = 0  # Total number of words in the corpus
        self.tokenizer = tokenizer

    @abstractmethod
    def _calc_idf(self):
        pass

    @abstractmethod
    def get_scores(self, query: str):
        pass

    def add(self, corpus: List[str]):

        self.corpus_size += len(corpus)

        # Iterates over the tokenized corpus
        for document in [self.tokenizer(doc.lower()) for doc in corpus]:
            self.doc_len.append(len(document))
            self.num_words += len(document)

            # Maps the frequency for each word
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            # Adds one to the number of documents that contain a word for every word the current document contains
            for word, freq in frequencies.items():
                if word in self.nd:
                    self.nd[word] += 1
                else:
                    self.nd[word] = 1

        # Compute the average document length
        self.avgdl = self.num_words / self.corpus_size

        self._calc_idf()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query.lower())
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


# BM25 Outlier Plus, my personal (and questionable version of BM25 Plus)
class BM25OPlus(BM25):
    def __init__(self, tokenizer: Callable[[str], List[str]], k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta

        super().__init__(tokenizer)

    def _calc_idf(self):
        for word, freq in self.nd.items():
            idf = (self.corpus_size / freq ** 2)
            self.idf[word] = idf

    def get_scores(self, query: str):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in self.tokenizer(query):
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))

        # Returns the normalized scores, 0 is the infimum of BM25
        return normalize_array(score, 0)


class HybridIndex:
    a: float
    index_id: str
    openai_key: str
    id_map: List
    semantic_index: IndexFlatIP
    lexical_index: BM25OPlus

    def __init__(self, index_id: str, openai_key: str = None, a: float = 0.7):
        self.index_id = index_id
        self.a = a

        if openai_key:
            openai.api_key = openai_key

        # Mapping from ids to unique string ids
        self.id_map = []

        # Sets up the tokenizer and the lexical index
        nltk.download('punkt')

        self.lexical_index = BM25OPlus(RegexpTokenizer(r'\w+').tokenize)

        # Sets up the faiss index with OpenAI embedding dimensions
        self.semantic_index = IndexFlatIP(1536)

    async def add(self, corpus: List[Tuple[str, str]]):
        ids, documents = zip(*corpus)

        # Update BM25 Index
        self.lexical_index.add(documents)

        # Get embeddings for new documents and adds them into the index
        embeddings = await openai_emb(documents)
        self.semantic_index.add(embeddings)

        self.id_map.extend(ids)

    async def query(self, query: str, top_n: int):

        # Get 10 times more results from each index to compare them
        query_length = min(top_n * 10, len(self.id_map))

        # Get semantic results
        query_vector = await openai_aemb([query])
        faiss_scores, faiss_indices = self.semantic_index.search(query_vector, query_length)
        faiss_scores, faiss_indices = normalize_array(faiss_scores[0], -1), faiss_indices[0]
        faiss_result = {self.id_map[r]: faiss_scores[idx] for idx, r in enumerate(faiss_indices)}

        # Returns the top n results in order from the bm25 index
        bm25_scores = self.lexical_index.get_scores(query)
        bm25_results = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:query_length]
        bm25_result = {self.id_map[r]: bm25_scores[r] for r in bm25_results}

        # Combine results
        combined_scores = {}
        for id in set(list(faiss_result.keys()) + list(bm25_result.keys())):
            faiss_score = faiss_result.get(id, 0)
            bm25_score = bm25_result.get(id, 0)

            combined_scores[id] = self.a * faiss_score + (1 - self.a) * bm25_score

        # Return the ranked top n indices
        return sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_n]

    def serialize(self):

        # Create a dictionary representation of your object
        data = self.__dict__

        # Serialize the index to a byte array
        data['semantic_index'] = faiss.serialize_index(self.semantic_index)

        # Pickle the dictionary
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, pickled_data: dict, openai_key: str = None):

        # Sets the openai key
        if openai_key:
            openai.api_key = openai_key

        # Unpickle the data
        data = pickle.loads(pickled_data)

        # Deserializes the faiss index
        data['semantic_index'] = faiss.deserialize_index(data['semantic_index'])

        # Sets the api key
        openai.api_key = data['openai_key']

        instance = cls.__new__(cls)
        instance.__dict__.update(data)
        return instance
