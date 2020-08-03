"""
Based on Tong and Evans (1996) A statistical Approach to Automatic OCR Error Correction in Context.
(http://www.aclweb.org/anthology/W96-0108)
"""
import os
import pickle
import time
import functools
from typing import List, Union, Optional, Tuple, Iterable, ClassVar

import numpy as np
import scipy
from sklearn.feature_extraction.text import CountVectorizer

from contextualized_transduction.utils import expand_path, read_counts

ScipySparseSCRCSCMatrix = Union[scipy.sparse.csr.csr_matrix, scipy.sparse.csc.csc_matrix]

class CharSpacer:
    BOS: ClassVar[str] = '_'  # sequence-boundary marker

    def __init__(self, words: Iterable[str],
                 vectorizer: CountVectorizer = None,
                 X: ScipySparseSCRCSCMatrix = None,
                 ngram_range: Tuple[int, int] = (3, 3),
                 random_seed: int = 42):
        """
        Class for holding a n-gram vector space model for representing words and searching
        for graphically similar words.
        :param words: All words that will define the n-gram model.
        :param vectorizer: Optionally, an sklearn CountVectorizer that implements the precomputed n-gram model.
        :param X: Optionally, the precomputed word-ngram matrix over `words` with dimensions (# words, # n-grams)
        :param ngram_range: A tuple that specifies the range of n-grams that will be extracted from `words`.
               (3, 3) extracts all and only trigrams.
        :param random_seed: Random seed for np.random.choice.
        """

        self.words = list(words)
        self.random_seed = random_seed

        if vectorizer is not None and X is not None:
            # loaded from file
            assert len(self.words) == X.shape[0], \
                f'Mismatch in number of words and number of rows in X: {len(self.words)} vs {X.shape[0]}'
            self.X = X
            self.vectorizer = vectorizer
            self.ngram_range = self.vectorizer.ngram_range
        else:
            self.ngram_range = ngram_range
            # @NB This is a "set-of-ngrams" representation. The assumption is that higher n n-grams
            # rarely repeat in a real word, so we ignore n-gram counts in a word (aka term frequencies).
            self.vectorizer = CountVectorizer(input='content',
                                              lowercase=False,
                                              analyzer='char',
                                              ngram_range=self.ngram_range,
                                              binary=True,
                                              dtype=np.bool)
            start = time.time()
            print('Start building vector space of ngrams...')
            # insert word boundary characters and populate vector space
            self.X = self.vectorizer.fit_transform([self.BOS + w + self.BOS for w in self.words])
            end = time.time() - start
            print(f'Constructed vector space in {end:.2f} sec.')
        print(f'Dimensions of vector space are: {self.X.shape}. n-gram range is: {self.ngram_range}.')
        self.X = self.X.tocsc()
        np.random.seed(self.random_seed)
        self.norm = self.X.sum(axis=1, dtype=np.uint16)  # @NB accurate enough without normalization

    @functools.lru_cache(maxsize=2 ** 14)
    def candidates(self, word: str, top_k: int = 200, s: float = 0.6) -> List[str]:
        """
        Retrieve `top_k` words that are the most similar to `word` under cosine similarity.
        :param word: The query word.
        :param top_k: The number of the most similar words to retrieve.
        :param s: A float in (0, 1] that defines the portion of n-grams of the word to be used. The lower the faster
               the computation. (Also, we don't even need a perfect match with the `word`s n-grams because some of them
               are due to an OCR error.)
        :return: The list of `top_k` most similar words.
        """
        # insert word boundaries
        q = self.vectorizer.transform([self.BOS + word + self.BOS])
        try:
            # Take a `s`*100% sample of n-gram features from q uniformly at random without replacement.
            # (The number of features is one bottleneck, hence we speed up by considering only a subset of n-grams).
            # We approximate cosine similarity by unnormalized dot product. Since q is a binary vector, we get the dot
            # product by summing only columns self.X[:, j] where q[j] is non-zero. The approximation works
            # because (i) the normalization for q does not affect the ranking order of candidates, and (ii)
            # the normalization for self.X is very roughly the same for all words (especially those that
            # ABBYY does not recognize---long compounds).
            r = self.X[:, np.random.choice(q.indices, round(q.indices.shape[0] * s), replace=False)]. \
                    sum(axis=1, dtype=np.uint8) / self.norm  # @NB avoid costly normamization
            # partial sort (-top_k-th element is in the right place, all elements larger than top_k-th value are
            # to its right) in O(n) vs O(n log n)
            # Sorting over non-zero elements can give a slight speed-up (yet finding non-zero elements is costly)
            rA = r.A[:, 0]  # repackage (# words, 1)-dim. matrix as 1-dim. array
            non_zero = np.where(rA != 0)[0]  # np.where returns a tuple of length=# array dim.
            top_k = min(top_k, non_zero.shape[0])  # ... otherwise error on highly unusual (non-German?) words
            idx = np.argpartition(rA[non_zero], -top_k)[-top_k:]
            return [self.words[non_zero[i]] for i in idx]
        except ValueError:
            # q is the empty vector
            return []

    def candidates_with_zeros(self, word: str, top_k=200, s=0.6) -> List[str]:
        # same logic as `self.candidates`, however outputs will differ because argpartition will
        # order equal elements (incl. elements equal to top_k-th) differently due to different input.
        q = self.vectorizer.transform([self.BOS + word + self.BOS])
        r = self.X[:, np.random.choice(q.indices, round(q.indices.shape[0] * s), replace=False)]. \
            sum(axis=1, dtype=np.uint8)
        idx = np.argpartition(r.A[:, 0], -top_k)[-top_k:]
        return [self.words[i] for i in idx]

    @classmethod
    def from_pickle(cls, pkl: str, words: Iterable[str], random_seed: Optional[int] = 42):
        """
        Read from pickle file a precomputed vector space model.
        :param pkl: The pickle filename.
        :param words: The words that correspond to the rows of the word-ngram matrix `X`.
        :param random_seed: Random seed for randomized sub-selection of feature n-grams.
        :return: A CharSpacer object.
        """
        path2pkl: str = expand_path(pkl)
        try:
            start = time.time()
            print(f'Start loading char spacer from {path2pkl}...')
            with open(path2pkl, 'rb') as w:
                spacer_data = pickle.load(w)
            end = time.time() - start
            print(f'Loaded in {end:.2f} sec.')
        except OSError as e:
            print(f'"{path2pkl}" exists?', os.path.exists(path2pkl))
            raise e
        return cls(words=words, vectorizer=spacer_data['vectorizer'], X=spacer_data['X'], random_seed=random_seed)

    def to_pickle(self, pkl: str, relative2data: bool = True):
        """
        Write vector space model to file.
        :param pkl: The pickle filename.
        :param relative2data: Write to data directory.
        """
        with open(expand_path(pkl) if relative2data else pkl, 'wb') as w:
            pickle.dump({'vectorizer': self.vectorizer, 'X': self.X}, w)
