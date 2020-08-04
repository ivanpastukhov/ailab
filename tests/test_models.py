import pytest
import numpy as np
from ai_tools.utils.models import BoWVectorizer, BaseVectorizer


def vectorizer(s):
    d = {
        'a': np.array([0.,0.,0.,0.]),
        'b': np.array([0.1, -0.4, 1.3, 0.5]),
        'c': np.array([-0.4, 0.3, 0.3, 1.2]),
        'd': np.array([1.2, 2.2, -4.3, -1.1]),
        'e': np.array([1.,1.,1.,1.,])
        }
    return np.array([d[i] for i in s])

class TestBoWVectorizer:
    def test_vectorize_default(self):
        self.s = 'a b c e d'
        self.bow_vectorizer = BoWVectorizer(vectorizer)
        res = self.bow_vectorizer.vectorize(self.s)
        expected = np.array([0.380, 0.620, -0.340, 0.320])
        np.testing.assert_almost_equal(res, expected, 1e-6)
        return

    def test_vectorize_aggs(self):
        self.s = 'a b c e d'
        self.bow_vectorizer = BoWVectorizer(vectorizer)
        res = self.bow_vectorizer.vectorize(self.s, aggs=[np.mean, np.min, np.max, np.sum])
        expected = np.concatenate((
            [0.380, 0.620, -0.340, 0.320],
            [-0.4, -0.4, -4.3, -1.1],
            [1.2, 2.2, 1.3, 1.2],
            [1.9, 3.1, -1.7, 1.6]
        ))
        np.testing.assert_almost_equal(res, expected, 1e-6)
        return




