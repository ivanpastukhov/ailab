from functools import lru_cache
from collections import Counter
from itertools import groupby
import re
import types
import numpy as np
from tqdm import tqdm
# import fasttext


def lazy_token_counter(str_generator):
    counter = Counter()
    if not isinstance(str_generator, types.GeneratorType):
        raise TypeError('str_generator must be generator object of strings')
    for string in str_generator:
        counter.update(string.split())
    return counter

@lru_cache(512)
def normalize_form(token, morpher):
    return morpher.parse(token)[0].normal_form

# def text_normalizer(string, morpher):
#     r = r'[А-Яа-яA-Za-z]+'
#     string = re.findall(r, string)
#     string = (normalize_form(token, morpher) for token in string)
#     return ' '.join(string)

# def embedding_matrix(token2id, ft_model_path, emb_size=100, show_progress=True):
#     if min(token2id.values()) != 1:
#         raise ValueError('IDs of tokenizer must be not less than 1, because "0" is reserved for padding.')
#     ##TODO: show_progress==False
#     if not show_progress:
#         raise NotImplementedError()
#     emb_mat = np.zeros((len(token2id)+1, emb_size))
#     model = fasttext.load_model(ft_model_path)
#     for token, idx in tqdm(token2id.items()):
#         emb_mat[idx] = model.get_word_vector(token)
#     return emb_mat

def preprocess(s):
      if not isinstance(s, str):
        raise ValueError('Object of type "str" is expected.')
      s = s.lower()
      pattern = r'[a-zа-я0-9]+|[\(\)]+'
      s = (token for token, _ in groupby(re.findall(pattern, s)))
      return ' '.join(s)
