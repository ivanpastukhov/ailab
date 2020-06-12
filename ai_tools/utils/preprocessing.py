from functools import lru_cache
from collections import Counter
import re
import types


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

def text_normalizer(string, morpher):
  r = r'[А-Яа-яA-Za-z]+'
  string = re.findall(r, string)
  string = (normalize_form(token, morpher) for token in string)
  return ' '.join(string)

