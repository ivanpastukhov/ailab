import pytest
from collections import Counter
from ai_tools.utils.preprocessing import lazy_token_counter

def test_lazy_token_counter_type():
    var = ['wewef fweff ewf' for i in range(10)]
    with pytest.raises(TypeError) as err:
        lazy_token_counter(var)
    assert err.match('str_generator must be generator object of strings')
    return

def test_lazy_token_counter():
    var = (i for i in ['azzzz b c', 'c b a', ' ', '', 'a  c z', 'abra', 'ab ra'])
    res = lazy_token_counter(var)
    assert res == Counter({'c': 3, 'b': 2, 'a': 2, 'azzzz': 1, 'z': 1, 'abra': 1, 'ab': 1, 'ra': 1})

