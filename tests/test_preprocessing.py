import pytest
from collections import Counter
from ai_tools.utils.preprocessing import lazy_token_counter, preprocess

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
    return

# def test_embedding_matrix_token2id_check():
#     token2id = {'abc': 0, 'bcd': 1, 'cde': 2}
#     with pytest.raises(ValueError) as err:
#         embedding_matrix(token2id, './',)
#     assert err.match('IDs of tokenizer must be not less than 1, because "0" is reserved for padding.')
#     return

def test_preprocess():
    s = 'привет))) кек)) ло(( л как дела?я абрикос НаЮге Рос heLlo 666 2342 ))) ))) 55 55 5 00!!00'
    res = preprocess(s)
    expected = 'привет ))) кек )) ло (( л как дела я абрикос наюге рос hello 666 2342 ))) 55 5 00'
    assert res == expected
    return

