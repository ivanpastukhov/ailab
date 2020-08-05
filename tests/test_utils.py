from ai_tools.utils.utils import positive_sampler, negative_sampler, sampler
import numpy as np
import pytest


def test_positive_sampler_non_equal():
    questions, answers = ['a','b','c'], ['aa', 'bb']
    with pytest.raises(ValueError) as err:
        res = positive_sampler(questions, answers)
        next(res)
    assert err.match('Length of array of questions must be equal to length of array of answers.')
    return

def test_positive_sampler_single_question():
    questions, answers = ['a', 'a', 'a'], ['aa', 'ab', 'ac']
    expected = [('a', 'aa', 1),
                ('a', 'ab', 1),
                ('a', 'ac', 1)]
    res = [*positive_sampler(questions, answers)]
    assert res == expected
    return

def test_positive_sampler_multiple_questions():
    questions, answers = ['a', 'a', 'b'], ['aa', 'ab', 'ba']
    expected = [('a', 'aa', 1),
                ('a', 'ab', 1),
                ('b', 'ba', 1)]
    res = [*positive_sampler(questions, answers)]
    assert res == expected
    return

def test_negative_sampler_unbalanced():
    question, questions, answers = 'foo', ['foo']*int(1e6) + ['bar'], ['baz' for _ in range(int(1e6)+1)]
    max_tries = 3
    with pytest.raises(ValueError) as err:
        res = negative_sampler(question, questions, answers, max_tries)
        next(res)
    assert err.match('Could not sample negative example after {} tries. More unique questions must be added'.format(max_tries))
    return

def test_sampler_frac_011():
    questions, answers = ['a', 'b', 'c'], ['aa', 'ba', 'ca']
    pos_frac = 0.11
    res = sampler(questions, answers, pos_frac, 432, 100)
    res_fraction = np.mean([label for _, _, label in res])
    np.testing.assert_allclose(res_fraction, pos_frac, atol=0.01)
    return

def test_sampler_frac_001():
    questions, answers = ['a', 'b', 'c'], ['aa', 'ba', 'ca']
    pos_frac = 0.01
    res = sampler(questions, answers, pos_frac, 432, 100)
    res_fraction = np.mean([label for _, _, label in res])
    np.testing.assert_allclose(res_fraction, pos_frac, atol=0.01)
    return

def test_sampler_frac_0999():
    questions, answers = ['a', 'b', 'c'], ['aa', 'ba', 'ca']
    pos_frac = 0.999
    with pytest.raises(NotImplementedError) as err:
        _ = sampler(questions, answers, pos_frac, 432, 100)
        next(_)
    assert err.match('Only < 0.5 pos_frac values can be used currently.')
    return

def test_sampler_npositives():
    n_answers = 1000
    questions = np.random.random_sample((n_answers, 10))
    answers = np.random.random_sample((n_answers, 10))
    pos_frac = 0.234
    res = sampler(questions, answers, pos_frac, 432, 100)
    assert np.sum([label for _, _, label in res]) == n_answers
    return


