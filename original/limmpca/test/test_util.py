from ..util import rescale, correct_scale

import numpy as np
import pandas as pd


def test_correct_scale():
    # generate test data - first subject has two entries
    ridno = ["R2234", "R2234", "R0452", "R4345"]
    labels = ["a", "b", "c", "d", "e"]
    scores = np.array([
        [0.1, 0.6, 0.3, 0.4, 0.5],
        [0.4, 0.8, 0.4, 0.1, 0.5],
        [0.3, 0.1, 0.3, 0.7, 0.2],
        [0.4, 0.3, 0.2, 0.8, 0.2]
        ])
    data = pd.DataFrame(scores, columns=labels)
    data["RIDNO"] = ridno
    data["sex"] = ["F", "F", "M", "F"]

    # manually compute the rescale data
    rs = np.vstack(
        (rescale(scores[:2]),
         np.array([rescale(s)for s in scores[2:]])
        ))
    data_rescaled = correct_scale(data, labels)
    comparison = data_rescaled.loc[:, labels].values == rs
    assert comparison.all() == True

def test_rescale():
    test_data1 = np.array([
        [1, 0, 3, 4, 5],
        [3, 1, 3, 2, 2]
        ])
    test_data2 = np.array([
        [0.1, 0.6, 0.3, 0.4, 0.5],
        [0.3, 0.1, 0.3, 0.7, 0.2]
        ])
    test_data3 = np.random.rand(55, 12)

    rescale1 = rescale(test_data1)
    rescale2 = rescale(test_data2)

    assert max(rescale1.ravel()) == 1
    assert min(rescale1.ravel()) == 0

    assert max(rescale2.ravel()) == 1
    assert min(rescale2.ravel()) == 0
    rescale3 = rescale(test_data3)
    assert max(rescale3.ravel()) == 1
    assert min(rescale3.ravel()) == 0