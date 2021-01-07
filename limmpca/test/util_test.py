from ..util import rescale

import numpy as np

def test_rescale():
    test_data1 = np.array([
        [1, 0, 3, 4, 5],
        [3, 1, 3, 2, 2]
        ])
    test_data2 = np.array([
        [0.1, 0.6, 0.3, 0.4, 0.5],
        [0.3, 0.1, 0.3, 0.7, 0.2]
        ])
    rescale1 = rescale(test_data1)
    rescale2 = rescale(test_data2)

    print(rescale1)
    assert max(rescale1.ravel()) == 1
    assert min(rescale1.ravel()) == 0

    print(rescale2)
    assert max(rescale2.ravel()) == 1
    assert min(rescale2.ravel()) == 0