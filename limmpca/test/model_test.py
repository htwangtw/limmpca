from ..model import _combine_data, _get_m
from .make_data import test_data

import numpy as np
import pandas as pd

# load and generate test data
exp_design, pca_scores, models = test_data()

def test_combine_data():
    data = _combine_data(exp_design, pca_scores)
    assert data.shape[-1] == exp_design.shape[-1] + pca_scores.shape[-1]
    assert data["factor_1"][0] == pca_scores[0, 0]

