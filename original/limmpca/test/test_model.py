from ..model import _combine_data, _fixed_effects
from .make_data import test_data

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf


# load and generate test data
exp_design, pca_scores, models = test_data()

def test_fixed_effects():
    data = _combine_data(exp_design, pca_scores)

    assert data.shape[-1] == exp_design.shape[-1] + pca_scores.shape[-1]
    assert data["factor_1"][0] == pca_scores[0, 0]

    model = models["full_model"]
    formula = "factor_1"  + model["formula"]
    mixed = smf.mixedlm(formula,
                        data,
                        groups=model["groups"],
                        re_formula=model["re_formula"],
                        vc_formula=model["vc_formula"])
    # fit the model
    fitted_model = mixed.fit(reml=True)

    mf = _fixed_effects(fitted_model)
    sum_mf = np.sum(mf, axis=1)
    dot_fixed = np.dot(fitted_model.fe_params, fitted_model.model.exog.T)
    check_mf = sum_mf == dot_fixed
    assert check_mf.all() == True