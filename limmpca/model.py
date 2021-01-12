import pandas as pd
import numpy as np

import re

import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

def _combine_data(exp_design, pca_scores):
    # combine PC scores and experiment design
    # generate a dataframe that's suitable for statsmodel
    data = exp_design.copy()
    for i in range(pca_scores.shape[-1]):
        data[f"factor_{i + 1}"] = pca_scores[:, i]
    return data

def _check_category(name):
    # categorical variables are broken into different levels
    # we need to find them and calculate a combined score for variance explained
    category_var = r"C\(([A-Za-z0-9_]+)\)\[T.[0-9]+\]$"
    cat_finder = re.search(category_var, name)
    if cat_finder is None:
        return name
    else:
        return cat_finder.group(1)

def _get_m(tau, coef):
    return np.dot(tau, coef)

def _fixed_effects(fitted_model):
    '''
    Input is a statsmodel LMM output
    '''
    fe_params = fitted_model.fe_params  # get the full fixed variable coefficient matrix
    fe_names = fitted_model.model.exog_names  # fixed variable names
    fe_input = fitted_model.model.exog

    # get fixed effect
    mf = {}
    for j, name in enumerate(fe_names):
        m = _get_m(fe_input[:, j], fe_params[name])

        # update the effective matrix
        name = _check_category(name)
        if name in mf.columns.tolist():
            mf[name] += m
        else: # not a categorical variable or found new category
            mf[name] = m
    return pd.DataFrame(mf)

def _get_random_structure(group_ix, k_vc, exog_vc_mat, exog_re_li):
    mat = []
    if exog_re_li is not None:
        mat.append(exog_re_li[group_ix])
    for j in range(k_vc):
        mat.append(exog_vc_mat[j][group_ix])
    return np.concatenate(mat, axis=1)

def _get_nested_structure(mr, mat, re_group, exog_vc_names):
    # handle random structure of multiple level
    # this loop would be ignored if no nested structure was modeled
    levels = []
    for vc in exog_vc_names:
        colname = f"random effect: {vc}"
        # get all columns related to this random structure
        vc_labels = [r for r in re_group.index if vc in r]
        levels += vc_labels

        mr_j = _get_m(mat[vc_labels], re_group[vc_labels])
        # create an entry in dict if not present
        try:
            mr[colname]
        except KeyError:
            mr[colname] = np.array([])

        # concatenate current subject's variance
        # to the data
        if not mr[colname].any():
            mr[colname] = mr_j
        else:
            mr[colname] = np.concatenate([mr[colname], mr_j], axis=0)
    return levels, mr

def _get_random_coeff(mr, mat, remained_labels):
    for re_name, coef in remained_labels.iteritems():
        colname = f"random effect: {re_name}"
        mr_j = _get_m(mat[re_name], coef)
        # create an entry in dict if not present
        try:
            mr[colname]
        except KeyError:
            mr[colname] = np.array([])
        # concatenate current subject's variance
        # to the data
        if not mr[colname].any():
            mr[colname] = mr_j
        else:
            mr[colname] = np.concatenate([mr[colname], mr_j], axis=0)
    return mr

def _random_effects(fitted_model):
    '''
    Input is a statsmodel LMM output
    '''
    # random effects of the current factor
    # this is a dictionary of data frames one entry per group
    re = fitted_model.random_effects
    groups = fitted_model.model.group_labels
    k = fitted_model.k_vc
    vc_mat = fitted_model.model.exog_vc.mats
    vc_names = fitted_model.model.exog_vc.names
    exog_re_li = fitted_model.model.exog_re_li

    mr = {}
    # for each group
    for group_ix, group in enumerate(groups):
        # get random structure design
        mat = _get_random_structure(group_ix, k, vc_mat, exog_re_li)
        mat = pd.DataFrame(mat, columns=re[group].index)

        # handle random structure of multiple level
        # this loop would be ignored if no nested structure was modeled
        levels, mr = _get_nested_structure(mr, mat, re[group], vc_names)

        # random coefficients or structure with one level only
        remained_labels = re[group][~re[group].index.isin(levels)]
        mr = _get_random_coeff(mr, mat, remained_labels)
    return pd.DataFrame(mr)

def parallel_mixed_model(model, exp_design, pca_scores):
    '''
    Linear mixed model step in LiMM-PCA.

    Attributes
    ----------
    model: dict
        a dictionary containing set up for statsmodel formulat for
        mixedml
        must contain following keys:
        - formula:
            The patsy style formula of the fixed effect
        - groups:
            random effect groups
            random effect passed to mixedml input of the same name
        - re_formula: random intercept/coefficient of "groups"
            default to random intercept model
        - vc_formula: variance componment formula for knested effect;
            set to None if not needed

        fit LMM on the PCA results with the specified model
    '''

    assert ["formula", "groups", "re_formula", "vc_formula"] in list(model.keys())
    m_components = pca_scores.shape[-1]
    data = combine_data(exp_design, pca_scores)
    llf, effectmat = [], []
    for i in range(m_components):
        print(f"{i + 1} / {m_components}")
        formula = f"factor_{i + 1}"  + model["formula"]
        mixed = smf.mixedlm(formula,
                            data,
                            groups=model["groups"],
                            re_formula=model["re_formula"],
                            vc_formula=model["vc_formula"])
        # fit the model
        fitted_model = mixed.fit(reml=True)

        # save fitted model
        # fittedmodels.append(fitted_model)

        # effect mat decomposition
        mf = fixed_effects(fitted_model)
        mr = random_effects(fitted_model)

        # fitted_val should be the same as fitted_model.fittedvalues
        fitted_val = mf.sum(axis=1) + mr.sum(axis=1)
        # resid_i should be the same as fitted_model.resid
        resid = fitted_model.model.endog - fitted_val

        resid = pd.Series(resid, name="residuals")
        em = pd.concat((mf, mr, resid), axis=1)

        # save the model
        effectmat.append(em)
        llf.append(fitted_model.llf)
    return llf, effectmat

def variance_explained(effect_mats):
    n_components = len(effect_mats)
    percent_var_exp = [em.apply(np.var) for em in effect_mats]
    percent_var_exp = pd.concat(percent_var_exp, axis=1)
    est_var_full = np.sum(percent_var_exp.values)
    percent_var_exp /= est_var_full / 100
    percent_var_exp.columns = range(1, n_components + 1)
    percent_var_exp = np.log1p(percent_var_exp)
    percent_var_exp["Effect"] = percent_var_exp.index
    percent_var_exp = percent_var_exp.melt(id_vars="Effect",
                                        value_name="log(variance %)",
                                        var_name="PC")
    chart = sns.barplot(x="Effect", y="log(variance %)",
                hue="PC", data=percent_var_exp
                )
    plt.show()
    return percent_var_exp
