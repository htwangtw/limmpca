import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from patsy import dmatrices

import seaborn as sns
import matplotlib.pyplot as plt


# To Do: isolate the effect matrix decomposition part
# create an abstract class for PMM for different behaviours of the same function
# during model fitting and bootstrapping?

# class pmm(ABC):
#     def fit(X, Y):
#       pass


class ParallelMixedModel:
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

    Methods
    -------
    fit(X, Y)
        fit LMM on the PCA results with the specified model
    '''

    def __init__(self, model):
        self.model = model  # TODO: check dictionary kes
        self.fittedmodels = []
        self.effectmat = []
        self.llf = []
        self.estimate = None

    def fit(self, X, Y):
        m_components = Y.shape[-1]
        data = self.combine_data(X, Y)
        for i in range(m_components):
            print(f"{i + 1} / {m_components}")
            formula = f"factor_{i + 1}"  + self.model["formula"]
            mixed = smf.mixedlm(formula,
                                data,
                                groups=self.model["groups"],
                                re_formula=self.model["re_formula"],
                                vc_formula=self.model["vc_formula"])
            # fit the model
            fitted_model = mixed.fit(reml=True, method='cg')

            # save fitted model
            self.fittedmodels.append(fitted_model)
            self.llf.append(fitted_model.llf)

            # effect mat decomposition
            mf = self.fixed_effects(fitted_model)
            mr = self.random_effects(fitted_model)

            # fitted_val should be the same as fitted_model.fittedvalues
            fitted_val = mf.sum(axis=1) + mr.sum(axis=1)
            # resid_i should be the same as fitted_model.resid
            resid = fitted_model.model.endog - fitted_val

            resid = pd.Series(resid, name="residuals")
            em = pd.concat((mf, mr, resid), axis=1)

            # save the model
            self.effectmat.append(em)

        # summarise variance explained
        self.estimate = self.variance_explained(self.effectmat)

    @staticmethod
    def combine_data(exp_design, pca_scores):
        # add PC scores to the data frame
        data = exp_design.copy()
        for i in range(pca_scores.shape[-1]):
            data[f"factor_{i + 1}"] = pca_scores[:, i]
        return data

    @staticmethod
    def fixed_effects(fitted_model):
        # fixed effect
        fe_params = fitted_model.fe_params
        mf = {}
        for j, name in enumerate(fitted_model.model.exog_names):
            coef = fe_params[name]
            tau_fe = fitted_model.model.exog[:, j]
            mf[name] = np.dot(coef, tau_fe)
        mf = pd.DataFrame(mf)
        return mf

    @staticmethod
    def get_random_structure(group_ix, k_vc, exog_vc_mat, exog_re_li):
        mat = []
        if fitted_model.model.exog_re_li is not None:
            mat.append(fitted_model.model.exog_re_li[group_ix])
        for j in range(k_vc):
            mat.append(exog_vc_mat[j][group_ix])
        return np.concatenate(mat, axis=1)

    @staticmethod
    def get_nested_structure(mr, mat, re_group, exog_vc_names):
        # handle random structure of multiple level
        # this loop would be ignored if no nested structure was modeled
        levels = []
        for vc in exog_vc_names:
            colname = f"random effect: {vc}"
            # get all columns related to this random structure
            vc_labels = [r for r in re_group.index if vc in r]
            levels += vc_labels

            coef = re_group[vc_labels]
            tau = mat[vc_labels]
            mr_j = np.dot(tau, coef)
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

    @staticmethod
    def get_random_coeff(mr, mat, remained_labels):
        for re_name, coef in remained_labels.iteritems():
            colname = f"random effect: {re_name}"
            tau = mat[re_name]
            mr_j = np.dot(tau, coef)
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

    @staticmethod
    def random_effects(fitted_model):
        # random effects of the current factor
        # this is a dictionary of data frames one entry per group
        re = fitted_model.random_effects
        mr = {}
        # for each group
        for group_ix, group in enumerate(fitted_model.model.group_labels):
            # get random structure design
            mat = get_random_structure(group_ix, fitted_model.k_vc,
                                       fitted_model.model.exog_vc.mats,
                                       fitted_model.model.exog_re_li)
            mat = pd.DataFrame(mat, columns=re[group].index)

            # handle random structure of multiple level
            # this loop would be ignored if no nested structure was modeled
            levels, mr = get_nested_structure(mr, mat, re[group],
                                              fitted_model.model.exog_vc.names)

            # random coefficients or structure with one level only
            remained_labels = re[group][~re[group].index.isin(levels)]
            mr = get_random_coeff(mr, mat, remained_labels)
        mr = pd.DataFrame(mr)
        return mr

    @staticmethod
    def variance_explained(effect_mats):
        n_components = len(effect_mats)
        percent_var_exp = [em.apply(np.var) for em in effect_mats]
        percent_var_exp = pd.concat(percent_var_exp, axis=1)
        est_var_full = np.sum(percent_var_exp.values)
        percent_var_exp /= est_var_full / 100
        percent_var_exp.columns = range(1, n_components + 1)
        percent_var_exp["Effect"] = percent_var_exp.index
        percent_var_exp = percent_var_exp.melt(id_vars="Effect",
                                            value_name="variance(%)",
                                            var_name="PC")
        chart = sns.barplot(x="Effect", y="variance(%)",
                    hue="PC", data=percent_var_exp
                    )
        plt.show()
        return percent_var_exp