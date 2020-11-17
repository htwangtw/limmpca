import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt

from patsy import dmatrices


def combine_data(exp_design, pca_scores):
    # add PC scores to the data frame
    data = exp_design.copy()

    for i in range(pca_scores.shape[-1]):
        data[f"factor_{i + 1}"] = pca_scores[:, i]
    return data

def model_comparison(candidate_models, exp_design, pca_scores):
    data = combine_data(exp_design, pca_scores)
    fitted_models = []
    for mn in candidate_models.keys():
        m = candidate_models[mn]
        mlm = smf.mixedlm("factor_1"  + m["formula"],
                            data,
                            groups=m["groups"],
                            re_formula=m["re_formula"],
                            vc_formula=m["vcf"])
        # fit the model
        mlm_res = mlm.fit(reml=True, method='cg')
        fitted_models.append(mlm_res)


def parallel_mixed_modelling(model, exp_design, pca_scores, plot=False):
    m_components = pca_scores.shape[-1]
    data = combine_data(exp_design, pca_scores)
    # run the model
    fitted_models = []
    for i in range(m_components):
        print(f"{i + 1} / {m_components}")
        formula = f"factor_{i + 1}"  + model["formula"]
        mixed = smf.mixedlm(formula,
                            data,
                            groups=model["groups"],
                            re_formula=model["re_formula"],
                            vc_formula=model["vcf"])
        # fit the model
        mixed_fit = mixed.fit(reml=True, method='cg')
        # print(mixed_fit.summary())
        # save fitted model
        fitted_models.append(mixed_fit)

        if plot:
            fe_params = pd.DataFrame(mixed_fit.fe_params,columns=['LMM'])
            random_effects = pd.DataFrame(mixed_fit.random_effects)
            random_effects = random_effects.transpose()

            Y, _   = dmatrices(formula, data=data, return_type='matrix')
            Y      = np.asarray(Y).flatten()

            plt.figure(figsize=(18,9))
            ax1 = plt.subplot2grid((2,2), (0, 0))
            ax2 = plt.subplot2grid((2,2), (0, 1))
            ax3 = plt.subplot2grid((2,2), (1, 0), colspan=2)

            fe_params.plot(ax=ax1)
            random_effects.plot(ax=ax2)

            ax3.plot(Y.flatten(),'o',color='k',label = 'Observed', alpha=.25)
            fitted = mixed_fit.fittedvalues
            print("The MSE is " + str(np.mean(np.square(Y.flatten()-fitted))))
            ax3.plot(fitted,lw=1,alpha=.5)
            ax3.legend(loc=0)
            #plt.ylim([0,5])
            plt.show()
    return fitted_models


def effect_matrix_decomposition(fitted_models):

    def fixed_effects(results):
        # fixed effect
        fe_params = results.fe_params
        mf = {}
        for j, name in enumerate(results.model.exog_names):
            coef = fe_params[name]
            tau_fe = results.model.exog[:, j]
            mf[name] = np.dot(coef, tau_fe)
        mf = pd.DataFrame(mf)
        return mf

    def random_effects(results):
        # random effects of the current factor
        # this is a dictionary of data frames one entry per group
        re = results.random_effects
        mr = {}
        # for each group
        for group_ix, group in enumerate(results.model.group_labels):
            # get random structure design
            ix = results.model.row_indices[group]
            mat = []
            if results.model.exog_re_li is not None:
                mat.append(results.model.exog_re_li[group_ix])
            for j in range(results.k_vc):
                mat.append(results.model.exog_vc.mats[j][group_ix])
            mat = np.concatenate(mat, axis=1)
            mat = pd.DataFrame(mat, columns=re[group].index)

            # handle random structure of multiple level
            # this loop would be ignored if no nested structure was modeled
            levels = []
            for vc in results.model.exog_vc.names:
                colname = f"random effect: {vc}"
                # get all columns related to this random structure
                vc_labels = [r for r in re[group].index if vc in r]
                levels += vc_labels

                coef = re[group][vc_labels]
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

            # random coefficients or structure with one level only
            remained_labels = re[group][~re[group].index.isin(levels)]
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
        mr = pd.DataFrame(mr)
        return mr

    # claculate the fitted values and model error
    m_components = len(fitted_models)
    effect_mats = []
    for i in range(m_components):
        results = fitted_models[i]
        mf_i = fixed_effects(results)
        mr_i = random_effects(results)
        # fitted_val should be the same as results.fittedvalues
        fitted_val = mf_i.sum(axis=1) + mr_i.sum(axis=1)
        # resid_i should be the same as results.resid
        resid_i = results.model.endog - fitted_val
        resid_i = pd.Series(resid_i, name="residuals")
        effect_mat = pd.concat((mf_i, mr_i, resid_i), axis=1)
        effect_mats.append(effect_mat)

    return effect_mats


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
    return percent_var_exp