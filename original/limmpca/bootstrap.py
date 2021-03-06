import pandas as pd
import numpy as np

from sklearn.utils import resample

from .model import ParallelMixedModel

def bootstrap_effect(obs_sigmasqr, boot_sample_size, h0_models):

    def residuals(sigma, boot_sample_size):
        return np.random.normal(0, sigma, boot_sample_size)

    def fixed_effects(exog, fe_params):
        # fixed effect
        return np.dot(exog, fe_params)

    def random_effects(results, sigmas):
        # random effects of the current factor
        # this is a dictionary of data frames one entry per group
        re = results.random_effects
        mr = []
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
            rand_idx = [idx.replace('random effect: ', '') for idx in re[group].index]
            null_ss = []
            for j in rand_idx:
                real_label = j.split("[")[0]
                ss = sigmas[real_label]
                null_ss.append(np.random.normal(0, ss))
            mr.append(np.dot(mat, np.array(null_ss)))
        mr = np.concatenate(mr)
        return mr

    m_components = len(h0_models)
    Y_ests = []
    for i in range(m_components):
        model = h0_models[i]
        ss = obs_sigmasqr.loc["residuals", i]
        Y_est = residuals(ss, boot_sample_size)
        Y_est += fixed_effects(model)
        ss = obs_sigmasqr.loc[:, i]
        Y_est += random_effects(model, ss)
        Y_ests.append(Y_est)
    Y_ests = np.array(Y_ests).T
    return Y_ests

def calculate_gllr(h1_llf, h0_llf):
    gllr = [h1_llf[i] - h0_llf[i]
                for i in range(len(h1_llf))]
    gllr = np.sum(gllr) * 2
    return gllr


# def fit_null_model(model, exp_design, pca_scores):

def get_sigma_sqr(obs_effect):
        # get sigma squared (variance) from redisuals and random factors
        obs_nullify_ix = [c for c in obs_effect[0].columns if "random effect:" in c or "residuals" in c]
        # obs_sigma = [np.var(orv[obs_nullify_ix]) for orv in obs_effect]
        obs_sigma = [np.std(orv[obs_nullify_ix]) for orv in obs_effect]
        obs_sigma = pd.concat(obs_sigma, axis=1)
        obs_nullify_ix = [o.replace('random effect: ', '') for o in obs_nullify_ix]
        obs_sigma.index = obs_nullify_ix
        return obs_sigma

def bootstrap_limmpca(h1_llf, models,
                      exp_design, scores, bootstrap_n=1000):

    boot_sample_size = data.shape[0]
    llf_collector = {}
    reduced_models = list(models.keys())
    reduced_models.remove("full_model")
    for nm in reduced_models:
        print(f"set up null model {nm}")
        # set up null model
        cur_null = models[nm]

        h0_models = ParallelMixedModel(cur_null)
        h0_models.fit(exp_design, scores)

        obs_gllr = calculate_gllr(h1_llf, h0_models.llf)
        print(f"true gllr: {obs_gllr}")

        obs_sigma = get_sigma_sqr(h0_models.effectmat)

        # boot strapping
        boot_gllrs = []
        bn = 0
        while len(boot_gllrs) < bootstrap_n:
            bn += 1
            print(f"Bootstrapping: {bn} / {bootstrap_n}")
            boot = resample(range(boot_sample_size), replace=True,
                            n_samples=boot_sample_size)
            est_scores = bootstrap_effect(obs_sigma, boot_sample_size, h0_models)
            print("restricted model")
            boot_restrict = parallel_mixed_modelling(cur_null,
                data.iloc[boot, :].reset_index(), est_scores[boot, :])
            print("full model")
            boot_full = parallel_mixed_modelling(models["full_model"],
                data.iloc[boot, :].reset_index(), est_scores[boot, :])

            boot_gllr = calculate_gllr(boot_full, boot_restrict)
            boot_gllrs.append(boot_gllr)
            print(f"bootstrap gllr: {obs_gllr}")

        boot_p = ( sum(boot_gllrs >= obs_gllr) + 1) / (bootstrap_n + 1)
        llf_collector[nm] = {"p-value": boot_p,
                             "observed global llr": obs_gllr,
                             "bootstrap global llr": boot_gllrs,
                             "restricted models llr": [m.llf for m in h0_models]}
    return llf_collector