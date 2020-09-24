import pandas as pd 
import numpy as np 

import statsmodels.api as sm
import statsmodels.formula.api as smf


def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 \
        - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)


def correct_scale(data, labels):
    # correct each subject by used scale range
    for id in np.unique(data.RIDNO):
        id_idx = data['RIDNO'].str.match(id)
        cur = data[id_idx].loc[:, labels].values
        scling = np.max(cur.flatten())
        floor = np.min(cur.flatten())
        cur = (cur - floor) / scling
        # update
        data[id_idx].loc[:, labels] = cur
    return data

def parallel_mixed_modelling(model, data, pca_scores):

    # add PC scores to the data frame
    m_components = pca_scores.shape[-1]
    for i in range(pca_scores.shape[-1]):
        data[f"factor_{i + 1}"] = pca_scores[:, i]

    # run the model
    h1_models = []
    for i in range(m_components):
        print(f"{i + 1} / {m_components}")
        mixed = smf.mixedlm(f"factor_{i + 1}"  + model["formula"],
                            data, 
                            groups=model["groups"],
                            re_formula=model["re_formula"],
                            vc_formula=model["vcf"])
        # fit the model
        mixed_fit = mixed.fit()
        # print(mixed_fit.summary())
        # save fitted model
        h1_models.append(mixed_fit)

    # get the design matrix for the fixed effect, random effect
    design = {
        "group_info": mixed.exog_re_li,
        "fe_mats": mixed.exog.copy(),
        "fe_names": mixed.exog_names,
        "vc_names": mixed.exog_vc.names,
        "vc_mats": mixed.exog_vc.mats,
    }
    return h1_models, design

def effect_matrix_decomposition(h1_models, design):

    def residuals(model): 
        resid = model.resid.copy()
        resid.name = "residual"
        resid = resid.reset_index(drop=True)
        return resid

    def fixed_effects(model, design):
        # fixed effect
        fe_params = model.fe_params
        mf = {}
        for j, name in enumerate(design["fe_names"]):
            coef = fe_params[name]
            tau_fe = design["fe_mats"][:, j]
            mf[name] = np.dot(coef, tau_fe)
        mf = pd.DataFrame(mf)
        return mf

    def random_effects(model, design):
        # random effects of the current factor
        # this is a dictionary of data frames one entry per group
        re_summary = model.random_effects
        mr = {}

        # handle re defined by variance component formula first
        tau_idx_multiple = []
        for mats, name in zip(design["vc_mats"], design["vc_names"]):
            mr_k = []
            cur_tau_index = []
            for k, g in enumerate(re_summary.keys()): # by subject
                tau_idx = [idx for idx in re_summary[g].index \
                    if name in idx]
                tau = re_summary[g][tau_idx].values
                z = mats[k]
                m_k = np.dot(z, tau)
                mr_k += list(m_k)
                # collect lable indicating effect with multuple levels
                if len(tau_idx) > len(cur_tau_index):
                    cur_tau_index += tau_idx
                else:
                    pass
            tau_idx_multiple += cur_tau_index
            # save the current matrix, matrix size should be N x 1
            mr["random effect: \n" + name] = np.array(mr_k)

        mr_k = []
        # effects with one or two level, 
        # not defined through variance component
        for k, g in enumerate(re_summary.keys()):
            group_eff = re_summary[g].index.difference(tau_idx_multiple)
            tau = re_summary[g][group_eff].values
            m_k = design["group_info"][k] * tau
            mr_k += list(m_k)
        mr_k = np.array(mr_k)
        for j, name in enumerate(group_eff):
            mr["random effect: \n" + name] = mr_k[:, j]
        mr = pd.DataFrame(mr)
        return mr
    
    m_components = len(h1_models)
    effect_mats = []
    for i in range(m_components):
        model = h1_models[i]
        mf_i = fixed_effects(model, design)
        mr_i = random_effects(model, design)
        resid_i = residuals(model)
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

def bootstrap_effect(obs_resid_sigmasqr, obs_rand_sigmasqr, 
                     boot_sample_size, h0_models, h0_design):

    def residuals(sigma, boot_sample_size): 
        boot_resid = np.random.normal(0, sigma, boot_sample_size)
        return boot_resid

    def fixed_eff(model, h0_design):
        obs_fixed = h0_design["fe_mats"].dot(model.fe_params)
        return obs_fixed

    def random_eff(model, h0_design, sigma):
        # random effects of the current factor
        # this is a dictionary of data frames one entry per group
        re_summary = model.random_effects
        mr = 0
        tau_idx_multiple = []
        # handle re defined by variance component formula first
        for mats, name in zip(h0_design["vc_mats"], h0_design["vc_names"]):
            mr_k = []
            for k, g in enumerate(re_summary.keys()): # by subject
                z = mats[k]
                tau_idx = f"random effect: \n{name}"
                tau_shape = z.shape[-1]
                tau = np.random.normal(0, sigma[tau_idx], tau_shape)
                m_k = np.dot(z, tau)
                mr_k += list(m_k)
                if tau_idx not in tau_idx_multiple:
                    tau_idx_multiple.append(tau_idx)
            # save the current matrix, matrix size should be N x 1
            mr += np.array(mr_k)

        mr_k = []
        # effects with one or two level, 
        # not defined through variance component
        for k, g in enumerate(re_summary.keys()):
            group_eff = sigma.index.difference(tau_idx_multiple)
            tau_shape = h0_design["group_info"][k].shape[1]
            tau = np.random.normal(0, sigma[group_eff], tau_shape)
            m_k = h0_design["group_info"][k] * tau
            mr_k += list(m_k)
        mr_k = np.array(mr_k)
        mr += np.sum(mr_k, axis=1)
        return mr
    
    m_components = len(h0_models)
    Y_ests = []
    for i in range(m_components):
        model = h0_models[i]
        sigma = obs_resid_sigmasqr[i]
        Y_est = residuals(sigma, boot_sample_size)
        Y_est += fixed_eff(model, h0_design) 
        sigma = obs_rand_sigmasqr[i]
        Y_est += random_eff(model, h0_design, sigma)
        Y_ests.append(Y_est)
    Y_ests = np.array(Y_ests).T
    return Y_ests