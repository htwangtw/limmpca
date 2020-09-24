#%% load libraries
import pandas as pd 
import numpy as np 

from sklearn.decomposition import PCA

from scipy.stats import zscore
from scipy import linalg

import seaborn as sns 
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from util import (parallel_mixed_modelling, effect_matrix_decomposition, 
                  variance_explained, correct_scale, bootstrap_effect)

#%% load data and basic clean up

# use the top three for dev
n_components = 2
pca_varimax = "; raw"
varimax_on = False

es_path = "data/task-nbackES_probes_trial_interval.tsv"
data = pd.read_csv(es_path, sep="\t")

# drop rows with no data
data = data.dropna()
data["group"] = 0  # nul model
data["nBack"] = data["nBack"].astype(int)
data["session"] = data["session"].astype(int)

# use the first cohort
data = data[data.IDNO < 500]

# normalise the scale to 0 -1 based on scale range used per individual
labels = [c for c in data.columns if "MWQ" in c]
data = correct_scale(data, labels)
data = data.reset_index(drop=True)
#%% naive PCA

# SPSS PCA was performed on correlation matrix 
# so we z-score the input data
X = data.loc[:, labels].values
# X = data.loc[:, 'MWQ_Future':'MWQ_Deliberate'].values
Xz = zscore(X)

pca = PCA(svd_solver='full').fit(Xz)
#%% scree plot
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, "-o")
plt.xticks(ticks=range(13),
           labels=range(1, 14))
#%%
# calculate principle component scores
if n_components:
    scores = pca.transform(Xz)[:, :n_components]
    pc = pca.components_[:n_components, :]
else:
    scores = pca.transform(Xz)
    pc = pca.components_ 
    n_components = pc.shape[0]

if varimax_on:
    from util import varimax
    pc = varimax(pc.T).T
    scores = np.dot(Xz, pc.T)
    pca_varimax = "; varimax"

pca_res = data.loc[:,:].copy()
m_components = scores.shape[-1]
for i in range(scores.shape[-1]):
    pca_res[f"factor_{i + 1}"] = scores[:, i]
#%%
# plt.figure()
# sns.scatterplot("factor_2", "factor_4", 
#                 data=pca_res.loc[:90,:], hue="nBack")
# plt.figure()
# sns.scatterplot("factor_2", "factor_1", 
#                 data=pca_res.loc[:90,:], hue="RIDNO")
#%% parallel mixed modelling - formula method

# define model
# this nested model can be recreated in R as follow
# lmer(factor_i ~  1 + C(nBack) + (1 + C(nBack)|C(RIDNO)/C(session)), data=data) 
# i = 1, ...., m 
# m is the number of componensts
model = {
    "formula": "~ 1 + C(nBack) + interval",  
    "groups": "RIDNO",
    "re_formula": "1 + C(nBack) + interval",  # fit random intercept (1) and slope (C(nBack)) 
    "vcf": {"session": "0 + C(session)"}  # nested random effect
}
# model = {
#     "formula": "~ 1 + C(nBack) : MWQ_Focus",  
#     "groups": "RIDNO",
#     "re_formula": "1 + C(nBack) : MWQ_Focus",  # fit random intercept (1) and slope (C(nBack)) 
#     "vcf": {"session": "0 + C(session)"}  # nested random effect
# }

h1_models, design = parallel_mixed_modelling(model, data, scores)

# effect matrix decomposition
# rewrite the LMM as effect matrix decomposition
effect_mats = effect_matrix_decomposition(h1_models, design)

# percentage of variance explained by variable
percent_var_exp = variance_explained(effect_mats)

#%% plotting
# plot results so far
chart = sns.barplot(x="Effect", y="variance(%)", 
                    hue="PC", data=percent_var_exp
                    )
chart.set_xticklabels(
    chart.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
)
plt.title("Variance of components" + pca_varimax)
plt.show()

plt.matshow(pc.T, cmap="RdBu_r")
plt.xticks(ticks=range(n_components), 
           labels=range(1, n_components + 1))
# plt.yticks(ticks=range(12),
#            labels=data.loc[:, 'MWQ_Future':'MWQ_Deliberate'].columns)
plt.yticks(ticks=range(13),
           labels=data.loc[:, labels].columns)
plt.title("Principle components" + pca_varimax)
plt.colorbar()
plt.show()

#%% significance testing
null_models = {
    "nBack": {
        "formula": "~ 1 + interval",
        "groups": "RIDNO",
        "re_formula": "1 + interval",
        "vcf": {"session": "0 + C(session)"}
        },
    "interval": {
        "formula": "~ 1 + C(nBack)",
        "groups": "RIDNO",
        "re_formula": "1 + C(nBack)",
        "vcf": {"session": "0 + C(session)"}
        },
    "RIDNO": {
        "formula": "~ 1 + C(nBack) + interval",
        "groups": "session",
        "re_formula": "1 + C(nBack) + interval",
        "vcf": None
        },
    "session": {
        "formula": "~ 1 + C(nBack) + interval",
        "groups": "RIDNO",
        "re_formula": "1 + C(nBack) + interval",
        "vcf": None
        },
}

#%% bootstrapping
bootstrap_n = 100
# scikit-learn bootstrap
from sklearn.utils import resample
def calculate_gllr(h1_res, h0_res):
    gllr = [h1_res[i].llf - h0_res[i].llf 
                for i in range(len(h1_res))]
    gllr = np.sum(gllr)
    return 2 * gllr

boot_sample_size = data.shape[0]


llf_collector = {}

#%% setup null
for nm in null_models.keys():
    print(nm)
    print("set up null model")
    # set up null model
    cur_null = null_models[nm] 
    h0_models, h0_design = parallel_mixed_modelling(cur_null, 
                data, scores)

    obs_effect = effect_matrix_decomposition(h0_models, h0_design)

    obs_gllr = calculate_gllr(h1_models, h0_models)

    obs_resid_sigmasqr = [np.var(orv["residual"]) for orv in obs_effect]

    obs_rand_label = [c for c in obs_effect[0].columns if "random effect:" in c]
    obs_rand_sigmasqr = [np.var(orv[obs_rand_label]) for orv in obs_effect]

    # boot strapping
    boot_gllrs = []
    bn = 0
    while len(boot_gllrs) < bootstrap_n:
        bn += 1
        print(f"Bootstrapping: {bn} / {bootstrap_n}")
        boot = resample(range(boot_sample_size), replace=True, 
                        n_samples=boot_sample_size)
        
        est_scores = bootstrap_effect(obs_resid_sigmasqr, obs_rand_sigmasqr, 
                        boot_sample_size, h0_models, h0_design)
        est_scores = np.array(est_scores).T
        print("restricted model")
        boot_restrict, _ = parallel_mixed_modelling(cur_null, 
            data.iloc[boot, :].reset_index(), est_scores[boot, :])
        print("full model")
        boot_full, _ = parallel_mixed_modelling(model, 
            data.iloc[boot, :].reset_index(), est_scores[boot, :])

        boot_gllr = calculate_gllr(boot_full, boot_restrict)
        boot_gllrs.append(boot_gllr)

    boot_p = ( sum(boot_gllrs >= obs_gllr) + 1) / (bootstrap_n + 1)
    llf_collector[nm] = {"p-value": boot_p, 
                         "observed gllr": obs_gllr,
                         "restricted model llr": [m.llr for m in h0_models]}

#%% Back transpose PCA - nback
cur_eff = np.array([mat["C(nBack)[T.1]"].values for mat in effect_mats]).T
weighted_scores = cur_eff.dot(pc)
cur_pca = PCA(svd_solver='full').fit(weighted_scores)

plt.plot(cur_pca.explained_variance_ratio_)
plt.show()

transpos_back_pca = cur_pca.components_[:1, :]
plt.matshow(transpos_back_pca.T, cmap="RdBu_r")
plt.yticks(ticks=range(13),
           labels=data.loc[:, labels].columns)
plt.title("Principle components" + pca_varimax)
plt.colorbar()
plt.show()

cur_scores = cur_pca.transform(weighted_scores)[:, :1]

#%% Back transpose PCA - nback random intercept
eff = np.array([mat["random effect: \nsession"].values for mat in effect_mats]).T
cur_pca = PCA(svd_solver='full').fit(eff.dot(pc))
plt.plot(cur_pca.explained_variance_ratio_)
plt.show()

transpos_back_pca = cur_pca.components_.dot(pc)
plt.matshow(transpos_back_pca[:2, :].T, cmap="RdBu_r")
plt.yticks(ticks=range(13),
           labels=data.loc[:, 'MWQ_Focus':'MWQ_Deliberate'].columns)
plt.title("Principle components" + pca_varimax)
plt.colorbar()
plt.show()
#%% Back transpose PCA - sample
eff = np.array([mat["random effect: \nRIDNO"].values for mat in effect_mats]).T
cur_pca = PCA(svd_solver='full').fit(eff.dot(pc))
plt.plot(cur_pca.explained_variance_ratio_)
plt.show()

transpos_back_pca = cur_pca.components_.dot(pc)
plt.matshow(transpos_back_pca[:2, :].T, cmap="RdBu_r")
plt.yticks(ticks=range(13),
           labels=data.loc[:, 'MWQ_Focus':'MWQ_Deliberate'].columns)
plt.title("Principle components" + pca_varimax)
plt.colorbar()
plt.show()
#%% Back transpose PCA - nback random intercept
eff = np.array([mat["random effect: \nC(nBack)[T.1]"].values for mat in effect_mats]).T
cur_pca = PCA(svd_solver='full').fit(eff.dot(pc))
plt.plot(cur_pca.explained_variance_ratio_)
plt.show()

transpos_back_pca = cur_pca.components_.dot(pc)
plt.matshow(transpos_back_pca[:2, :].T, cmap="RdBu_r")
plt.yticks(ticks=range(13),
           labels=data.loc[:, 'MWQ_Focus':'MWQ_Deliberate'].columns)
plt.title("Principle components" + pca_varimax)
plt.colorbar()
plt.show()

#%% Back transpose PCA - residual
eff = np.array([mat["residual"].values for mat in effect_mats]).T
cur_pca = PCA(svd_solver='full').fit(eff.dot(pc))
plt.plot(cur_pca.explained_variance_ratio_)
plt.show()

transpos_back_pca = cur_pca.components_.dot(pc)
plt.matshow(transpos_back_pca[:2, :].T, cmap="RdBu_r")
plt.yticks(ticks=range(13),
           labels=data.loc[:, 'MWQ_Focus':'MWQ_Deliberate'].columns)
plt.title("Principle components" + pca_varimax)
plt.colorbar()
plt.show()

