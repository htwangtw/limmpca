#%% load libraries
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from scipy.stats import zscore

import seaborn as sns
import matplotlib.pyplot as plt

from limmpca.util import correct_scale
from limmpca.mixedmodel import (parallel_mixed_modelling,
                                effect_matrix_decomposition,
                                variance_explained,)
from limmpca.bootstrap import bootstrap_limmpca

#%% load data and basic clean up

# use the top three for dev
n_components = 4
pca_varimax = "; raw"
varimax_on = False
bootstrap_n = 2000


project_path = Path.home() / "projects/lmm_pca/"
es_path = project_path / "data/task-nbackES_probes_trial_interval.tsv"
data = pd.read_csv(es_path, sep="\t")
# use the first cohort
data = data[data.IDNO < 500]

# drop rows with no data
data = data.dropna()
data["groups"] = 0  # nul model
data["nBack"] = data["nBack"].astype(int)
data["session"] = data["session"].astype(int)
data["intervals"] = zscore(data["interval"])

# normalise the scale to 0 -1 based on scale range used per individual
# labels = [c for c in data.columns if "MWQ" in c]
# I would like to have the labels ordered as follow.
labels = ['MWQ_Focus','MWQ_Future','MWQ_Past','MWQ_Self','MWQ_Other',
          'MWQ_Emotion','MWQ_Words', 'MWQ_Images',
          'MWQ_Deliberate','MWQ_Detailed','MWQ_Evolving','MWQ_Habit','MWQ_Vivid',]
data = correct_scale(data, labels)
data = data.reset_index(drop=True)
#%% naive PCA

# SPSS PCA was performed on correlation matrix
# so we z-score the input data
# X = data.loc[:, 'MWQ_Future':'MWQ_Deliberate'].values
# labels = labels
X = data.loc[:, labels].values
Xz = zscore(X)

pca = PCA(svd_solver='full').fit(Xz)
#%% scree plot
plt.figure()
plt.plot(pca.explained_variance_ratio_ * 100, "-o")
plt.xticks(ticks=range(13),
           labels=range(1, 14))
plt.ylabel("explained variance (%)")
plt.title("Scree plot")

plt.show()
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
    from limmpca.util import varimax
    pc = varimax(pc.T).T
    scores = np.dot(Xz, pc.T)
    pca_varimax = "; varimax"

plt.matshow(pc.T, cmap="RdBu_r", vmax=0.7, vmin=-0.7)
plt.xticks(ticks=range(n_components),
           labels=range(1, n_components + 1))
plt.yticks(ticks=range(len(labels)),
           labels=labels)
plt.title("Principle components" + pca_varimax)
plt.colorbar()
plt.show()

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
# lmer(factor_i ~  1 + C(nBack) * interval
#      + (1 + interval|C(RIDNO)/C(session)), data=data)
# i = 1, ...., m
# m is the number of componensts
models = {
    "full_model": {
        "formula": "~ 1 + C(nBack) * interval",
        "groups": "RIDNO",
        "re_formula": "1",
        "vcf": {"session": "0 + C(session)"}
        },
    "nBack": {
        "formula": "~ 1 + C(nBack)",
        "groups": "RIDNO",
        "re_formula": "1",
        "vcf": {"session": "0 + C(session)"}
        },
    "interval": {
        "formula": "~ 1 + interval",
        "groups": "RIDNO",
        "re_formula": "1",
        "vcf": {"session": "0 + C(session)"}
        },
    "RIDNO": {
        "formula": "~ 1 + C(nBack) * interval",
        "groups": "groups",
        "re_formula": "1",
        "vcf": {"session": "0 + C(session)"}
        },
    "session": {
        "formula": "~ 1 + C(nBack) * interval",
        "groups": "RIDNO",
        "re_formula": "1",
        "vcf": None
        },
}

#%% run true model
h1_models = parallel_mixed_modelling(models["full_model"], data, scores)

# effect matrix decomposition
# rewrite the LMM as effect matrix decomposition
effect_mats = effect_matrix_decomposition(h1_models)

# percentage of variance explained by variable
percent_var_exp = variance_explained(effect_mats)
#%%
sns.color_palette("tab10")
# plot results so far
chart = sns.barplot(x="Effect", y="variance(%)",
                    hue="PC", data=percent_var_exp
                    )
# chart.set_ylim(0, 1)
chart.set_xticklabels(
    chart.get_xticklabels(),
    rotation=45,
    horizontalalignment='right',
)
plt.title("Variance of components" + pca_varimax)
plt.show()

#%% bootstrapping
llf_collector = bootstrap_limmpca(h1_models, models,
                                  data, scores, bootstrap_n)

#%% save results
try:
	llf_file = open(project_path / 'results/llf_bootstrap_results.pkl', 'wb')
	pickle.dump(llf_collector, llf_file)
	llf_file.close()

except:
	print("Something went wrong")

#%% visualise patterns explained
# #%% Back transpose PCA - interval
# cur_eff = np.array([mat["interval"].values for mat in effect_mats]).T
# weighted_scores = cur_eff.dot(pc)
# cur_pca = PCA(svd_solver='full').fit(weighted_scores)

# plt.plot(cur_pca.explained_variance_ratio_)
# plt.show()

# transpos_back_pca = cur_pca.components_[:1, :]
# plt.matshow(transpos_back_pca.T, cmap="RdBu_r")
# plt.yticks(ticks=range(13),
#            labels=data.loc[:, labels].columns)
# plt.title("Principle components" + pca_varimax)
# plt.colorbar()
# plt.show()

# cur_scores = cur_pca.transform(weighted_scores)[:, :1]
# #%% Back transpose PCA - nback
# cur_eff = np.array([mat["C(nBack)[T.1]"].values for mat in effect_mats]).T
# weighted_scores = cur_eff.dot(pc)
# cur_pca = PCA(svd_solver='full').fit(weighted_scores)

# plt.plot(cur_pca.explained_variance_ratio_)
# plt.show()

# transpos_back_pca = cur_pca.components_[:1, :]
# plt.matshow(transpos_back_pca.T, cmap="RdBu_r")
# plt.yticks(ticks=range(13),
#            labels=data.loc[:, labels].columns)
# plt.title("Principle components" + pca_varimax)
# plt.colorbar()
# plt.show()

# cur_scores = cur_pca.transform(weighted_scores)[:, :1]

# #%% Back transpose PCA - nback random intercept
# eff = np.array([mat["random effect: \nsession"].values for mat in effect_mats]).T
# cur_pca = PCA(svd_solver='full').fit(eff.dot(pc))
# plt.plot(cur_pca.explained_variance_ratio_)
# plt.show()

# transpos_back_pca = cur_pca.components_.dot(pc)
# plt.matshow(transpos_back_pca[:2, :].T, cmap="RdBu_r")
# plt.yticks(ticks=range(13),
#            labels=data.loc[:, 'MWQ_Focus':'MWQ_Deliberate'].columns)
# plt.title("Principle components" + pca_varimax)
# plt.colorbar()
# plt.show()
# #%% Back transpose PCA - sample
# eff = np.array([mat["random effect: \nRIDNO"].values for mat in effect_mats]).T
# cur_pca = PCA(svd_solver='full').fit(eff.dot(pc))
# plt.plot(cur_pca.explained_variance_ratio_)
# plt.show()

# transpos_back_pca = cur_pca.components_.dot(pc)
# plt.matshow(transpos_back_pca[:2, :].T, cmap="RdBu_r")
# plt.yticks(ticks=range(13),
#            labels=data.loc[:, 'MWQ_Focus':'MWQ_Deliberate'].columns)
# plt.title("Principle components" + pca_varimax)
# plt.colorbar()
# plt.show()
# #%% Back transpose PCA - nback random intercept
# eff = np.array([mat["random effect: \nC(nBack)[T.1]"].values for mat in effect_mats]).T
# cur_pca = PCA(svd_solver='full').fit(eff.dot(pc))
# plt.plot(cur_pca.explained_variance_ratio_)
# plt.show()

# transpos_back_pca = cur_pca.components_.dot(pc)
# plt.matshow(transpos_back_pca[:2, :].T, cmap="RdBu_r")
# plt.yticks(ticks=range(13),
#            labels=data.loc[:, 'MWQ_Focus':'MWQ_Deliberate'].columns)
# plt.title("Principle components" + pca_varimax)
# plt.colorbar()
# plt.show()

# #%% Back transpose PCA - residual
# eff = np.array([mat["residual"].values for mat in effect_mats]).T
# cur_pca = PCA(svd_solver='full').fit(eff.dot(pc))
# plt.plot(cur_pca.explained_variance_ratio_)
# plt.show()

# transpos_back_pca = cur_pca.components_.dot(pc)
# plt.matshow(transpos_back_pca[:2, :].T, cmap="RdBu_r")
# plt.yticks(ticks=range(13),
#            labels=data.loc[:, 'MWQ_Focus':'MWQ_Deliberate'].columns)
# plt.title("Principle components" + pca_varimax)
# plt.colorbar()
# plt.show()

