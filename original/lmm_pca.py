#%% load libraries
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from scipy.stats import zscore

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.extmath import weighted_mode

from limmpca.util import correct_scale
from limmpca.model import (parallel_mixed_model,
    variance_explained)


#%% load data and basic clean up

# use the top three for dev
n_components = 4
pca_varimax = "; raw"
varimax_on = False
bootstrap_n = 2000

# set random seed
np.random.seed(42)

project_path = Path.home() / "projects/limmpca/"
es_path = project_path / "data/task-nbackES_probes_trial_interval.tsv"
exp_design = pd.read_csv(es_path, sep="\t", index_col=0)
# use the first cohort
exp_design = exp_design[exp_design.IDNO < 500]
exp_design = exp_design[exp_design.session < 3]

# drop rows with no data
exp_design = exp_design.dropna()
# standardise everything
exp_design["groups"] = 0  # nul model
exp_design["nBack"] = exp_design["nBack"].astype(int)
exp_design["session"] = exp_design["session"].astype(int)
exp_design["interval"] = zscore(exp_design["interval"])

# normalise the scale to 0 -1 based on scale range used per individual
# labels = [c for c in data.columns if "MWQ" in c]
# I would like to have the labels ordered as follow.
labels = ['MWQ_Focus','MWQ_Future','MWQ_Past','MWQ_Self','MWQ_Other',
          'MWQ_Emotion','MWQ_Words', 'MWQ_Images',
          'MWQ_Deliberate','MWQ_Detailed','MWQ_Evolving','MWQ_Habit','MWQ_Vivid',]
display = ['Task','Future','Past','Self','People',
          'Emotion','Words', 'Images',
          'Deliberate','Detailed','Evolving','Habit','Vivid',]

exp_design = correct_scale(exp_design, labels)
exp_design = exp_design.reset_index(drop=True)
#%% naive PCA

# SPSS PCA was performed on correlation matrix
# so we z-score the input data
# X = exp_design.loc[:, 'MWQ_Future':'MWQ_Deliberate'].values
# labels = labels
X = exp_design.loc[:, labels].values
Xz = zscore(X)

pca = PCA(svd_solver='full').fit(Xz)

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

m_components = scores.shape[-1]

#%% scree plot
plt.figure()
plt.plot(pca.explained_variance_ratio_ * 100, "-o")
plt.xticks(ticks=range(13),
           labels=range(1, 14))
plt.ylabel("explained variance (%)")
plt.title("Scree plot")

plt.show()

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, "-o")
plt.xticks(ticks=range(13),
           labels=range(1, 14))
plt.ylabel("explained variance (%)")
plt.title("Cumulated variance explained")

plt.show()

# loading plot
plt.figure()
plt.plot(pc[0, :], pc[1, :], ".")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
for i in range(13):
    plt.annotate(display[i].split("_")[-1],
                 (pc[0, i], pc[1, i]), )
#%% pc loadings
plt.matshow(pc.T, cmap="RdBu_r", vmax=0.7, vmin=-0.7)
plt.xticks(ticks=range(n_components),
           labels=range(1, n_components + 1))
plt.yticks(ticks=range(len(display)),
           labels=display)
plt.title("Principle components" + pca_varimax)
plt.colorbar()
plt.show()

#%% parallel mixed modelling - formula method
# define model
# this nested model can be recreated in R as follow
# lmer(factor_i ~  1 + C(nBack) + interval
#      + (1 | C(RIDNO)), data=data)
# i = 1, ...., m
# m is the number of componensts
models = {
    "full_model": {'formula': '~ 1 + C(nBack) + interval',
    'groups': 'RIDNO',
    're_formula': '1 ',
    'vc_formula': None},
    "nBack": {
        "formula": "~ 1 + interval",
        "groups": "RIDNO",
        "re_formula": "1",
        "vc_formula": None
        },
    "interval": {
        "formula": "~ 1 + C(nBack)",
        "groups": "RIDNO",
        "re_formula": "1",
        "vc_formula": None
        },
    "RIDNO": {
        "formula": "~ 1 + C(nBack) + interval",
        "groups": "groups",
        "re_formula": "1",
        "vc_formula": None
        },
}

#%% run true model
llf, effectmat = parallel_mixed_model(models["full_model"],
    exp_design, scores)

percent_var_exp = variance_explained(effectmat)

#%% bootstrapping - need work
# llf_collector = bootstrap_limmpca(h1_models, models,
#                                   data, scores, bootstrap_n)

# #%% save results
# try:
# 	llf_file = open(project_path / 'results/llf_bootstrap_results_test.pkl', 'wb')
# 	pickle.dump(llf_collector, llf_file)
# 	llf_file.close()

# except:
# 	print("Something went wrong")

# #%% load bootstrap results
# llf_file = open(project_path / 'results/llf_bootstrap_results_interaction.pkl', 'rb')
# llf_collector = pickle.load(llf_file)
# llf_file.close()
# #%% visualise bootstrap results

# restricted_llr = []
# true_llf = [m.llf for m in h1_models]
# for k in llf_collector.keys():
#     boot_llr = llf_collector[k]["bootstrap global llr"]
#     p_value = llf_collector[k]["p-value"]
#     obs_llf = llf_collector[k]["observed global llr"]
#     rest_llf = llf_collector[k]["restricted models llr"]
#     plt.figure()
#     dist = sns.distplot(boot_llr, kde=False)
#     plt.title(k)
#     plt.vlines(obs_llf, 0, dist.get_ylim()[-1] + 5,
#                linestyles="--")
#     plt.savefig(project_path / f'results/bootstrap_{k}.png')

#     df = pd.DataFrame({"(R)LLR": 2 * (np.array(true_llf) - np.array(rest_llf)),
#                        "Removed factor": [k] * 13,
#                        "Principle component": list(range(1, 14))})

#     restricted_llr.append(df)
#     print(p_value)

# restricted_llr = pd.concat(restricted_llr)
# rllr_plot = sns.catplot(x="Principle component", y="(R)LLR", hue="Removed factor",
#             data=restricted_llr, kind="bar")
# rllr_plot.savefig(project_path / 'results/rllr.png')

# potential idea:
# if the model isolated out the effect of the experiment,
# would it make sense to explore how much other participant trait level
# measures are explaining the residual?

#%% visualise pure patterns explained
effect_names = effectmat[0].columns.tolist()[1:]
weighted_pca={}
for name in effect_names:
    cur_eff = np.array([mat[name].values for mat in effectmat]).T
    weighted_scores = cur_eff.dot(pc)
    cur_pca = PCA(svd_solver='full').fit(weighted_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(cur_pca.explained_variance_ratio_)
    ax1.set_title(name)

    if "random" in name or "residuals" in name:
        transpos_back_pca = cur_pca.components_[:2, :].T
        ax2.plot(transpos_back_pca[:, 0], transpos_back_pca[:, 1], ".")
        ax2.set_xlabel("PC 1")
        ax2.set_ylabel("PC 2")
        for i in range(13):
            ax2.annotate(labels[i].split("_")[-1],
                        (transpos_back_pca[i, 0] + 0.01,
                         transpos_back_pca[i, 1]), )
    else:
        transpos_back_pca = cur_pca.components_[:1, :].T
        print(transpos_back_pca.shape)
        weighted_pca[name] = transpos_back_pca
        ax2.matshow(transpos_back_pca, cmap="RdBu_r")
        ax2.set_yticks(range(13))
        ax2.set_yticklabels(display)
        ax2.set_xticklabels("")
        # ax2.colorbar()
    ax2.set_title(f"Pure principle components")
    plt.show()

    cur_scores = cur_pca.transform(weighted_scores)[:, :1]
    # save the score for gradient analysis
# %% project these to the data
factors = exp_design.copy()
factors["factor_nBack"] = Xz.dot(weighted_pca["nBack"])
factors["factor_interval"] = Xz.dot(weighted_pca["interval"])
for i in range(scores.shape[-1]):
    factors[f"factor_{i + 1}"] = scores[:, i]
original = [f"factor_{i + 1}" for i in range(n_components)]
factors_noCond_split = factors.pivot_table(
    values=["factor_nBack", "factor_interval", "interval"] + original ,
    index=['RIDNO', 'groups'])
factors_noCond_split = factors_noCond_split.reset_index().set_index("RIDNO")

factors_0back = factors[factors["nBack"] == 0].pivot_table(
    values=["factor_nBack", "factor_interval"] + original,
    index=['nBack', 'RIDNO', 'IDNO', 'groups'])
factors_0back = factors_0back.rename(
    columns={"factor_nBack": "nback0_factor_nBack",
            "factor_interval": "nback0_factor_interval",
            "factor_1": "nback0_factor_1",
            "factor_2": "nback0_factor_2",
            "factor_3": "nback0_factor_3",
            "factor_4": "nback0_factor_4",
    })
factors_0back = factors_0back.reset_index().set_index("RIDNO")

factors_1back = factors[factors["nBack"] == 1].pivot_table(
    values=["factor_nBack", "factor_interval"] + original,
    index=['nBack', 'RIDNO', 'IDNO', 'groups'])
factors_1back = factors_1back.rename(
    columns={"factor_nBack": "nback1_factor_nBack",
            "factor_interval": "nback1_factor_interval",
            "factor_1": "nback1_factor_1",
            "factor_2": "nback1_factor_2",
            "factor_3": "nback1_factor_3",
            "factor_4": "nback1_factor_4",
    })
factors_1back = factors_1back.reset_index().set_index("RIDNO")

factors_limm = pd.concat([factors_noCond_split, factors_0back, factors_1back],
    axis=1)
factors_limm = factors_limm.loc[:,~factors_limm.columns.duplicated()]

# gradient
gradients = pd.read_csv("data/gradients.csv")
gradients = gradients.rename(columns={"Rnumber": "RIDNO"})
gradients = gradients.set_index("RIDNO").drop_duplicates()

master = pd.concat([gradients, factors_limm], axis=1)
master = master.dropna()

master.to_csv(f"results/limmpca_factors_top{n_components}.tsv", sep="\t")