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
                  variance_explained, correct_scale)

#%% load data and basic clean up

# use the top three for dev
n_components = 13
pca_varimax = "; raw"
varimax_on = False

es_path = "data/CS_MWQ_LabOnlineThoughtProbesScores_rescaled.csv"
task_path = "data/CS_Tasks_withEFF.csv"
task_labels = ["RNO", "MWQ_WM.ACC", "MWQ_WM.RT", "MWQ_CRT.ACC", "MWQ_CRT.RT"]
new_labels = ["RIDNO", "oneBackAcc", "oneBackRT", "zeroBackAcc", "zeroBackRT"]
data = pd.read_csv(es_path)
task = pd.read_csv(task_path)[task_labels]
task.columns = new_labels

# drop rows with no data
data = data.dropna()
data["group"] = 0  # nul model

# add behavioural task performace
data["Acc"] = 0
data["RT"] = 0
data["nBack"] = data["nBack"].astype(int)

for id in np.unique(data.RIDNO).tolist():
    cur_task = task[task['RIDNO'].str.match(id)]
    for b, con in enumerate(["zero", "one"]):
        id_idx = data.query(f"nBack=={b} & RIDNO=='{id}'").index
        for t in ["Acc", "RT"]:
            data.loc[id_idx, t] = cur_task[f"{con}Back{t}"].values

# use the first cohort
data = data[data.IDNO < 500]

# normalise the scale to 0 -1 based on scale range used per individual
data = correct_scale(data)

#%% naive PCA

# SPSS PCA was performed on correlation matrix 
# so we z-score the input data
X = data.loc[:, 'MWQ_Focus':'MWQ_Deliberate'].values
Xz = zscore(X)

pca = PCA(svd_solver='full').fit(Xz)

# calculate principle component scores
scores = pca.transform(Xz)[:, :n_components]
pc = pca.components_[:n_components, :].T
if varimax_on:
    from util import varimax
    pc = varimax(pc)
    scores = np.dot(Xz, pc)
    pca_varimax = "; varimax"

#%% parallel mixed modelling - formula method

# define model
# this nested model can be recreated in R as follow
# lmer(factor_i ~  1 + C(nBack) + (1 + C(nBack)|C(RIDNO)/C(session)), data=data) 
# i = 1, ...., m 
# m is the number of componensts
model = {
    "formula": "~ 1 + C(nBack)",  
    "groups": "RIDNO",
    "re_formula": "1 + C(nBack)",  # fit random intercept (1) and slope (C(nBack)) 
    "vcf": {"session": "0 + C(session)"}  # nested random effect
}

h1_models, design = parallel_mixed_modelling(model, data, scores)

#%% effect matrix decomposition
# rewrite the LMM as effect matrix decomposition
effect_mats = effect_matrix_decomposition(h1_models, design)

#%% percentage of variance explained by variable
percent_var_exp = variance_explained(effect_mats)

#%% plot results so far
sns.barplot(x="Effect", y="variance(%)", hue="PC",
            data=percent_var_exp,
            )
plt.title("Variance of components" + pca_varimax)
plt.show()

plt.matshow(pc, cmap="RdBu_r")
plt.xticks(ticks=range(n_components), 
           labels=range(1, n_components + 1))
plt.yticks(ticks=range(13),
           labels=data.loc[:, 'MWQ_Focus':'MWQ_Deliberate'].columns)
plt.title("Principle components" + pca_varimax)
plt.colorbar()
plt.show()


#%% significance testing

null_models = {
    "nBack": {
        "formula": f"factor_{i + 1} ~ 1"
        "group": "RIDNO",
        "re_formula": "1",
        "vcf": {"session": "0 + C(session)"}
        },
    "RIDNO": {
        "formula": f"factor_{i + 1} ~ 1 + C(nBack)"
        "group": "session",
        "re_formula": "1 + C(nBack)",
        "vcf": None
        },
    "session": {
        "formula": f"factor_{i + 1} ~ 1 + C(nBack)"
        "group": "RIDNO",
        "vcf": None
        },
}

# run the null model per factor, bootstrap=1000


#%% significant testing based on the null
