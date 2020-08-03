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

from util import varimax, correct_scale

#%% load data and basic clean up

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

# calculate principle component scores; use the top three for dev
scores = Xz.dot(pca.components_[:1, :].T)
for i in range(scores.shape[-1]):
    data[f"factor_{i + 1}"] = scores[:, i]

#%% parallel mixed modelling - formula method

# define model
# this nested model can be recreated in R as follow
# lmer(factor_i ~  1 + C(nBack) + (1 + C(nBack)|C(RIDNO)/C(session)), data=data) 
# i = 1, ...., m 
# m is the number of componensts

model = {
    "formula": f"~ 1 + C(nBack)",  
    "groups": "RIDNO",
    "re_formula": "1 + C(nBack)",  # fit random intercept (1) and slope (C(nBack)) 
    "vcf": {"session": "0 + C(session)"}  # nested random effect
}

model_parameters = []
residual_matrix = []
randome_effect = []
m_components = scores.shape[-1]
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

#%% effect matrix decomposition
# rewrite the LMM as effect matrix decomposition

# get the design matrix for the fixed effect (one vs zero back)
var = mixed.exog_re.copy()
fixed_var = mixed.exog_names # save the real fixed effect var names

# get the design matrix for the random effect
re_names = mixed.exog_vc.names
re_mats = mixed.exog_vc.mats
assert len(re_names) == len(re_mats)
group_info = mixed.exog_re_li

# calculat the effect matrix
Mf = []
Mr = []
residual_matrix = []
for i in range(m_components):
    model_parameters = h1_models[i].params

    # fixed effect
    fe = h1_models[i].fe_params
    mf_j = []
    for j, coef in enumerate(fe):
        tau_fe = var[:, j]
        mf_j.append(np.dot(coef, tau_fe))
    Mf.append(np.array(mf_j))

    # random effect
    randome_effect = h1_models[i].random_effects
    mr_i = []
    for mats, name in zip(re_mats, re_names):
        print(name)
        mr_k = []
        for k, g in enumerate(randome_effect.keys()): # by subject
            # handle re defined by variance component formula first
            tau_idx = [idx for idx in randome_effect[g].index \
                if name in idx]

            # effect with multiple level
            tau = randome_effect[g][tau_idx].values
            z = mats[k]
            m = np.dot(z, tau)
            mr_k += list(m)
        
        # effects with one or two level
        group_eff = randome_effect[g].index.difference(tau_idx)
        tau = randome_effect[g][group_eff].values
        m = group_info[k] * tau
        mr_k += list(m)

        mr_i.append(np.array(mr_k))
    Mr.append(mr_i)

    # residual
    residual_matrix.append(h1_models[i].resid)

#%% quantify of effects importance with percent variance explained by each factor
est_var_resid = np.var(np.array(residual_matrix), axis=1)  # redisuals

est_var_random_j = []
for mr_i in Mr:
    est_var_random_j.append(np.var(mr_i, axis=1))

est_var_fixed_j = []
for mf_i in Mf:
    est_var_fixed_j.append(np.var(mf_i, axis=1))

est_var_full = np.sum(est_var_fixed_j, axis=1) \
    + np.sum(est_var_random_j, axis=1) \
    + est_var_resid

# plot variace full model
percent_var_exp = np.vstack((
    np.array(est_var_fixed_j).T, 
    np.array(est_var_random_j).T,
    est_var_resid)) / est_var_full * 100
percent_var_exp = pd.DataFrame(percent_var_exp,
                               columns=np.arange(1, m_components + 1),
                               index=fixed_var + re_names + ["residual"])
percent_var_exp["Effect"] = percent_var_exp.index
percent_var_exp = percent_var_exp.melt(id_vars="Effect", value_name="variance(%)", var_name="PC")

# ignore intercept when plotting
sns.barplot(x="Effect", y="variance(%)", hue="PC",
            data=percent_var_exp[percent_var_exp["Effect"]!="Intercept"])
plt.title("Variance of components")
plt.show()

#%% significance testing

null_models = {
    "nBack": {
        "formula": f"factor_{i + 1} ~ 1"
        "group": "RIDNO",
        "vcf": {"session": "0 + C(session)", "RIDNO": "0 + C(RIDNO)"}
        },
    "RIDNO": {
        "formula": f"factor_{i + 1} ~ 1 + C(nBack)"
        "group": "group",
        "vcf": {"session": "0 + C(session)"}
        },
    "session": {
        "formula": f"factor_{i + 1} ~ 1 + C(nBack)"
        "group": "RIDNO",
        "vcf": {"RIDNO": "0 + C(RIDNO)"}
        },
}

# run the null model


#%% visual representation of the effect matrices
