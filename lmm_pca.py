import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from scipy.stats import zscore
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from util import varimax

data_path = "data/CS_MWQ_LabOnlineThoughtProbesScores_rescaled.csv"
data = pd.read_csv(data_path)

print(np.unique(data.RIDNO).shape)

# drop rows with no data
data = data.dropna()
print(np.unique(data.RIDNO).shape)

# SPSS PCA was performed on correlation matrix 
# so we z-score the input data
clean_data = zscore(data.iloc[:, 4:].values)

# do pca
pca = PCA(svd_solver='full').fit(clean_data)

# varimax rotation
vm = varimax(pca.components_.T * -1)
# plot output
plt.matshow(vm, cmap="RdBu")
plt.yticks(ticks=range(13), labels=data.columns[4:])

# calculate principle component scores
scores = clean_data.dot(vm)
for i in range(scores.shape[-1]):
    data[f"factor_{i + 1}"] = scores[:, i]
# data.iloc[:, 4:17] = data.iloc[:, 4:17].apply(zscore)

# do lmm on each pc that take experiment design into account
data["groups"] = 0  # we have only one group

# variance component to specify the design
vcf = {"RIDNO": "0 + C(RIDNO)",  # participant ID
       "nBack": "0 + C(nBack)",  # two expeirmental conditions: 0-back and 1-back
       "session": "0 + C(session)"}  # time - not expecting much 

model_parameters = []
residual_matrix = []
randome_effect = []
for i in range(scores.shape[-1]):
    # model formula, intercept as fixed effec
    model_formula = f"factor_{i + 1} ~ 1"

    # fit the model
    # this is the same as the following model in R
    # lmer(factor_i ~ (1|RIDNO) + (1|nBack) + (1|session), data=data) 
    # i = 1, ...., m 
    # m is the number of componensts
    mixed = smf.mixedlm(model_formula, data, 
                        groups="groups", vc_formula=vcf)
    mixed_fit = mixed.fit()
    # save model parameters
    model_parameters.append(mixed_fit.params)
    residual_matrix.append(mixed_fit.resid)
    randome_effect.append(mixed_fit.random_effects)
    # print(mixed_fit.summary())

vc = {}
for i, name in enumerate(mixed.exog_vc.names):
    vc[name] = pd.DataFrame(mixed.exog_vc.mats[i][0], 
                            columns=mixed.exog_vc.colnames[i][0])

# rewrite the LMM as effect matrix decomposition
Mf = []
Mr = []
for i in range(scores.shape[-1]):
    mf = np.ones((data.shape[0], 1)).dot(model_parameters[i].Intercept)
    Mf.append(np.squeeze(mf))

    # get the estimate by variable just in case I need to break it down later
    re = randome_effect[i][0]
    mr = np.zeros((data.shape[0]))
    for name in vc.keys():
        idx_name = [f"{name}[{cn}]" for cn in vc[name].columns]
        mat = vc[name]
        tau_re = re[idx_name]
        mr += np.dot(mat, tau_re)
    Mr.append(mr)

# quantify of effects importance with percent variance explained by each factor

# significance testing

# visual representation of the effect matrices
