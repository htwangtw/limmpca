#%% load libraries
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from scipy.stats import zscore

import seaborn as sns
import matplotlib.pyplot as plt

from limmpca.model import ParallelMixedModel

# %%
data = pd.read_csv("limmpca/test/data/Candies.tsv", sep="\t")
exp_design = data.iloc[:, :2]
X = data.iloc[:, 2:].values
Xz = zscore(X)


pca = PCA(svd_solver='full').fit(Xz)
pc = pca.components_
scores = pca.transform(Xz)

for i in range(8):
    data[f"PC{i + 1}"] = scores[:, i]
# %%
# loading plot
plt.figure()
plt.plot(pc[0, :], pc[1, :], ".")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
for i in range(9):
    plt.annotate(data.iloc[:, 2:].columns[i],
                 (pc[0, i], pc[1, i]), )

plt.figure()
sns.scatterplot(x="PC1", y="PC2", hue="Candies", data=data)
# %%
models = {
    "full_model": {'formula': '~ C(Candies)',
    'groups': 'group',
    're_formula': '0',
    'vc_formula': {"Judges": "0 + C(Judges)",
                   "CandiesJudges": "0 + C(CandiesJudges)"}
                   },
    "Candies": {'formula': '~ 1',
    'groups': 'group',
    're_formula': '0',
    'vc_formula': {"Judges": "0 + C(Judges)",
                   "CandiesJudges": "0 + C(CandiesJudges)"}
                   },
    "Judges": {'formula': '~ C(Candies)',
    'groups': 'group',
    're_formula': '0',
    'vc_formula': {
                   "CandiesJudges": "0 + C(CandiesJudges)"}
                   },
    "CandiesJudges": {'formula': '~ C(Candies)',
    'groups': 'group',
    're_formula': '0',
    'vc_formula': {"Judges": "0 + C(Judges)"}
                   },
}
# %%
exp_design['group'] = 1
count = 0
exp_design['CandiesJudges'] = 0
for i in range(165):
    if np.mod(i, 3) == 0:
        count += 1
    exp_design['CandiesJudges'][i] = count

h1_models = ParallelMixedModel(models["full_model"])
h1_models.fit(exp_design, scores[:, :8])
# %%
