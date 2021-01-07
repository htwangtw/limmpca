import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
from patsy import dmatrices


import seaborn as sns
import matplotlib.pyplot as plt

#%%
data = combine_data(exp_design, scores)
models = {"cand1": {'formula': '~ 1 + C(nBack) + interval',
    'groups': 'RIDNO',
    're_formula': '1',
    'vcf': {'session': '0 + C(session)'}},
    "cand2": {'formula': '~ 1 + C(nBack) * interval',
    'groups': 'RIDNO',
    're_formula': '1',
    'vcf': {'session': '0 + C(session)'}},
    }
i = 1
for key, model in models.items():
    formula = f"factor_{i}"  + model["formula"]

    mlm = smf.mixedlm(formula,
                        data,
                        groups=model["groups"],
                        re_formula=model["re_formula"],
                        vc_formula=model["vcf"])
    mlmf = mlm.fit(reml=True, method='cg')
    print(mlmf.summary())

    fe_params = pd.DataFrame(mlmf.fe_params,columns=['LMM'])
    random_effects = pd.DataFrame(mlmf.random_effects)
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
    fitted = mlmf.fittedvalues
    print("The MSE is " + str(np.mean(np.square(Y.flatten()-fitted))))
    ax3.plot(fitted,lw=1,alpha=.5)
    ax3.legend(loc=0)
    #plt.ylim([0,5])
    plt.show()
