import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore

import os

def test_data():
    # load data
    data = pd.read_csv("data/Candies.tsv", sep="\t")

    # PCA
    X = data.iloc[:, 2:].values
    Xz = zscore(X)
    pca_scores = PCA(svd_solver='full').fit_transform(Xz)

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

    # generate experiment design
    exp_design = data.iloc[:, :2]
    exp_design['group'] = 1
    exp_design['CandiesJudges'] = 0
    count = 0
    for i in range(165):
        if np.mod(i, 3) == 0:
            count += 1
        exp_design['CandiesJudges'][i] = count

    return exp_design, pca_scores, models