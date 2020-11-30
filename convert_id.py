"""
Script for converting the old data log to what resemble 
to data used in Turnbull et al 2019 NIMG with some added info:
1. Previous trial type
2. correct / incorrect
"""

#%% setup
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

master = pd.read_csv("data/CS_Tasks_withEFF.csv")
id_ref = master[["IDNO", "RNO"]]

path_data = Path("data/original/")
#%% add RIDNO
subject_logs = []
date_pattern = "%Y_%b_%d_%H%M"
# rename files and add session info, R number
for _, subject in id_ref.iterrows():
    ref_idno, ref_rno = subject["IDNO"], subject["RNO"]

    # find files of the current subject
    files = list(path_data.glob(f"{ref_idno}_*.csv"))
    # strip date and time info and sort
    dates_list = [re.search(
                  r"\d{4}_[a-zA-Z]{3}_\d{2}_\d{4}", f.name)[0] 
                  for f in files]
    dates_list = sorted((datetime.strptime(d, date_pattern) \
                         for d in dates_list))

    # add session number and R number in the file
    for i, d in enumerate(dates_list):
        f = list(path_data.glob(
                 f"{ref_idno}_*_{d.strftime(date_pattern)}*.csv"))
        assert len(f) == 1

        new_log = pd.read_csv(f[0])
        if ref_idno < 500:  # the first cohort
            new_log = pd.read_csv(f[0], index_col=0)

        new_log["RIDNO"] = ref_rno
        new_log["Session"] = i + 1
        new_log = new_log.replace("Spontaneous", "Deliberate") # first cohort
        new_log = new_log.replace("Vague", "Detailed") # first cohort
        # create new file name by BIDS-ish rules
        new_fn = f"sub-{ref_rno}_ses-{i + 1}_task-ESnback_beh.tsv"
        # save new file
        new_log.to_csv(Path(f"data/task-ESnback/{new_fn}"), 
                       index=None, sep="\t")
        # keep the file
        subject_logs.append(new_log)

#%% summarise the time between a probe the closest event
master = []
for cur in subject_logs:
    MWQ_bool = (cur["mwType"] == "Focus")
    real_probe_idx = cur.index[MWQ_bool].tolist()
    if cur.loc[real_probe_idx[-1], "keyResp"] is np.nan:
        real_probe_idx = real_probe_idx[:-2]
    probes = []
    # save the accuracy of that event 
    # (switch would be logged as correct all the time)
    for i in real_probe_idx:
        probe = cur.loc[i : (i + 12), ["mwType", "keyResp"]]
        probe['mwType'] = 'MWQ_' + probe['mwType']
        probe = probe.set_index("mwType").T
        start_probe = cur.loc[i, "fixStart"] + cur.loc[i, "fixT"]
        nBack = cur.loc[i, "nBack"]
        stim = "NT"  # initiate
        stim_i = i
        # search for the closest event or when reach the start of exp
        while stim == "NT" and stim_i != 0: 
            stim_i -= 1
            stim = cur.loc[stim_i, "stimType"]

            start_stim = cur.loc[stim_i, "fixStart"] \
                      + cur.loc[stim_i, "fixT"]
            if stim == "MWQ":
                stim_i += 1
                start_stim = cur.loc[stim_i, "fixStart"]
            corr = cur.loc[stim_i, "respCORR"]
        if stim == "NT":
            stim = "task_start"
            print(cur["IDNO"][0])
        start_probe -= start_stim
        probe["corr"] = corr
        probe["interval"] = start_probe
        probe["prev_task"] = stim
        probe["nBack"] = nBack
        probes.append(probe)
    probes = pd.concat(probes).reset_index(drop=True)
    probes["RIDNO"] = cur["RIDNO"][0]
    probes["IDNO"] = cur["IDNO"][0]
    probes["session"] = cur["Session"][0]
    master.append(probes)

# save info with the question
all_files = pd.concat(master)
all_files.index.set_names = "idxES"
all_files.to_csv(Path("data/task-nbackES_probes_trial_interval.tsv"),
                 sep="\t") 