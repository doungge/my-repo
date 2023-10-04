import sys
# sys.path.insert(0,'/workspace/')
from general_processor import Utils
import numpy as np
import os

# channels = [["FC1", "FC2"],
#             ["FC3", "FC4"],
#             ["FC5", "FC6"],
#             ["C5", "C6"],
#             ["C3", "C4"],
#             ["C1", "C2"],
#             ["CP1", "CP2"],
#             ["CP3", "CP4"],
#             ["CP5", "CP6"]]

exclude = [89]
subjects = [n for n in np.arange(1, 110) if n not in exclude]

runs = [4, 6, 8, 10, 12, 14]

    
base_path = './data/eegmmidb/S%03d/S%03dR%02d.edf'
save_path = os.path.join('./pre_data/')
for sub in subjects:
    # x, y = Utils.epoch(Utils.select_channels
    #     (Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
    #     Utils.load_data(subjects=[sub], runs=runs, data_path=base_path)))), couple),
    #     exclude_base=False)
    x, y = Utils.epoch(Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
        Utils.load_data(subjects=[sub], runs=runs, data_path=base_path)))), exclude_base=True)    
    np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
    np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)
