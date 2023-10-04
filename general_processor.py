
import numpy as np
import os
from mne.io import read_raw_edf, concatenate_raws
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.epochs import Epochs
import mne
from typing import List
import wget
import sys
from sklearn.preprocessing import minmax_scale
from mne.io import BaseRaw
import torch

channels = [["FC1", "FC2"],
            ["FC3", "FC4"],
            ["FC5", "FC6"],
            ["C5", "C6"],
            ["C3", "C4"],
            ["C1", "C2"],
            ["CP1", "CP2"],
            ["CP3", "CP4"],
            ["CP5", "CP6"]]


class Utils:
    """
    A static class that contains all the functions to generate the dataset and other
    useful functionality
    """
    combinations = {"a": [["FC1", "FC2"],
                          ["FC3", "FC4"],
                          ["FC5", "FC6"]],

                    "b": [["C5", "C6"],
                          ["C3", "C4"],
                          ["C1", "C2"]],

                    "c": [["CP1", "CP2"],
                          ["CP3", "CP4"],
                          ["CP5", "CP6"]],

                    "d": [["FC3", "FC4"],
                          ["C5", "C6"],
                          ["C3", "C4"],
                          ["C1", "C2"],
                          ["CP3", "CP4"]],

                    "e": [["FC1", "FC2"],
                          ["FC3", "FC4"],
                          ["C3", "C4"],
                          ["C1", "C2"],
                          ["CP1", "CP2"],
                          ["CP3", "CP4"]],

                    "f": [["FC1", "FC2"],
                          ["FC3", "FC4"],
                          ["FC5", "FC6"],
                          ["C5", "C6"],
                          ["C3", "C4"],
                          ["C1", "C2"],
                          ["CP1", "CP2"],
                          ["CP3", "CP4"],
                          ["CP5", "CP6"]]}

    @staticmethod
    def download_data(save_path: str = os.getcwd()) -> str:
        """
        This create a new folder data and download the necessary files
        WARNING: The physionet server is super-slow
        :save_path: data are saved here
        :return: the path
        """
        def bar_progress(current, total, width=80):
            progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()

        data_url = "https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip"
        data_path = os.path.join(save_path, "eegbci")
        try:
            os.makedirs(data_path)
        except:
            raise Exception("The folder alredy exists")

        wget.download(data_url, os.path.join(data_path, "eegbci.zip"), bar=bar_progress)
        return data_path

    @staticmethod
    def load_data(subjects: List, runs: List, data_path: str) -> List[List[BaseRaw]]:
        """
        Given a list of subjects, a list of runs, and the database path. This function iterates
        over each subject, and subsequently over each run, loads the runs into memory, modifies
        the labels and returns a list of runs for each subject.
        :param subjects: List, list of subjects
        :param runs: List, list of runs
        :param data_path: str, the source path
        :return: List[List[BaseRaw]]
        """
        all_subject_list = []
        subjects = [str(s) for s in subjects]
        runs = [str(r) for r in runs]
        task2 = [4, 8, 12]
        task4 = [6, 10, 14]
        for sub in subjects:
            # if len(sub) == 1:
            #     sub_name = "S"+"00"+sub
            # elif len(sub) == 2:
            #     sub_name = "S"+"0"+sub
            # else:
            #     sub_name = "S"+sub
            # path = data_path % (sub, sub, current_run)
            # sub_folder = os.path.join(data_path, sub_name)
            single_subject_run = []
            for run in runs:
                # if len(run) == 1:
                #     path_run = os.path.join(sub_folder, sub_name+"R"+"0"+run+".edf")
                # else:
                #     path_run = os.path.join(sub_folder, sub_name+"R"+ run +".edf")
                path = data_path % (int(sub), int(sub), int(run))
                # print(path) 
                raw_run = read_raw_edf(path, preload=True)
                len_run = np.sum(raw_run._annotations.duration)
                if len_run > 124:
                    raw_run.crop(tmax=124)

                """
                B indicates baseline
                L indicates motor imagination of opening and closing left fist;
                R indicates motor imagination of opening and closing right fist;
                LR indicates motor imagination of opening and closing both fists;
                F indicates motor imagination of opening and closing both feet.
                """

                if int(run) in task2:
                    for index, an in enumerate(raw_run.annotations.description):
                        if an == "T0":
                            raw_run.annotations.description[index] = "B"
                        if an == "T1":
                            raw_run.annotations.description[index] = "L"
                        if an == "T2":
                            raw_run.annotations.description[index] = "R"
                if int(run) in task4:
                    for index, an in enumerate(raw_run.annotations.description):
                        if an == "T0":
                            raw_run.annotations.description[index] = "B"
                        if an == "T1":
                            raw_run.annotations.description[index] = "LR"
                        if an == "T2":
                            raw_run.annotations.description[index] = "F"
                single_subject_run.append(raw_run)
            all_subject_list.append(single_subject_run)
        return all_subject_list

    @staticmethod
    def concatenate_runs(list_runs: List[List[BaseRaw]]) -> List[BaseRaw]:
        """
        Concatenate a list of runs
        :param list_runs: List[List[BaseRaw]],  list of raw
        :return: List[BaseRaw], list of concatenate raw
        """
        raw_conc_list = []
        for subj in list_runs:
            raw_conc = concatenate_raws(subj)
            raw_conc_list.append(raw_conc)
        return raw_conc_list

    @staticmethod
    def del_annotations(list_of_subraw: List[BaseRaw]) -> List[BaseRaw]:
        """
        Delete "BAD boundary" and "EDGE boundary" from raws
        :param list_of_subraw: list of raw
        :return: list of raw
        """
        list_raw = []
        for subj in list_of_subraw:
            indexes = []
            for index, value in enumerate(subj.annotations.description):
                if value == "BAD boundary" or value == "EDGE boundary":
                    indexes.append(index)
            subj.annotations.delete(indexes)
            list_raw.append(subj)
        return list_raw

    @staticmethod
    def eeg_settings(raws:  List[BaseRaw]) -> List[BaseRaw]:
        """
        Standardize montage of the raws
        :param raws: List[BaseRaw] list of raws
        :return: List[BaseRaw] list of standardize raws
        """
        raw_setted = []
        for subj in raws:
            eegbci.standardize(subj)
            montage = make_standard_montage('standard_1005')
            subj.set_montage(montage)
            raw_setted.append(subj)

        return raw_setted

    @staticmethod
    def filtering(list_of_raws: List[BaseRaw]) -> List[BaseRaw]:
        """
        Perform a band_pass and a notch filtering on raws, UNUSED!
        :param list_of_raws:  list of raws
        :return: list of filtered raws
        """
        raw_filtered = []
        for subj in list_of_raws:
            if subj.info["sfreq"] == 160.0:
                subj.filter(1.0, 79.0, fir_design='firwin', skip_by_annotation='edge')
                subj.notch_filter(freqs=60)
                raw_filtered.append(subj)
            else:
                subj.filter(1.0, (subj.info["sfreq"] / 2) - 1, fir_design='firwin',
                            skip_by_annotation='edge')
                subj.notch_filter(freqs=60)
                raw_filtered.append(subj)

        return raw_filtered

    @staticmethod
    def select_channels(raws: List[BaseRaw], ch_list: List ) -> List[BaseRaw]:
        """
        Slice channels
        :raw: List[BaseRaw], List of Raw EEG data
        :ch_list: List
        :return: List[BaseRaw]
        """
        s_list = []
        for raw in raws:
            s_list.append(raw.pick_channels(ch_list))

        return s_list

    @staticmethod
    def epoch(raws: List[BaseRaw], exclude_base: bool =False,
              tmin: int =0, tmax: int =4) -> (np.ndarray, List):
        """
        Split the original BaseRaw into numpy epochs
        :param raws: List[BaseRaw]
        :param exclude_base: bool, If True exclude baseline
        :param tmin: int, Onset
        :param tmax: int, Offset
        :return: np.ndarray (Raw eeg datas in numpy format) List (a List of strings)
        """
        xs = list()
        ys = list()
        
        for raw in raws:
            if exclude_base:
                event_id = dict(F=2, L=3, LR=4, R=5)
            else:
                event_id = dict(B=1, F=2, L=3, LR=4, R=5)
            tmin, tmax = tmin, tmax
            
            events, _ = mne.events_from_annotations(raw, event_id=event_id)
            print("evenet : ", events)
            picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                                   exclude='bads')
            epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                            baseline=None, preload=True)
            
            y = list()
            for index, data in enumerate(epochs):
                y.append(epochs[index]._name)

            xs.append(np.array([epoch for epoch in epochs]))
            # print("xs : " ,np.array(xs).shape)
            ys.append(y)
            # print("ys : ", ys)
        return np.concatenate(tuple(xs), axis=0), [item for sublist in ys for item in sublist]

    @staticmethod
    def cut_width(data):
        
        # new_data = np.zeros((data.shape[0], data.shape[1], data.shape[2] - 1))
        new_data = np.zeros((84, data.shape[1], data.shape[2] - 1))

        for index, line in enumerate(data[:84,:,:]):
            new_data[index] = line[:, : -1]
        return new_data
    
    def cut_widthy(data):
        new_data = np.zeros(84)
        mapping = {'L': 1, 'R': 2, 'F': 3, 'LR': 4}
        for index, line in enumerate(data[:84]):
            new_data[index] = mapping[line]
        return new_data
    
    @staticmethod
    def load_sub_by_sub(subjects, data_path, name_single_sub):
        xs = list()
        ys = list()
        for sub in subjects:
            xs.append(Utils.cut_width(np.load(os.path.join(data_path, "x" + name_single_sub + str(sub) + ".npy"))))
            # ys.append(np.load(os.path.join(data_path, "y" + name_single_sub + str(sub) + ".npy")))
            ys.append(Utils.cut_widthy(np.load(os.path.join(data_path, "y" + name_single_sub + str(sub) + ".npy"))))
            # data = (np.load(os.path.join(data_path, "y" + name_single_sub + str(sub) + ".npy")))
            # print(ys)
            # ys.append(data[:84])
            
        return xs, ys
    
    @staticmethod
    def scale_sub_by_sub(xs, ys):
        for sub_x, sub_y, sub_index in zip(xs, ys, range(len(xs))):
            for sample_index, x_data in zip(range(sub_x.shape[0]), sub_x):
                xs[sub_index][sample_index] = minmax_scale(x_data, axis=1)

        return xs, ys


    @staticmethod
    # def to_one_hot(y, by_sub=False):
    #     if by_sub:
    #         new_array = np.array(["nan" for nan in range(len(y))])
    #         for index, label in enumerate(y):
    #             new_array[index] = ''.join([i for i in label if not i.isdigit()])
    #     else:
    #         new_array = y.copy()
    #     total_labels = np.unique(new_array)
    #     mapping = {}
    #     for x in range(len(total_labels)):
    #         mapping[total_labels[x]] = x
    #     print(new_array.shape())
    #     for x in range(new_array.shape[1]):
    #         for y in range(new_array.shape[2]):
    #             new_array[0,x,y] = mapping[new_array[0,x,y]]
    #     new_array = np.array(new_array, dtype = float)
    #     # new_array = torch.Tensor(new_array)
        
        
    #     # return tf.keras.utils.to_categorical(new_array)
    #     return new_array
    def to_one_hot(data):
        num_classes = len(np.unique(data))
        one_hot_encoded = np.zeros((len(data), num_classes))
        for i in range(len(data)):
            one_hot_encoded[i, int(data[i]) - 1] = 1
        
        return one_hot_encoded

    @staticmethod
    def train_test_split(x, y, perc):
        from numpy.random import default_rng
        rng = default_rng()
        test_x = list()
        train_x = list()
        train_y = list()
        test_y = list()
        for sub_x, sub_y in zip(x, y):
            how_many = int(len(sub_x) * perc)
            indexes = np.arange(0, len(sub_x))
            choices = rng.choice(indexes, how_many, replace=False)
            for sample_x, sample_y, index in zip(sub_x, sub_y, range(len(sub_x))):
                if index in choices:
                    test_x.append(sub_x[index])
                    test_y.append(sub_y[index])
                else:
                    train_x.append(sub_x[index])
                    train_y.append(sub_y[index])
        return np.dstack(tuple(train_x)), np.dstack(tuple(test_x)), np.array(train_y), np.array(test_y)

    @staticmethod
    def load(subjects, base_path):
        data_x = list()
        data_y = list()

       
        data_path = os.path.join(base_path)
        sub_name = "_sub_"
        xs, ys = Utils.load_sub_by_sub(subjects, data_path, sub_name)
        data_x.append(np.concatenate(xs))
        data_y.append(np.concatenate(ys))
        # data_x.append(xs)
        # data_y.append(ys)
        # print(torch.Tensor(data_x).size())
        return np.concatenate(data_x), np.concatenate(data_y)
        # return np.array(data_x), np.array(data_y)

if __name__ == "__main__":
    # subject = range(1, 110)
    # runs = [4,6,8,10,12,14]
    # base_path = './data/eegmmidb/S%03d/S%03dR%02d.edf'

    # a = Utils.load_data(subjects = subject, runs = runs, data_path = base_path)
    # print(dir(a))
    pass