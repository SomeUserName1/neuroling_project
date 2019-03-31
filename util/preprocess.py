import os
from _bisect import bisect_left
from bisect import bisect_right

import scipy.io as sio
import numpy as np


def preprocess(path, electrode_list, frequency_component):
    """
    1. reads the mat files
    2. extract the coherence spectra and comprehension and frequencies scores
    3. restricts the coherence spectra to only the specified electrodes and frequencies and to be z-scored
    Args:
        path: location of the .mat files
        electrode_list: list of electrodes to be used for classification
        frequency_component: the frequency component to be used (+-1)

    Returns:
        standardized and filtered coherence spectrum, comprehension scores
    """
    coherence_spectra = []
    comprehension_scores = []
    frequencies = []

    # extract the data set from .mat files
    for mat_file in sorted(os.listdir(path)):
        mat_contents = sio.loadmat(os.sep.join([path, mat_file]))
        for i in range(0, 2):
            electrodes = []
            cluster_indexes = []
            session_i = mat_contents['conn'][0, i][0]

            # get the electrode names
            label_struct = session_i['labelcmb'][0]
            for label in label_struct:
                electrodes = np.append(electrodes, label[0])

            # filter for the electrodes to be used
            for el in electrode_list:
                cluster_indexes.append(np.where(electrodes == el)[0][0])

            # get the coherence spectra and slice to use the specified electrodes only
            coherence_spectra.append(session_i['cohspctrm'][0][:, cluster_indexes, :])

            # get the comprehension scores
            for j in range(0, len(session_i['trialinfo'][0])):
                comprehension_scores.append(session_i['trialinfo'][0][j][2])

            # get the available frequencies
            if len(frequencies) == 0:
                frequencies = session_i['freq']
                frequencies = np.concatenate([x for x in frequencies]).flatten()

    # adjust the shape to be (no_trials, no_electrodes, no_frequencies)
    coherence_spectra = np.concatenate([x for x in coherence_spectra])
    comprehension_scores = np.array(comprehension_scores)

    # filter for the desired frequencies using binary search
    if frequency_component is not None:
        if frequency_component < 1 or frequency_component > 50:
            raise Exception("Select a frequency between 1 and 50 Hz!")
        else:
            lower = frequency_component - 1
            upper = frequency_component + 1

            idx_lower = bisect_right(frequencies, lower)
            idx_upper = bisect_left(frequencies, upper)

            # slice the coherence spectra according to the frequencies to be used
            coherence_spectra = coherence_spectra[:, :, [x for x in range(idx_lower, idx_upper + 1)]]

        # standardize the coherence spectra
        coherence_spectra = (coherence_spectra - np.mean(coherence_spectra)) / np.std(coherence_spectra)
    print(coherence_spectra.shape)
    return coherence_spectra, comprehension_scores


def get_electrodes(path):
    """
    Lists all available electrodes, from a .mat file with structure as in the given dataset
    Args:
        path: where the .mat data is

    Returns:
        a list of strings containing the electrode names
    """
    all_electrodes = []
    for mat_file in sorted(os.listdir(path)):
        mat_contents = sio.loadmat(os.sep.join([path, mat_file]))
        for i in range(0, 2):
            electrodes = []
            session_i = mat_contents['conn'][0, i][0]

            label_struct = session_i['labelcmb'][0]
            for label in label_struct:
                electrodes = np.append(electrodes, label[0])
            all_electrodes.append(electrodes)
    return all_electrodes


def mean_cov(spectra, scores):
    covs = []
    for idx in range(0, spectra.shape[1]):
        new_spec = spectra[:, idx, 2]

        stack = np.vstack((new_spec,scores))
        covs.append(np.cov(stack))
    covs = np.array(covs)

    return np.mean(covs, axis=0)