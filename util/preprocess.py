import os
from _bisect import bisect_left
from bisect import bisect_right

import scipy.io as sio
import numpy as np


def preprocess(path, electrode_list, frequency_component, export_to_csv):
    """
    Args:
        path:
        electrode_list:
        export_to_csv:
        frequency_component:

    Returns:

    """
    coherence_spectra = []
    comprehension_scores = []
    frequencies = []

    for mat_file in sorted(os.listdir(path)):
        mat_contents = sio.loadmat(os.sep.join([path, mat_file]))
        for i in range(0, 2):
            electrodes = []
            cluster_indexes = []
            session_i = mat_contents['conn'][0, i][0]

            label_struct = session_i['labelcmb'][0]
            for label in label_struct:
                electrodes = np.append(electrodes, label[0])

            for el in electrode_list:
                cluster_indexes.append(np.where(electrodes == el)[0][0])

            coherence_spectra.append(session_i['cohspctrm'][0][:, cluster_indexes, :])

            for j in range(0, len(session_i['trialinfo'][0])):
                comprehension_scores.append(session_i['trialinfo'][0][j][2])

            if len(frequencies) == 0:
                frequencies = session_i['freq']
                frequencies = np.concatenate([x for x in frequencies]).flatten()

    coherence_spectra = np.concatenate([x for x in coherence_spectra])
    comprehension_scores = np.array(comprehension_scores)

    if frequency_component is not None:
        if frequency_component < 1 or frequency_component > 50:
            raise Exception("Select a frequency between 1 and 50 Hz!")
        else:
            lower = frequency_component - 1
            upper = frequency_component + 1

            idx_lower = bisect_right(frequencies, lower)
            idx_upper = bisect_left(frequencies, upper)

            coherence_spectra = coherence_spectra[:, :, [x for x in range(idx_lower, idx_upper + 1)]]

        coherence_spectra = (coherence_spectra - np.mean(coherence_spectra)) / np.std(coherence_spectra)

    if export_to_csv:
        coherence_spectra.tofile(path + '\\..\\coherence_spectra.csv', sep=',')
        comprehension_scores.tofile(path + '\\..\\comprehension_scores.csv', sep=',')
        frequencies.tofile(path + '\\..\\frequencies.csv', sep=',')

    return coherence_spectra, comprehension_scores


def get_electrodes(path):
    """
    Args:
        path:

    Returns:
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