import scipy.io as sio
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report

from models import EEGLangComprehensionNAS
import argparse
import os
import numpy as np
import time


def parse_mat_files(path, export_to_csv):
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

            for el in ['AFZ', 'C2', 'C4', 'CP4', 'CP6', 'F1']:
                cluster_indexes.append(np.where(electrodes == el)[0][0])

            coherence_spectra.append(session_i['cohspctrm'][0][:, cluster_indexes, :])

            for j in range(0, len(session_i['trialinfo'][0])):
                comprehension_scores.append(session_i['trialinfo'][0][j][2])

            if len(frequencies) == 0:
                frequencies = session_i['freq']
                frequencies = np.concatenate([x for x in frequencies]).flatten()

    coherence_spectra = np.concatenate([x for x in coherence_spectra])
    comprehension_scores = np.array(comprehension_scores)

    if export_to_csv:
        coherence_spectra.tofile(path + '\\..\\coherence_spectra.csv', sep=',')
        comprehension_scores.tofile(path + '\\..\\comprehension_scores.csv', sep=',')
        frequencies.tofile(path + '\\..\\frequencies.csv', sep=',')

    return coherence_spectra, comprehension_scores, frequencies


def main(data_path):
    start = time.time()
    spectra, scores, frequencies = parse_mat_files(data_path, False)
    end = time.time()
    print("Importing data from mat files finished! Took %.3f s" % (end - start))

    for i in [5, 10]:
        classifier_i = EEGLangComprehensionNAS.EEGLangComprehensionNAS(False, frequencies, i)  # , verbose=True)
        start = time.time()
        preprocessed_x = classifier_i.preprocess(spectra)
        end = time.time()
        print("Pre-processing finished! Took %.3f s" % (end - start))

        start = time.time()
        classifier_i.fit(preprocessed_x, scores, time_limit=60 * 60)
        end = time.time()
        print("Model search finished! Took %.3f s" % (end - start))

        kf = KFold(n_splits=2)
        for train_index, test_index in kf.split(spectra):
            train_x, test_x = spectra[train_index], spectra[test_index]
            train_y, test_y = scores[train_index], scores[test_index]
            classifier_i.final_fit(train_x, train_y, test_x, test_y, trainer_args={'max_no_improvement_num': 30}, retrain=True)
            print(classifier_i.evaluate(test_x, test_y))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO final interface
    parser.add_argument("--data_path",
                        default='C:\\Users\\Fabi\\ownCloud\\workspace\\uni\\7\\neuroling\\neuroling_project\\data\\v1',
                        # default='data/v1',
                        type=str)

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise Exception("Data set path given does not exists")

    main(args.data_path)